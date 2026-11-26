"""One-thread, no-name ablation of the public notebook's numerical cluster encoding.

On this Mac run DYLD_LIBRARY_PATH=.venv-tabpfn/lib/python3.12/site-packages/torch/lib
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python lopez_numeric_clusters_n26.py
(one shell command). Only our signals and locally generated predictions are read.
"""
import os
for limit in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[limit]='1'

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from full_roster_booking_validation_n23 import ROOT, data
from nori_booking_regression_n26 import FEES
from paired_mechanisms_on_retained_orders_n26 import tally

OUT=ROOT/'artifacts/nov26_lopez_numeric_clusters'
REPORT=ROOT/'reports/nov23_matched_learning_curve/lopez_numeric_clusters.json'
MODES=['plain','cluster','cluster_woe']
WEIGHTS=np.array([.4766,.2901,.2333])
REV='7ed4f703bc57e3e32784965f02416d89d8fe1d3d'
SRC=f'https://github.com/jclopezsDS/kaggle-spaceship-titanic/blob/{REV}/notebooks/05_titanic_spaceship_prediction.ipynb'


def matrices(frame):
    cabin=frame.Cabin.str.split('/',expand=True)
    group=frame.PassengerId.str.split('_').str[0]
    fee=frame[FEES]
    num=pd.concat([frame.Age,pd.to_numeric(cabin[1]).rename('CabinNumber'),fee,
                   fee[['RoomService','Spa','VRDeck']].sum(axis=1,min_count=1).rename('Expenditure'),
                   fee.sum(axis=1,min_count=1).rename('TotalExp')],axis=1)
    cat=pd.DataFrame(dict(origin=frame.HomePlanet,destination=frame.Destination,deck=cabin[0],side=cabin[2],
                          cryosleep=frame.CryoSleep,vip=frame.VIP)).astype('string').fillna('?')
    size=group.map(group.value_counts()).to_numpy()
    plain=np.column_stack([pd.get_dummies(cat,dtype=float),num,num.isna(),np.log1p(size),
                           fee.gt(0).sum(axis=1),fee.div(1+num.TotalExp,axis=0)])
    return plain.astype(np.float32),num.to_numpy(),num.columns.tolist()


def encode(values,t,q,g,y):
    # The source reconstructs each eligible feature from its rejected transforms.
    # This controlled variant always uses the same six label-free transforms.
    left,right=values[t].copy(),values[q].copy()
    fill=np.nanmedian(left,axis=0)
    left=np.where(np.isfinite(left),left,fill)
    right=np.where(np.isfinite(right),right,fill)
    codes_t,codes_q=[],[]
    for j in range(left.shape[1]):
        a,b=left[:,j],right[:,j]
        ref=np.sort(a)
        def lifted(z):
            u=np.maximum(0,(z-ref[0])/max(ref[-1]-ref[0],1.))
            ranks=(np.searchsorted(ref,z,'left')+np.searchsorted(ref,z,'right'))/(2*len(ref))
            return np.column_stack([z,np.log1p(u),np.sqrt(u),u**.25,u**.1,ranks])
        scaler=StandardScaler().fit(lifted(a))
        at,bt=[scaler.transform(lifted(z)) for z in [a,b]]
        # 25 is the notebook's explicit fallback if the Calinski-Harabasz elbow is absent.
        m=KMeans(n_clusters=min(25,len(np.unique(a))),n_init=1,random_state=42).fit(at)
        codes_t.append(m.labels_)
        codes_q.append(m.predict(bt))
    clusters_t=np.column_stack(codes_t)
    clusters_q=np.column_stack(codes_q)
    folds=list(GroupKFold(5,shuffle=True,random_state=42).split(t,groups=g[t]))
    woe_t=np.empty_like(clusters_t,float)
    woe_q=np.empty_like(clusters_q,float)
    dummies_t,dummies_q=[],[]
    for j in range(clusters_t.shape[1]):
        a,b=clusters_t[:,j],clusters_q[:,j]
        k=int(a.max()+1)
        labels=np.arange(k)
        # The original drops the least common dummy. Use training-only counts.
        keep=labels[labels!=np.bincount(a,minlength=k).argmin()]
        dummies_t.append(a[:,None]==keep)
        dummies_q.append(b[:,None]==keep)
        def log_ratio(bank):
            count=np.bincount(a[bank],minlength=k)
            pos=np.bincount(a[bank],weights=y[t[bank]],minlength=k)
            return np.log((pos+.5)/(count-pos+.5))
        for train,test in folds:
            assert not np.isin(g[t[train]],g[t[test]]).any()
            woe_t[test,j]=log_ratio(train)[a[test]]
        woe_q[:,j]=log_ratio(np.arange(len(t)))[b]
    return [(np.column_stack(dummies_t),np.column_stack(dummies_q)),(woe_t,woe_q)]


def trees(tx,qx,labels):
    models=[
        CatBoostClassifier(iterations=420,depth=6,learning_rate=.055,random_seed=133,
                           thread_count=1,allow_writing_files=False,verbose=False),
        LGBMClassifier(n_estimators=320,num_leaves=17,min_child_samples=30,learning_rate=.045,
                       verbosity=-1,n_jobs=1,random_state=133),
        XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.055,min_child_weight=5,
                      n_jobs=1,tree_method='hist',random_state=133)]
    members=[]
    for model in models:
        model.fit(tx,labels)
        members.append(model.predict_proba(qx)[:,1])
    return np.column_stack(members)


def run():
    OUT.mkdir(exist_ok=True)
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    frame=pd.concat([raw,test],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    ids=frame.PassengerId.to_numpy(str)
    assert not np.isin(g,groups[len(y):]).any()
    plain,num,num_names=matrices(frame)
    oof=np.full((len(y),3),np.nan)
    for k in range(9):
        t,q=(np.flatnonzero(f!=k),np.flatnonzero(f==k)) if k<8 else (np.arange(len(y)),np.arange(len(y),len(frame)))
        assert not np.isin(groups[t],groups[q]).any()
        cache=OUT/f'fold{k}.npz'
        if not cache.exists():
            extra=encode(num,t,q,groups,y)
            tx,qx=plain[t],plain[q]
            members=[]
            for j,mode in enumerate(MODES):
                if j:
                    tx=np.column_stack([tx,extra[j-1][0]]).astype(np.float32)
                    qx=np.column_stack([qx,extra[j-1][1]]).astype(np.float32)
                members.append(trees(tx,qx,y[t]))
            member_scores=np.stack(members,axis=1)
            scores=member_scores@WEIGHTS
            temp=cache.with_suffix('.tmp')
            with temp.open('wb') as out:
                np.savez_compressed(out,t=t,q=q,labels=y[t],ids=ids[q],p=scores,members=member_scores)
            temp.replace(cache)
        with np.load(cache) as z:
            for key,expected in [('t',t),('q',q),('labels',y[t]),('ids',ids[q])]:
                np.testing.assert_array_equal(z[key],expected)
            np.testing.assert_allclose(z['members']@WEIGHTS,z['p'])
            assert np.isfinite(z['p']).all()
            if k<8:
                oof[q]=z['p']
                print(k,((z['p']>=.5)==y[q,None]).sum(0).tolist(),flush=True)
            else:
                target=z['p']
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        for key,expected in [('passenger',ids[:len(y)]),('y',y),('fold',f)]:
            np.testing.assert_array_equal(z[key],expected)
        base,eligible,known,fresh=[z[k] for k in ['reference161','eligible','known','fresh']]
        all_agree=((z['pairs']>=.5)==base[:,None,None]).all((1,2))&((z['original_panel']>=.5)==base[:,None]).all(1)
    txt=pd.read_csv(ROOT/'artifacts/proposal_n26_joint_fare_foundations/submission.csv')
    np.testing.assert_array_equal(test.PassengerId,txt.PassengerId)
    test_base=txt.Transported.to_numpy(bool)
    wrong=base!=y
    assert np.count_nonzero(eligible&wrong)==855 and np.count_nonzero(eligible&all_agree&wrong)==266
    labels=oof>=.5
    scopes=dict(known=known,fresh=known&fresh,retained855=eligible,silent266=eligible&all_agree)
    rules=dict(plain=labels[:,0]!=base,cluster=labels[:,1]!=base,cluster_woe=labels[:,2]!=base,
               new_cluster=(labels[:,0]==base)&(labels[:,1]!=base),
               new_woe=(labels[:,0]==base)&(labels[:,1]==base)&(labels[:,2]!=base),
               both_cluster_models=(labels[:,0]==base)&(labels[:,1]!=base)&(labels[:,2]!=base))
    test_rules=dict(plain=target[:,0]>=.5,cluster=target[:,1]>=.5,cluster_woe=target[:,2]>=.5)
    t0,t1,t2=list(test_rules.values())
    test_acts={name:v!=test_base for name,v in test_rules.items()}
    test_acts.update(new_cluster=(t0==test_base)&(t1!=test_base),new_woe=(t0==test_base)&(t1==test_base)&(t2!=test_base),
                     both_cluster_models=(t0==test_base)&(t1!=test_base)&(t2!=test_base))
    report=dict(source=SRC,code_cells=[40,43,48,51,60,71,72,73,76],weights=WEIGHTS.tolist(),clustered_fields=num_names,
        related_local_controls=['broad_chained_tree_ensemble_n23.py: posterior numeric imputation and three-tree voting',
                                'full_roster_permutation_selection_n24.py: 191-feature permutation selection'],
        published_claim='82.02% per author README; not a score of these models',
        fidelity='Isolates numerical cluster OHE plus smoothed within-cluster outcome log odds from cell 60. Nine fields; six fixed transforms and published fallback K=25. Uses group-held target encoding, never target-encodes a training order with its own labels. Uses the local three-tree settings, single seed and shared one-hot base; keeps fold-median cluster fill, no surnames, no ExtraTrees imputation, Optuna or target-driven transform selection. Original notebook fits cluster outcome counts before CV, and downstream trees use stratified passenger folds.',
        original_imputation='Cell 40 uses joint unlabeled train+test, IterativeImputer with 100 ExtraTrees, max_depth=15, min_samples_split=5, max_iter=10, seed=42. Cell 48 defines Expenditure as RoomService+Spa+VRDeck; TotalExp includes all five. Cells 71-73 derive downstream hyperparameters through Optuna (100+80+40 trials per model) rather than specifying one numerical winner in the training function.',
        original_validation='The published run_optimized_nested_cv_ensemble_with_saving fits the three weights via 75 Optuna trials using the validation labels in each passenger-stratified fold, then scores the optimized weights on those same labels. Its reported CV has a different selection and split design.',
        standalone={mode:dict(correct=int((labels[:,j]==y).sum()),accuracy=float((labels[:,j]==y).mean()),
                           brier=float(((oof[:,j]-y)**2).mean())) for j,mode in enumerate(MODES)},
        paired={f'{a}_to_{b}':tally(labels[:,i]!=labels[:,j],labels[:,i]!=y,np.ones(len(y),bool),f)
                for i,a,j,b in [(0,'plain',1,'cluster'),(1,'cluster',2,'woe'),(0,'plain',2,'woe')]},
        scopes={s:dict(rows=int(m.sum()),errors=int((m&wrong).sum()),rules={mode:tally(a,wrong,m,f) for mode,a in rules.items()}) for s,m in scopes.items()},
        test=dict(rows=len(test),vs_161={mode:int(a.sum()) for mode,a in test_acts.items()},pairwise_edits=dict(plain_cluster=int((t0!=t1).sum()),cluster_woe=int((t1!=t2).sum()),plain_woe=int((t0!=t2).sum()))))
    np.savez_compressed(OUT/'aligned.npz',row=np.arange(len(y)),passenger=ids[:len(y)],fold=f,y=y,oof=oof,known=known,
                        reference161=base,test_ids=ids[len(y):],test_scores=target,test_reference161=test_base)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['standalone'],report['test'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
