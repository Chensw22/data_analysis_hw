"""Learn risk from paired deletions of manifest blocks on previously unseen bookings."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from archive_source_harm_veto_n28 import inputs,PARENTS
from repeated_archive_misses_n26 import ART,cohort,count
from export_budget_quarter_n23 import source
from price_implied_deck_n26 import FEES
from full_roster_booking_validation_n23 import ROOT
from kaggle_workflow import validate_submission,sha256
from relational_error_corrector import write

OUT=ART/'nov28_block_response_curvature'
NAME='proposal_n28_block_response_curvature'


def design(frame):
    def dummy(a):return pd.get_dummies(a.astype('string').fillna('?'),dtype=float).to_numpy()
    cabin=frame.Cabin.str.split('/',expand=True)
    group=frame.PassengerId.str[:4]
    amounts=frame[FEES]
    blocks=[np.c_[frame.Age,dummy(frame.HomePlanet),dummy(frame.Destination),dummy(frame.VIP)],
            np.c_[dummy(cabin[0]),pd.to_numeric(cabin[1]),dummy(cabin[2]),group.map(group.value_counts()),
                  pd.to_numeric(frame.PassengerId.str[5:]),frame.Cabin.map(frame.Cabin.value_counts())],
            dummy(frame.CryoSleep),np.c_[np.log1p(amounts),amounts.isna(),amounts.gt(0),
                np.log1p(amounts.sum(axis=1,min_count=1))]]
    return np.column_stack(blocks),np.split(np.arange(sum(b.shape[1] for b in blocks)),np.cumsum([b.shape[1] for b in blocks])[:-1])


def masked(x,blocks,code):
    out=x.copy()
    for bit,columns in enumerate(blocks):
        if code&(1<<bit):out[:,columns]=-150
    return out


def response(x,blocks,t,q,y):
    codes=[0,1,2,4,8,3,5,9,6,10,12]
    # Every pair of omitted blocks is seen in training; the missing-field sentinel is distinct from real NaN.
    rng=np.random.default_rng(28126)
    parts=[x[t]];labels=[y[t]];weights=[np.ones(len(t))]
    for shuffled in [rng.permutation(t),rng.permutation(t)]:
        for code,subset in zip(codes[1:],np.array_split(shuffled,10),strict=True):
            parts.append(masked(x[subset],blocks,code));labels.append(y[subset]);weights.append(np.full(len(subset),.5))
    model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=19,learning_rate=.065,
             l2_regularization=3,min_samples_leaf=30,early_stopping=False,random_state=28126)
    model.fit(np.vstack(parts),np.concatenate(labels),sample_weight=np.concatenate(weights))
    z=np.column_stack([model.decision_function(masked(x[q],blocks,k)) for k in codes])
    single=z[:,1:5]-z[:,[0]]
    # Predicted log-odds movement under paired omission beyond the two individual movements.
    interaction=np.column_stack([z[:,j+5]-z[:,0]-single[:,a]-single[:,b]
                                 for j,(a,b) in enumerate(combinations(range(4),2))])
    return np.c_[z[:,0],single,interaction]


def calibrated(z,t,q,y,g,mode,seed):
    baseline=z[:,[0]]
    design=z.copy()
    if mode=='plain':design=baseline
    elif mode=='shuffled':
        rng=np.random.default_rng(seed)
        edges=np.quantile(baseline[t,0],np.arange(1,10)/10)
        for rows in (t,q):
            cell=np.digitize(baseline[rows,0],edges)
            for c in np.unique(cell):
                ix=rows[cell==c]
                design[ix,1:]=design[rng.permutation(ix),1:]
    scaler=StandardScaler().fit(design[t])
    model=LogisticRegression(C=.12,max_iter=450)
    return model.fit(scaler.transform(design[t]),y[t]).decision_function(scaler.transform(design[q]))


def run():
    frame,y,g,f,ids,_,_,seven,versions=inputs();n=len(y)
    _,_,known,fresh=cohort(frame.iloc[:n],y,f);parent=versions['random'][:,0]
    qtest=np.arange(n,len(frame)); groups=frame.PassengerId.str[:4].to_numpy()
    soft,actual,digest=source(PARENTS['direction'],frame.iloc[n:])
    np.testing.assert_array_equal(parent[n:],actual)
    x,blocks=design(frame);wrong=parent[:n]!=y
    ordered=np.empty(n,int)
    for k,chunk in enumerate(np.array_split(np.unique(g),8)):ordered[np.isin(g,chunk)]=k
    same7=((seven>=.5)==parent[:,None]).all(1)
    margin=np.where(parent,1.,-1.)*np.maximum(.00001,abs(logit(np.median(seven[:,1:4],axis=1).clip(.003,.997))))
    report={};data={};OUT.mkdir(exist_ok=True)
    for tag,partition in [('random',f),('ordered',ordered)]:
        exact=(versions[tag]==parent[:,None]).all(1)
        cross=exact&same7
        assert exact[n:].sum()==4198
        scopes=dict(all=known,fresh=known&fresh,four=known&exact[:n],seven=known&same7[:n],
                    four_and_seven=known&cross[:n],outside=known&~cross[:n])
        z=np.empty((len(frame),11))
        for k in range(9):
            t=np.flatnonzero(partition!=k)
            q=np.flatnonzero(partition==k) if k<8 else qtest
            assert not np.isin(groups[t],groups[q]).any()
            path=OUT/(f'{tag}_{k}.npz' if k<8 else 'official_test.npz')
            if not path.exists():np.savez_compressed(path,t=t,q=q,y=y[t],response=response(x,blocks,t,q,y))
            with np.load(path) as a:
                for k0,v in [('t',t),('q',q),('y',y[t])]:np.testing.assert_array_equal(a[k0],v)
                z[q]=a['response']
        schemes=['plain','sensitivity','shuffled'];pooled=[]
        for mode in schemes:
            result=np.full(len(frame),np.nan);votes=[]
            for k in range(8):
                ids_fold=np.flatnonzero(partition==k)
                unique=np.unique(g[ids_fold])
                if tag=='random':unique=np.random.default_rng(726+k).permutation(unique)
                for j,subset in enumerate(np.array_split(unique,4)):
                    accept=np.isin(g[ids_fold],subset); t=ids_fold[~accept];q=ids_fold[accept]
                    assert not np.isin(g[t],g[q]).any()
                    result[q]=calibrated(z,t,q,y,g,mode,k*13+j)
                votes.append(calibrated(z,ids_fold,qtest,y,g,mode,k*13+11))
            result[n:]=np.mean(votes,axis=0);pooled.append(result)
        old,sens,sham=pooled
        opp=(sens>=0)!=parent
        flag=opp&((margin+sens-old>=0)!=parent)
        rules={'plain':(old>=0)!=parent,'sensitivity':opp,'paired_increment':flag,
               'intersection':flag&cross,'shuffled_increment':((sham>=0)!=parent)&((margin+sham-old>=0)!=parent)}
        report[tag]={'scope_sizes':{s:dict(rows=int(m.sum()),errors=int((m&wrong).sum())) for s,m in scopes.items()},'rules':{}}
        for name,v in rules.items():
            report[tag]['rules'][name]={s:count(v[:n],m,wrong) for s,m in scopes.items()}
            report[tag]['rules'][name].update(accuracy=float(np.mean((parent[:n]^v[:n])[known]==y[known])),test=int(v[n:].sum()),test_four_seven=int((v[n:]&cross[n:]).sum()))
        for j,mode in enumerate(schemes):
            risk=expit(np.where(parent[:n],-1,1)*pooled[j][:n])
            report[tag][mode+'_error_auc']={s:float(roc_auc_score(wrong[m],risk[m])) for s,m in scopes.items()}
        data[tag+'_scores']=np.column_stack(pooled);data[tag+'_flag']=flag;data[tag+'_intersection']=rules['intersection']
        print(tag,report[tag]['rules']['paired_increment'],flush=True)
    kept=data['random_flag']&data['ordered_flag']
    report['replicated']={s:count(kept[:n],m,wrong) for s,m in dict(all=known,fresh=known&fresh,intersection=known&same7[:n]&(versions['random'][:n]==parent[:n,None]).all(1)&(versions['ordered'][:n]==parent[:n,None]).all(1)).items()}
    report['test_both']=int(kept[n:].sum())
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,**data)
    # The matched-exposure export is evaluated separately.
    report['exported']=False
    write(ROOT/'reports/nov23_matched_learning_curve/block_response_curvature_n28.json',report)


if __name__=='__main__':
    with threadpool_limits(2):run()
