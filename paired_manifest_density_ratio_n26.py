"""Estimate outcome-specific registration/bill dependence by distinguishing intact and crossed records."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data
from full_roster_two_view_reliability_n24 import design
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART/'nov26_paired_manifest_density_ratio'
REPORT = ROOT/'reports/nov23_matched_learning_curve/paired_manifest_density_ratio.json'
BILLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def tree():
    return HistGradientBoostingClassifier(max_iter=125, max_leaf_nodes=15, min_samples_leaf=30,
        learning_rate=.065,l2_regularization=4.,early_stopping=False,random_state=729)


def odds(p):
    return logit(np.clip(p,.005,.995))


def donor_draw(ids,state,rng):
    donors=np.empty(len(ids),int)
    for key in np.unique(state[ids]):
        ix=state[ids]==key
        donors[ix]=rng.choice(ids[ix],int(ix.sum()),replace=True)
    return donors


def joined(a,b,i,j):
    return np.column_stack([a[i],b[j]])


def density_ratio(a,b,state,t,q,partition,seed):
    scores=[]
    discrimination=[]
    for label in (False,True):
        ids=t[partition[t]==label]
        rng=np.random.default_rng(seed+int(label))
        donors=[donor_draw(ids,state,rng) for _ in range(3)]
        actual=joined(a,b,ids,ids)
        crossed=np.vstack([joined(a,b,ids,j) for j in donors])
        x=np.vstack([actual,crossed])
        s=np.r_[np.ones(len(ids),bool),np.zeros(len(crossed),bool)]
        model=tree().fit(x,s,sample_weight=np.where(s,3.,1.))
        prob=model.predict_proba(joined(a,b,q,q))[:,1]
        scores.append(odds(prob))
        # Independent swaps within each held-out class are only used to evaluate the discriminator.
        held=q[q<len(partition)]
        held=held[partition[held]==label]
        if len(held)>1:
            j=donor_draw(held,state,rng)
            pair_scores=np.r_[model.predict_proba(joined(a,b,held,held))[:,1],
                              model.predict_proba(joined(a,b,held,j))[:,1]]
            discrimination.append(roc_auc_score(np.repeat([1,0],len(held)),pair_scores))
    return scores[1]-scores[0], discrimination


def fit(a,b,state,y,g,t,q,seed):
    assert not np.isin(g[t],g[q]).any()
    encoded=np.eye(6)[state]
    views=[np.column_stack([a,encoded]),np.column_stack([b,encoded]),joined(a,b,np.arange(len(a)),np.arange(len(a)))]
    pa,pb,direct=[tree().fit(x[t],y[t]).predict_proba(x[q])[:,1] for x in views]
    prior=(np.bincount(state[t],weights=y[t],minlength=6)+.5)/(np.bincount(state[t],minlength=6)+1)
    independent=odds(pa)+odds(pb)-odds(prior[state[q]])
    shuffled=y.copy()
    rng=np.random.default_rng(seed)
    # Shuffle only training outcomes. The matched-null AUC uses the corresponding held-out permutation.
    for rows in (t,q[q<len(y)]):
        for c in np.unique(state[rows]):
            ix=rows[state[rows]==c]
            shuffled[ix]=rng.permutation(y[ix])
    true,auc=density_ratio(a,b,state,t,q,y,seed)
    sham,nauc=density_ratio(a,b,state,t,q,shuffled,seed)
    return dict(independent=expit(independent),direct=direct,real=expit(independent+true),sham=expit(independent+sham),
                true_log_ratio=true,sham_log_ratio=sham),np.array([auc,nauc])


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    a,b=design(frame,groups)
    # State C is available to every member of a synthetic pair. The original full features already include it.
    sleep=np.select([frame.CryoSleep.eq(True),frame.CryoSleep.eq(False)],[2,1],default=0)
    state=2*sleep+frame[BILLS].gt(0).any(axis=1).to_numpy()
    n=len(y)
    OUT.mkdir(exist_ok=True)
    scores={}
    discrimination=[]
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(state))
        path=OUT/f'fold{k}.npz'
        if not path.exists():
            pred,auc=fit(a,b,state,y,groups,t,q,77321+k)
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,t=t,q=q,y=y[t],ids=frame.PassengerId.iloc[q].to_numpy(str),auc=auc,**pred)
            tmp.replace(path)
        with np.load(path) as z:
            for key,v in [('t',t),('q',q),('y',y[t]),('ids',frame.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[key],v)
            if k<8:discrimination.append(z['auc'])
            for key in ('independent','direct','real','sham','true_log_ratio','sham_log_ratio'):
                scores.setdefault(key,np.full(len(state),np.nan))[q]=z[key]
        print(k,flush=True)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        anchor=z['anchor']
        np.testing.assert_array_equal(anchor[:n],base)
        strength=abs(odds(z['scores'][:,-3:].mean(1)))
    reference=np.where(anchor,strength,-strength)
    scores['real_on_anchor']=expit(reference+scores['true_log_ratio'])
    scores['sham_on_anchor']=expit(reference+scores['sham_log_ratio'])
    tags=[k for k in scores if not k.endswith('log_ratio')]
    edits={k:(scores[k]>=.5)!=anchor for k in tags}
    edits['exclusive']=edits['real_on_anchor'] & ~edits['sham_on_anchor'] & edits['real'] & ~edits['independent']
    edits['confirmed']=edits['exclusive'] & edits['direct']
    report=dict(models={k:dict(accuracy=float(((scores[k][:n]>=.5)==y).mean()),log_loss=float(log_loss(y,scores[k][:n])))
                        for k in ('direct','independent','real','sham')},
                held_discrimination=np.mean(discrimination,axis=0).tolist(),rules={})
    for k,act in edits.items():
        delta=act[:n]*np.where(base!=y,1,-1)
        report['rules'][k]=dict(test_changes=int(act[n:].sum()),by_fold=[int(delta[known&(f==k)].sum()) for k in range(8)],
            scopes={name:count(act[:n],m,base!=y) for name,m in [('all',known),('fresh',fresh&known),('blind',blind)]})
    np.savez_compressed(OUT/'aligned.npz',ids=frame.PassengerId.to_numpy(str),y=y,fold=f,**scores)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['models'],report['held_discrimination'],report['rules']['exclusive'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
