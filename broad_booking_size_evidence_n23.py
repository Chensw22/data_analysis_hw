"""Estimate the class information in observed party size beyond personal fields and class-specific size rates."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from audit_pending_booking_influence_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from exaone_validation_data import CATEGORICAL


def forecast(frame,groups,y,t,q):
    assert not np.intersect1d(groups[t],groups[q]).size
    x=frame.drop(columns='BookingSize')
    size=np.minimum(frame.BookingSize.to_numpy(int),4)-1
    bins=4
    qt=size[q]
    shuffled=size[t].copy()
    rng=np.random.default_rng(518)
    # Within each outcome shuffle the auxiliary size labels. Marginal category counts stay exact.
    for cond in (False,True):
        at=np.flatnonzero(y[t]==cond)
        shuffled[at]=rng.permutation(shuffled[at])
    results={}
    for name,bucket in [('actual',size[t]),('sham',shuffled)]:
        clf=CatBoostClassifier(iterations=210,depth=5,learning_rate=.1,one_hot_max_size=32,l2_leaf_reg=6,
                               verbose=False,thread_count=2,random_seed=114,allow_writing_files=False)
        clf.fit(x.iloc[t],bins*y[t].astype(int)+bucket,cat_features=CATEGORICAL)
        np.testing.assert_array_equal(clf.classes_,np.arange(2*bins))
        joint=clf.predict_proba(x.iloc[q]).reshape(len(q),2,bins)
        marginal=joint[:,1].sum(1)
        selected=joint[np.arange(len(q)),:,qt]
        conditional=selected[:,1]/selected.sum(1)
        results[name+'_marginal']=marginal
        results[name+'_conditional']=conditional
    counts=np.stack([np.bincount(size[t[y[t]==s]],minlength=bins)+1 for s in (False,True)])
    rate=counts/counts.sum(1,keepdims=True)
    results['size_odds']=np.log(rate[1,qt]/rate[0,qt])
    return results


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,ref,base=z['fold'],z['ref'],z['base']
        data={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for k,v in forecast(frame,g,y,t,q).items():
                data.setdefault(k,np.empty(len(y)))[q]=v
        odds=lambda p:logit(p.clip(.002,.998))
        actual=odds(data['actual_conditional'])-odds(data['actual_marginal'])
        sham=odds(data['sham_conditional'])-odds(data['sham_marginal'])
        scores={k:v for k,v in data.items() if k!='size_odds'}
        scores.update(quarter_conditional=.75*base+.25*data['actual_conditional'],
                      quarter_sham=.75*base+.25*data['sham_conditional'],
                      quarter_marginal=.75*base+.25*data['actual_marginal'])
        for k,delta in [('actual',actual),('sham',sham),('difference',actual-sham),('within_size',actual-data['size_odds'])]:
            scores['transfer_'+k]=expit(odds(base)+delta)
        active=ref==(base>=.5)
        hits={k:(np.where(active,p>=.5,ref)==y) for k,p in scores.items()}
        hits['reference']=ref==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        size=frame.BookingSize.to_numpy()
        by_size={}
        for k,h in hits.items():
            delta=h.astype(int)-(ref==y).astype(int)
            by_size[k]=[int(delta[mask].sum()) for mask in (size==1,size==2,size>=3)]
        plans[seed]=dict(metrics=metrics,net_by_booking_size=by_size)
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    stable={k:dict(hard_stable=int((hard&np.all(h,axis=0)).sum()),hard_ever=int((hard&np.any(h,axis=0)).sum()),
                   common_stable=int((wrong&np.all(h,axis=0)).sum()),stable_harms=int((right&~np.any(h,axis=0)).sum()))
            for k,h in history.items()}
    write(REPORT/'size_evidence.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
