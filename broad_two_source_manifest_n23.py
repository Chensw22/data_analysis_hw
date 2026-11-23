"""Recover a passenger outcome by conditioning a two-source label prediction on agreement."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_link_authenticity_n23 import REPORT
from broad_local_profile_fit_n23 import cohorts
from broad_matched_companions_n23 import donor_ids
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_within_booking_bill_matching_n23 import encode
from class_analogue_experts_n23 import design


def fit_model(x, target, weights=None):
    fit=CatBoostClassifier(iterations=310,depth=5,learning_rate=.055,l2_leaf_reg=4,thread_count=2,
                           random_seed=391,verbose=False,allow_writing_files=False)
    fit.fit(x,target,sample_weight=weights)
    np.testing.assert_array_equal(fit.classes_,np.arange(len(np.unique(target))))
    return fit


def model(x, target, query, weights=None):
    return fit_model(x, target, weights).predict_proba(query)


def run_fold(frame,g,size,person,bill,y,t,q,f):
    assert not np.intersect1d(g[t],g[q]).size
    own,_=design(frame,t)
    rng=np.random.default_rng(193+int(f))
    replacement=donor_ids(own,g,size,t,t,rng)[:,0]
    related=replacement.copy()
    # Prefer another labeled training member when available; singletons use the same
    # external partner as in the control. Pair construction does not read outcomes.
    for ids in pd.Series(np.arange(len(t))).groupby(g[t]).agg(list):
        if len(ids)>1:
            ordered=rng.permutation(ids)
            related[ordered]=t[np.roll(ordered,1)]
    def combine(i,j):
        return np.c_[person[i],bill[j],np.log(size[i])]
    actual=combine(t,t)
    query=combine(q,q)
    yt=y[t].astype(int)
    result={'direct':model(actual,yt,query)[:,1]}
    for name,donor in [('external',replacement),('booking',related)]:
        assert np.all(t!=donor) and np.isin(donor,t).all()
        artificial=combine(t,donor)
        pair_labels=2*yt+y[donor].astype(int)
        # Every real example has two identical donor outcomes, available only in training.
        codes=np.r_[3*yt,pair_labels]
        joint=model(np.vstack([actual,artificial]),codes,query).reshape(len(q),2,2)
        result[name+'_agree']=joint[:,1,1]/(joint[:,0,0]+joint[:,1,1])
        result[name+'_focal']=joint[:,1,:].sum(1)
        result[name+'_bill']=joint[:,:,1].sum(1)
        result[name+'_mean']=(result[name+'_focal']+result[name+'_bill'])/2
        # Two half-weight copies give the binary learner the average of the two labels.
        duplicated=np.vstack([actual,artificial,artificial])
        labels=np.r_[yt,yt,y[donor]]
        result[name+'_soft']=model(duplicated,labels,query,np.repeat([1.,.5,.5],len(t)))[:,1]
    return result,int(np.sum(g[t]==g[related]))


def main():
    row,y,frame,g,_=load()
    size=frame.BookingSize.to_numpy()
    single=size==1
    wrong,right,hard=cohorts(row,y)
    person,bill=encode(frame)
    out=ART/'nov23_two_source_manifest'
    out.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        base=op[:,0]
        bank,paired_people={},[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            scores,n=run_fold(frame,g,size,person,bill,y,t,q,f)
            paired_people.append(n)
            for k,p in scores.items():bank.setdefault(k,np.full(len(y),np.nan))[q]=p
        hits={'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all()
            hits[k]=(p>=.5)==y
            hits['quarter_'+k]=np.where(ref==(base>=.5),.75*base+.25*p>=.5,ref)==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,singles={k:paired(v[single],(ref==y)[single]) for k,v in hits.items()},
                         paired_training_passengers=paired_people)
        if seed!='v131':
            for k,v in hits.items():hist.setdefault(k,[]).append(v)
        print(seed,{k:m['fixed']-m['harmed'] for k,m in metrics.items() if k.startswith('quarter_')},flush=True)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**bank)
    r=np.stack(hist['reference'])
    stats={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_solo_stable=int((np.all(v,axis=0)&hard&single).sum()),hard_ever=int((np.any(v,axis=0)&hard).sum()),
                  stable_harms=int((~np.any(v,axis=0)&right).sum()),gain95_pp=interval((np.stack(v).astype(float)-r).mean(0),g))
           for k,v in hist.items()}
    write(REPORT/'two_sources.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(2):main()
