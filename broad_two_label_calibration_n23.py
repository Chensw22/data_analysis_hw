"""Test whether expert disagreements can survive both hypothetical labels in held-out booking calibration."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from audit_disputed_passenger_separation_n23 import REPORT


def hypothetical(train_score,errors,query_score):
    cuts=np.unique(np.quantile(train_score,np.arange(1,21)/21))
    bins=np.searchsorted(cuts,train_score)
    qb=np.searchsorted(cuts,query_score)
    counts=np.bincount(bins,minlength=len(cuts)+1)
    hits=np.bincount(bins,weights=errors,minlength=len(counts))
    axis=np.arange(len(counts))
    low,high=np.zeros(len(qb)),np.zeros(len(qb))
    for i in np.unique(qb):
        cnt=counts.copy()
        cnt[i]+=1
        known=cnt>0
        for label,out in [(0,low),(1,high)]:
            num=hits.copy()
            num[i]+=label
            model=IsotonicRegression(out_of_bounds='clip').fit(axis[known],num[known]/cnt[known],sample_weight=cnt[known])
            out[qb==i]=model.predict([i])[0]
    assert np.all(low<=high+1e-10)
    return low,high


def main():
    row,y,frame,groups,_=load()
    wrong,right,hard=cohorts(row,y)
    history,plans={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,base,ref,families=family_scores(seed,row,y,frame)
        active=ref==(base>=.5)
        soft=np.column_stack([p.mean(1) for p in families.values()]).mean(1)
        with np.load(ART/f'nov23_service_pairs/{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('ref',ref),('base',base),('fold',fold)]:
                np.testing.assert_array_equal(z[k],v)
            pair=z['same_passenger_both']
        available={'anchor':base,'families':.75*base+.25*soft,'service_pairs':.75*base+.25*pair}
        error=ref!=y
        rules={}
        stats={}
        for name,opinion in available.items():
            score=np.where(ref,1-opinion,opinion)
            low=np.zeros(len(row))
            high=np.zeros(len(row))
            for f in np.unique(fold):
                for side in (False,True):
                    ids=np.flatnonzero((fold==f)&(ref==side)&active)
                    for a,b in GroupKFold(3,shuffle=True,random_state=921).split(ids,groups=groups[ids]):
                        t,q=ids[a],ids[b]
                        assert not np.intersect1d(groups[t],groups[q]).size
                        low[q],high[q]=hypothetical(score[t],error[t],score[q])
            rules[name+'_uncalibrated']=active&(score>.5)
            rules[name+'_both']=active&(low>.5)
            rules[name+'_regularized']=active&(high/(1-low+high)>.5)
            stats[name]=dict(straddle=int((active&(low<=.5)&(high>.5)).sum()),
                             mean_width=float((high-low)[active].mean()))
        hits={k:(~error^flip) for k,flip in rules.items()}
        hits['reference']=~error
        metrics={k:paired(h,~error) for k,h in hits.items()}
        plans[seed]=dict(metrics=metrics,sensitivity=stats)
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),common_stable=int((wrong&h.all(0)).sum()),
                       stable_harms=int((right&~h.any(0)).sum()),gain95_pp=interval((h.astype(float)-prev).mean(0),groups))
    write(REPORT/'two_label.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
