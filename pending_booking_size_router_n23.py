"""Test whether the most distinct pending model can correct the queued model on different party sizes."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_pending_booking_influence_n23 import REPORT,votes_for
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from pending_submission_diversity_n23 import DONOR,LINEAR,PAIRS


def rules(donor,linear,ref,size):
    large=size>=3
    return {'queued':donor,'linear':linear,'union':np.where(linear!=ref,linear,donor),
            'large_linear':np.where(large,linear,donor),
            'large_union':np.where(large&(linear!=ref),linear,donor)}


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    size=frame.BookingSize.to_numpy()
    weight=json.loads((PAIRS/'test_budget.json').read_text())['test']['change_150']['weight']
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        ref,all_votes=votes_for(frame,g,row,y,seed,weight)
        donor,linear=all_votes[DONOR],all_votes[LINEAR]
        dispute=donor!=linear
        votes=rules(donor,linear,ref,size)
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            fold=z['fold']
        winner=linear==y
        side=(linear!=ref).astype(int)
        region=np.minimum(size,3).astype(int)-1
        for k in ('pooled_gate','size_gate'):
            votes[k]=donor.copy()
        # Learn only on other bookings scored by the same outer-fold models.
        for f in np.unique(fold):
            ids=np.flatnonzero((fold==f)&dispute)
            for a,b in GroupKFold(3,shuffle=True,random_state=223).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                prob=(np.bincount(side[t],weights=winner[t],minlength=2)+10)/(np.bincount(side[t],minlength=2)+20)
                code=side*3+region
                local=(np.bincount(code[t],weights=winner[t],minlength=6)+15*np.repeat(prob,3))/(np.bincount(code[t],minlength=6)+15)
                for k,select in [('pooled_gate',prob[side[q]]>.5),('size_gate',local[code[q]]>.5)]:
                    votes[k][q[select]]=linear[q[select]]
        hits={k:v==y for k,v in votes.items()}
        hits['reference']=ref==y
        metrics={k:dict(vs_reference=paired(h,ref==y),vs_queued=paired(h,donor==y)) for k,h in hits.items()}
        plans[seed]=metrics
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        print(seed,{k:v['vs_reference']['fixed']-v['vs_reference']['harmed'] for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),g))
    write(REPORT/'combined.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
