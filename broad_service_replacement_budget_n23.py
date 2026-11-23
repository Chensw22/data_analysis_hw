"""Test wide correction policies at the same unlabeled change budget for each model."""
import numpy as np
from broad_booking_policy_replay_n23 import interval
from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def allocate(fold,n):
    groups,count=np.unique(fold,return_counts=True)
    shares=n*count/count.sum()
    per=np.floor(shares).astype(int)
    per[np.argsort(per-shares,kind='stable')[:n-per.sum()]]+=1
    assert per.sum()==n
    return zip(groups,per)


def switch(base,ref,p,fold,n):
    # Each probability moves along the straight line from incumbent to challenger.
    # Rank only valid reversals by the mixing weight needed to cross .5.
    eligible=(ref==(base>=.5)) & (ref!=(p>=.5))
    cost=np.full(len(p),np.inf)
    cost[eligible]=np.abs(base[eligible]-.5)/np.abs(p[eligible]-base[eligible])
    assert np.all(cost[eligible]<=1+1e-12)
    change=np.zeros(len(p),bool)
    for group,budget in allocate(fold,n):
        ids=np.flatnonzero((fold==group)&eligible)
        ranked=ids[np.argsort(cost[ids],kind='stable')]
        change[ranked[:budget]]=True
    return ref^change,int(eligible.sum())


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_service_bag/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            base,ref,fold,service=z['base'],z['ref'],z['fold'],z['fixed']
        with np.load(ART/f'nov23_two_source_manifest/{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('ref',ref),('base',base),('fold',fold)]:
                np.testing.assert_array_equal(z[k],v)
            personal=z['direct']
        sources={'service':service,'personal':personal,'mean':(service+personal)/2}
        hits={'reference':ref==y}
        metrics={}
        for key,p in sources.items():
            for target in (250,350):
                name=f'{key}_{target}'
                budget=round(len(y)*target/4277)
                decision,eligible=switch(base,ref,p,fold,budget)
                hits[name]=decision==y
                values=paired(hits[name],ref==y)
                changes=values['fixed']+values['harmed']
                metrics[name]=dict(**values,proposed=budget,eligible=eligible,changes=changes,
                                   correction_precision=values['fixed']/changes)
        plans[seed]=metrics
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:(v['fixed'],v['harmed']) for k,v in metrics.items()},flush=True)
    ref=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-ref).mean(0),g))
    write(REPORT/'change_budget.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    main()
