"""Require independent families to challenge the current prediction on the same passenger."""
import numpy as np

from broad_abstaining_fields_n23 import REPORT
from broad_bill_neighbor_decision_n23 import opinions
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write


def main():
    row,y,frame,groups,_ = load()
    wrong,right,hard = cohorts(row,y)
    many = frame.BookingSize.to_numpy()>1
    plans,history,changed = {},{},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,base,ref,bill,neighbors = opinions(seed,row,y,many)
        with np.load(ART/f'nov23_abstaining_fields/{seed}.npz') as data:
            for key, expected in [('row',row),('y',y),('fold',fold),('ref',ref),('base',base)]:
                np.testing.assert_array_equal(data[key], expected)
            fields = np.column_stack([data[k] for k in ('personal_first','bill_first','averaged_order')])
        families = [((p>=.5) != ref[:,None]) for p in (bill,neighbors,fields)]
        active = ref==(base>=.5)
        decisions = {'reference':ref}
        for mode in ['any','majority']:
            signals = np.column_stack([v.any(1) if mode=='any' else v.mean(1)>.5 for v in families])
            for k,mask in [('old_two',signals[:,:2].all(1)),('two_of_three',signals.sum(1)>=2),
                           ('three_of_three',signals.all(1))]:
                decisions[mode+'_'+k] = ref ^ (active & mask)
        plans[seed] = {k:paired(v==y,ref==y) for k,v in decisions.items()}
        if seed!='v131':
            for k,v in decisions.items():
                history.setdefault(k,[]).append(v==y)
            for i,votes in enumerate(families):
                for j in range(votes.shape[1]):
                    changed.setdefault((i,j),[]).append(active & votes[:,j])
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    reference = np.stack(history['reference'])
    stable = {k:dict(hard_stable=int((hard & np.all(h,0)).sum()),hard_ever=int((hard & np.any(h,0)).sum()),
                     common_stable=int((wrong & np.all(h,0)).sum()),stable_harms=int((right & ~np.any(h,0)).sum()),
                     gain95_pp=interval((np.stack(h).astype(float)-reference).mean(0),groups))
              for k,h in history.items()}
    # On the curated hard cohort the reference was always wrong. OR across models only
    # after taking the intersection across all three splits for each individual model.
    unions = [np.any([np.all(v,0) for (i,_),v in changed.items() if i==family],0)
              for family in range(3)]
    prior = unions[0] | unions[1]
    novel = hard & unions[2] & ~prior
    diagnostic = dict(previous=int((hard&prior).sum()),fields=int((hard&unions[2]).sum()),
                      overlap=int((hard&prior&unions[2]).sum()),new=int(novel.sum()),
                      new_solo=int((novel & ~many).sum()),combined=int((hard & (prior|unions[2])).sum()))
    assert diagnostic['previous']==33
    write(REPORT/'corroboration.json',dict(plans=plans,stable=stable,hard_cohort_union=diagnostic))
    print(diagnostic)


if __name__=='__main__':
    main()
