"""Test the conditional-vs-pooled bill evidence on the incumbent without replacing its other information."""
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from broad_bill_completion_states_n23 import REPORT
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write


def main():
    row,y,frame,groups,_=load()
    wrong,right,hard=cohorts(row,y)
    missing=frame[BILLS].isna().any(axis=1).to_numpy()
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_conditional_bill_completion/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            base,ref=z['base'],z['ref']
            reference=ref==y
            active=ref==(base>=.5)
            odds=lambda p:logit(np.clip(p,.002,.998))
            baseline=odds(base)
            hits={'reference':reference}
            for view,scope in [('without_bills',active),('completion',active & missing)]:
                orig,real,sham=[odds(z[f'{k}_{view}']) for k in ['pooled','stationary','shuffled']]
                for name,increment in [('raw',real-orig),('sham',sham-orig),('excess',real-sham)]:
                    vote=np.where(scope,baseline+increment>=0,ref)
                    hits[f'{view}_{name}']=vote==y
        plans[seed]={k:paired(v,reference) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:m['fixed']-m['harmed'] for k,m in plans[seed].items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,s in history.items():
        h=np.stack(s)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),groups))
    write(REPORT/'transferred_odds.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
