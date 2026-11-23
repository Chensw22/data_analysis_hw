"""Compare large-party additions to equally many errors reassigned across matched bookings."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from audit_pending_booking_influence_n23 import REPORT,votes_for
from audit_stable_rescue_placebo_n23 import booking_blocks,simulate
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from pending_submission_diversity_n23 import DONOR,LINEAR,PAIRS


def main():
    row,y,frame,g,_=load()
    combined=json.loads((REPORT/'combined.json').read_text())
    weight=json.loads((PAIRS/'test_budget.json').read_text())['test']['change_150']['weight']
    size=frame.BookingSize.to_numpy()
    plans,history={},[]
    for seed in [*map(str,SEEDS),'v131']:
        ref,all_votes=votes_for(frame,g,row,y,seed,weight)
        donor,linear=all_votes[DONOR],all_votes[LINEAR]
        eligible=(donor==ref)&(linear!=ref)
        chosen=eligible&(size>=3)
        error=donor!=y
        observed=int((chosen&error).sum())
        assert observed==combined['plans'][seed]['large_union']['vs_queued']['fixed']
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            base=z['base']
        risk=np.where(ref,1-base,base)
        bands=np.searchsorted(np.unique(np.quantile(risk[eligible],[.25,.5,.75])),risk)
        blocks=booking_blocks(g,eligible,np.c_[ref,bands].astype(int),risk)
        mean,draws,movable=simulate(blocks,error,chosen.astype(int)[:,None],3000,864)
        plans[seed]=dict(eligible=int(eligible.sum()),selected=int(chosen.sum()),fixed=observed,
                         expected_if_exchangeable=float(mean[0]),interval=np.quantile(draws[:,0],[.025,.975]).tolist(),
                         movable_selected=int((movable&chosen).sum()))
        if seed!='v131':
            history.append(chosen.astype(int)*np.where(error,1,-1))
    diffs=np.stack(history)
    write(REPORT/'large_union_control.json',dict(plans=plans,increment95_pp=interval(diffs.mean(0),g),
          all_three_additional_fixes=int((diffs==1).all(0).sum()),all_three_additional_harms=int((diffs==-1).all(0).sum())))
    print(plans,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        main()
