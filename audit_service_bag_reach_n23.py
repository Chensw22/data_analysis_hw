"""Compute the joint scope and true development opportunity of three pending submissions."""
import json
import numpy as np
from audit_flip_capacity_n23 import required
from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_three_corrections_replay_n23 import scores
from export_service_bag_n23 import NAME


def main():
    row,y,frame,g,_=load()
    plans={}
    for seed in [*map(str,SEEDS),'v131']:
        older,ref=scores(frame,g,row,y,seed)
        with np.load(ART/f'nov23_service_bag/{seed}.npz') as z:
            for key,v in [('ref',ref),('y',y),('row',row)]:
                np.testing.assert_array_equal(z[key],v)
            service=np.where(ref==(z['base']>=.5),.75*z['base']+.25*z['fixed']>=.5,ref)
        old_union=(ref!=(older[:,0]>=.5))|(ref!=(older.mean(1)>=.5))
        new=(ref!=service)&~old_union
        union=old_union|(ref!=service)
        errors=y!=ref
        need=required(len(y))-int((~errors).sum())
        plans[seed]=dict(required=need,old_union=int(old_union.sum()),union=int(union.sum()),
                         reachable_true_errors=int((union&errors).sum()),uncovered_gap=need-int((union&errors).sum()),
                         new_changes=int(new.sum()),new_targets_corrected=int((new&errors).sum()))
    reach=json.loads((ROOT/'reports'/NAME/'manifest.json').read_text())
    capacity=json.loads((ROOT/'reports/nov23_flip_capacity/capacity.json').read_text())
    public=capacity['official']
    needed=public['needed_net']
    pooled=reach['three_candidate_union']
    result=dict(development=plans,public=dict(net_needed=needed,three_union=pooled,
                                             all_correct_max=(public['before']+pooled)/public['n'],
                                             fixed_needed_if_all_changed=(pooled+needed+1)//2,
                                             service_alone_max=reach['unchecked_ceiling_correct']/public['n']))
    write(REPORT/'three_candidate_scope.json',result)
    print(result)


if __name__=='__main__':
    main()
