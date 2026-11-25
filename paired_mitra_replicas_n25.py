"""Test independent half-roster replicas as challengers and as confirmations of a tree correction."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from audit_selected_tree_pair_transfer_n25 import stages
from export_selected_tree_pair_n25 import SEEDS
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote,refit
from full_roster_six_failure_archive_n24 import prediction_sets


def routes(incumbent,anchor,replicas,tree,old_base):
    sign=np.where(incumbent,-1.,1.)
    vote=(replicas-.5)*sign[:,None]
    headroom=(anchor-.5)*-sign
    offered=stages(tree,anchor,incumbent,old_base)[:,-1]
    both=vote.min(1)>0
    return {
        'first_replica':vote[:,0]>0,
        'second_replica':vote[:,1]>0,
        'both_opposed':both,
        'both_stronger':both & (vote.min(1)>headroom),
        'replica_mean_stronger':both & (vote.mean(1)>headroom),
        'tree_confirmed_one':offered & (vote.max(1)>0),
        'tree_confirmed_both':offered & both,
        'tree_confirmed_by_mean':offered & (vote.mean(1)>0),
        'tree_without_any_replica':offered & (vote.max(1)<=0),
    }


def run():
    raw,y,g,f,names,h,full,ref,_,cols,rot=panel_data()
    _,oy,og,rows,of,_,oldbase,_,_,_,_,old,_=aligned()
    j=names.index('mitra96')
    copies=np.column_stack([h[:,j],rot[:,cols.index(j)]])
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    ix=np.arange(len(y))
    orig_tree=original_prediction(ix,f,g,y,'full')
    champ=prepared_vote(orig_tree,ref,nine)
    past=prepared_vote(original_prediction(rows,of,g,y,'historical'),oldbase,old)
    fresh=np.mean([refit(raw,y,g,s) for s in SEEDS],axis=0)
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    report={}
    edits=[]
    for (tag,ids,inc,anchor,base,fold),pair in zip([
        ('full',ix,champ,full[:,j],ref,f),('historical_three_vote',rows,past,old[:,0],oldbase,of)],panels,strict=True):
        truth=y[ids]
        all_wrong=(inc!=truth)&((np.column_stack(pair)>=.5)==inc[:,None]).all(1)
        new=~np.isin(g[ids],og)
        actions=routes(inc,anchor,copies[ids],fresh[ids],base)
        summary={}
        for name,flip in actions.items():
            pred=inc^flip
            r=stats(pred,inc,truth,g[ids],fold)
            r['all_wrong_fixed']=int((flip&all_wrong).sum())
            if new.any():
                r['new_orders']=stats(pred[new],inc[new],truth[new],g[ids][new],fold[new])
            summary[name]=r
        report[tag]=dict(all_wrong=int(all_wrong.sum()),policies=summary)
        edits.append(actions)
        print(tag,[(n,v['corrected'],v['harmed'],v['all_wrong_fixed']) for n,v in summary.items()],flush=True)
    wrong=(champ[rows]!=oy)&(past!=oy)
    correct=(champ[rows]==oy)&(past==oy)
    report['matched']={k:dict(fixed_both=int((edits[0][k][rows]&edits[1][k]&wrong).sum()),
                              harmed_both=int((edits[0][k][rows]&edits[1][k]&correct).sum()))
                       for k in edits[0]}
    (ROOT/'reports/nov23_matched_learning_curve/paired_mitra_replicas.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
