"""Hold query passengers and tree scores fixed while exchanging the two archived baseline pipelines."""
import json
from itertools import product

import numpy as np
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote,refit
from audit_selected_tree_pair_transfer_n25 import stages
from full_roster_disagreement_calibration_n24 import stats
from export_selected_tree_pair_n25 import SEEDS


def run():
    raw,y,g,fold=data()
    _,oy,og,rows,of,_,oldbase,newbase,_,_,_,old,new=aligned()
    fresh=np.mean([refit(raw,y,g,s) for s in SEEDS],axis=0)[rows]
    oldtree=original_prediction(rows,of,g,y,'historical')
    newtree=original_prediction(np.arange(len(y)),fold,g,y,'full')[rows]
    data_by_fit={'eight':(newbase,newtree,new),'historical':(oldbase,oldtree,old)}
    result={}
    flips={}
    for inc_key,anchor_key in product(data_by_fit,repeat=2):
        base,tree,opinions=data_by_fit[inc_key]
        inc=prepared_vote(tree,base,opinions)
        anchor=data_by_fit[anchor_key][2][:,0]
        stages_seen=stages(fresh,anchor,inc,base)
        edit=stages_seen[:,-1]
        key=inc_key+'_'+anchor_key
        flips[key]=edit
        result[key]=dict(stages=stages_seen.sum(0).tolist(),effects=stats(inc^edit,inc,oy,og,of))
        print(key,result[key]['stages'],result[key]['effects']['corrected'],result[key]['effects']['harmed'],flush=True)
    left,right=flips['eight_eight'],flips['historical_historical']
    result['overlap']=dict(both=int((left&right).sum()),eight_only=int((left&~right).sum()),
                           historical_only=int((~left&right).sum()),
                           anchor_decision_different=int(((old[:,0]>=.5)!=(new[:,0]>=.5)).sum()))
    (ROOT/'reports/nov23_matched_learning_curve/selected_tree_anchor_swap.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
