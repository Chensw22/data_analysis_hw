"""Evaluate prospective group refits of the selected tree and a fixed two-fit release rule."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote,refit
from selected_tree_disputed_extension_n25 import policies


def run():
    raw,y,g,fold=data()
    names,h,full=banks(raw,y,fold)
    ref=latest_reference(y,g,fold,names,h,full)
    _,_,og,*_=aligned()
    ix=np.arange(len(y))
    orig=original_prediction(ix,fold,g,y,'full')
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    strong=full[:,[names.index(k) for k in ('mitra96','v44_without_name','tabicl')]]
    champion=prepared_vote(orig,ref,nine)
    new=~np.isin(g,og)
    predictions={}
    for seed in (1021,1022,11251,11252):
        predictions[str(seed)]=refit(raw,y,g,seed)
    report={}
    for pair in [('1021','1022'),('11251','11252')]:
        estimates=[predictions[k] for k in pair]
        combined=np.mean(estimates,axis=0)
        decisions={}
        for key,tree in [*zip(pair,estimates,strict=True),('mean',combined)]:
            decisions[key]=policies(champion,ref,tree,strong[:,0],strong[:,1:])['stronger_tree']
        decisions['both_accept']=np.where(decisions[pair[0]]==decisions[pair[1]],decisions[pair[0]],champion)
        for k,pred in decisions.items():
            outcome=stats(pred,champion,y,g,fold)
            outcome['new_orders']=stats(pred[new],champion[new],y[new],g[new],fold[new])
            report['+'.join(pair)+'/'+k]=outcome
            print(pair,k,outcome['corrected'],outcome['harmed'],outcome['new_orders']['net'],flush=True)
    target=ROOT/'reports/nov23_matched_learning_curve/selected_tree_order_transfer.json'
    target.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(3):
        run()
