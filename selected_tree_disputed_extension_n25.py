"""Test a selected-tree override beyond unanimous cases against the newly scored release."""
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote,refit


def policies(champion,original,tree,anchor,opinions):
    # Preserve every correction from the release that actually scored 0.81739.
    available = (champion==original) & ((tree>=.5)!=champion) & ((anchor>=.5)==champion)
    opposition = np.where(champion[:,None],1-opinions,opinions)
    endorsement = np.sum(opposition >= .5,axis=1)
    margin = abs(logit(tree.clip(.003,.997)))
    hurdle = abs(logit(anchor.clip(.003,.997)))
    rivals = logit(opposition.clip(.003,.997))
    # One score per rival family prevents repeated variants from multiplying its evidence.
    return {name:champion ^ (available & flip) for name,flip in {
        'all_requests':np.ones(len(tree),bool),
        'stronger_tree':margin>hurdle,
        'one_rival':(endorsement>=1) & (margin>hurdle),
        'two_rivals':(endorsement==2) & (margin>hurdle),
        'mean_log_odds':margin+rivals.mean(1)>hurdle,
        'unanimous_margin':margin+rivals.min(1)>hurdle,
    }.items()}


def run():
    raw,y,g,fold=data()
    names,h,full=banks(raw,y,fold)
    ref=latest_reference(y,g,fold,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_=aligned()
    indices=[names.index(k) for k in ('mitra96','v44_without_name','tabicl')]
    bank=full[:,indices]
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    tree=original_prediction(np.arange(len(y)),fold,g,y,'full')
    native=original_prediction(rows,of,g,y,'historical')
    base=prepared_vote(tree,ref,nine)
    # Historical archive has three shared release probabilities. Compare under that same
    # restriction in both protocols as a separate check of transfer between training sizes.
    same_base=prepared_vote(tree,ref,bank)
    history_base=prepared_vote(native,champ,old)
    history=policies(history_base,champ,native,old[:,0],old[:,1:])
    unseen=~np.isin(g,og)
    out={'incumbent':stats(base,ref,y,g,fold),'historical_three_vote':{},'protocols':{}}
    for name,pred in history.items():
        out['historical_three_vote'][name]=stats(pred,history_base,oy,og,of)
    for key,inc,new_tree in [('original',base,tree),('same_three',same_base,tree),
                            ('order_refit_1',base,refit(raw,y,g,1021)),('order_refit_2',base,refit(raw,y,g,1022))]:
        result={}
        for name,pred in policies(inc,ref,new_tree,bank[:,0],bank[:,1:]).items():
            assert not (pred[inc!=ref]!=inc[inc!=ref]).any()
            vals=stats(pred,inc,y,g,fold)
            vals['unseen_net']=int((pred[unseen]==y[unseen]).sum()-(inc[unseen]==y[unseen]).sum())
            if key=='same_three':
                repeat_wrong=(history_base!=oy)&(inc[rows]!=oy)
                repeat_right=(history_base==oy)&(inc[rows]==oy)
                vals['repeat_corrected']=int((repeat_wrong&(pred[rows]==oy)&(history[name]==oy)).sum())
                vals['repeat_harmed']=int((repeat_right&(pred[rows]!=oy)&(history[name]!=oy)).sum())
            result[name]=vals
        out['protocols'][key]=result
        print(key,[(n,v['corrected'],v['harmed'],v['unseen_net']) for n,v in result.items()],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/selected_tree_disputed_extension.json'
    path.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
