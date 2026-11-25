"""Resolve actual class-probability ties with independent experts and compare to adjacent score levels."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote
from full_roster_six_failure_archive_n24 import prediction_sets


def proposals(p,orig,inc,opinions,tree):
    choices={
        'negative_on_tie':np.zeros(len(p),bool),
        'v44':opinions[:,1]>=.5,
        'tabicl':opinions[:,2]>=.5,
        'two_model_mean':opinions[:,1:].mean(1)>=.5,
        'three_model_median':np.median(np.column_stack([tree,opinions[:,1:]]),axis=1)>=.5,
        'selected_tree':tree>=.5,
        'all_three_negative':np.column_stack([tree,opinions[:,1:]]).max(1)>=.5,
    }
    # Preserve all overrides of the anchor label by either release stage.
    free=(inc==orig)&((p>=.5)==orig)
    masks={'tie':p==.5,'adjacent_only':(abs(p-.5)<=.033)&(p!=.5)}
    return {level:{k:np.where(free&mask,v,inc) for k,v in choices.items()} for level,mask in masks.items()}


def run():
    raw,y,g,f=data()
    names,h,full=banks(raw,y,f)
    ref=latest_reference(y,g,f,names,h,full)
    _,oy,og,rows,of,_,base,_,_,_,_,old,_=aligned()
    ix=np.arange(len(y))
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    three=full[:,[names.index(k) for k in ('mitra96','v44_without_name','tabicl')]]
    tree=original_prediction(ix,f,g,y,'full')
    ot=original_prediction(rows,of,g,y,'historical')
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    report={}
    decisions=[]
    for (tag,ids,fold,p,orig,inc,rivals,t),panel in zip([
        ('full',ix,f,three[:,0],ref,prepared_vote(tree,ref,nine),three,tree),
        ('historical_three_vote',rows,of,old[:,0],base,prepared_vote(ot,base,old),old,ot)],panels,strict=True):
        a=proposals(p,orig,inc,rivals,t)
        decisions.append((inc,a))
        strict=(inc!=y[ids])&((np.column_stack(panel)>=.5)==inc[:,None]).all(1)
        tied=p==.5
        new=~np.isin(g[ids],og)
        result={'ties':dict(rows=int(tied.sum()),actually_false=int((tied&~y[ids]).sum()),
                            already_overridden=int((tied & (inc!=(p>=.5))).sum()),
                            min_nonzero_margin=float(abs(p[~tied]-.5).min()))}
        for level,options in a.items():
            result[level]={}
            for name,pred in options.items():
                v=stats(pred,inc,y[ids],g[ids],fold)
                v['all_wrong_fixed']=int((strict & (pred==y[ids])).sum())
                if new.any():
                    v['new_orders']=stats(pred[new],inc[new],y[ids][new],g[ids][new],fold[new])
                result[level][name]=v
        report[tag]=result
        print(tag,result['ties'],[(k,v['corrected'],v['harmed']) for k,v in result['tie'].items()],flush=True)
    (inc,a),(past,b)=decisions
    report['matched']={}
    for k,pred in a['tie'].items():
        both=(pred[rows]!=inc[rows])&(b['tie'][k]!=past)
        report['matched'][k]=dict(fixed=int((both&(inc[rows]!=oy)&(past!=oy)).sum()),
                                 harmed=int((both&(inc[rows]==oy)&(past==oy)).sum()))
    (ROOT/'reports/nov23_matched_learning_curve/mitra_tied_vote.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
