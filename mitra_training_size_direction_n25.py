"""Test whether changes in Mitra scores with larger training rosters transfer to label decisions."""
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote
from full_roster_six_failure_archive_n24 import prediction_sets


def movement(p,copies,inc,base,alpha):
    x=logit(p.clip(.002,.998))
    small=logit(copies.clip(.002,.998))
    delta=x[:,None]-small
    concordant=(delta[:,0]*delta[:,1])>0
    free=(inc==base)&((p>=.5)==inc)
    estimate=delta.mean(1)
    rules={}
    for name,weight in [('pool',-.5),('extra_training',alpha),('full_step',1.)]:
        choice=x+weight*estimate>=0
        rules[name]=np.where(free,choice,inc)
        rules[name+'_same_direction']=np.where(free&concordant,choice,inc)
    return rules


def run():
    raw,y,g,f,names,h,b,ref,_,cols,rot=panel_data()
    _,oy,og,rows,of,_,base,_,_,_,_,old,_=aligned()
    j=names.index('mitra96')
    replicas=np.column_stack([h[:,j],rot[:,cols.index(j)]])
    ix=np.arange(len(y))
    tree=original_prediction(ix,f,g,y,'full')
    ot=original_prediction(rows,of,g,y,'historical')
    nine=b[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    # Assume score bias is proportional to 1/sqrt(training size). Use the
    # nominal half and seven-eighths training fractions, then project up to 100%.
    alpha=(1/np.sqrt(7/8)-1)/(1/np.sqrt(.5)-1/np.sqrt(7/8))
    report={'nominal_extrapolation_coefficient':alpha}
    both=[]
    for (tag,ids,fold,inc,orig,p),panel in zip([
            ('full',ix,f,prepared_vote(tree,ref,nine),ref,b[:,j]),
            ('historical_three_vote',rows,of,prepared_vote(ot,base,old),base,old[:,0])],panels,strict=True):
        strict=(inc!=y[ids])&((np.column_stack(panel)>=.5)==inc[:,None]).all(1)
        new=~np.isin(g[ids],og)
        choices=movement(p,replicas[ids],inc,orig,alpha)
        results={}
        for name,pred in choices.items():
            v=stats(pred,inc,y[ids],g[ids],fold)
            v['common_fixed']=int((strict & (pred==y[ids])).sum())
            if new.any():
                v['new_orders']=stats(pred[new],inc[new],y[ids][new],g[ids][new],fold[new])
            results[name]=v
        report[tag]=results
        print(tag,[(k,v['corrected'],v['harmed'],v.get('new_orders',{}).get('net'),v['common_fixed']) for k,v in results.items()],flush=True)
        both.append((inc,choices))
    (inc,here),(past,hist)=both
    report['paired']={}
    for k,pred in here.items():
        changes=(pred[rows]!=inc[rows])&(past!=hist[k])
        report['paired'][k]=dict(fixed=int((changes&(inc[rows]!=oy)&(past!=oy)).sum()),
                                 harmed=int((changes&(inc[rows]==oy)&(past==oy)).sum()))
    (ROOT/'reports/nov23_matched_learning_curve/mitra_training_size_direction.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
