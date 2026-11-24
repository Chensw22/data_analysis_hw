"""Use all eligible single-protocol training records when fitting a shared correction."""
import json

import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_same_model_calibration_n23 import learn


def design(p, attributes):
    z=logit(p.clip(.005,.995))
    contrasts=z[:,1:]-z[:,:1]
    return np.column_stack([np.ones(len(z)),z[:,0],contrasts,attributes,
                            attributes*(z[:,0:1]/3),contrasts[:,0]*contrasts[:,1]]),z[:,0]


def train(left,right,old_ids,new_ids,mode,y,baseline_new):
    # Historical incumbent overrides depend on a witness fitted on another split.
    # Keep these decisions exclusively for query evaluation.
    lo=np.flatnonzero(old_ids)
    hi=np.flatnonzero(new_ids & ((right[1]>=0)==baseline_new))
    if mode=='intersection':
        common=np.intersect1d(lo,hi)
        lo,hi=common,common
    if mode=='eightfold_only':
        lo=np.empty(0,dtype=int)
    if mode=='historical_only':
        hi=np.empty(0,dtype=int)
    xx=np.vstack([left[0][lo],right[0][hi]])
    yy=np.r_[y[lo],y[hi]]
    offset=np.r_[left[1][lo],right[1][hi]]
    w=learn(xx,yy,offset,np.zeros(xx.shape[1]))
    return w,len(lo),len(hi)


def predict(baseline,representation,ix,w):
    x,margin=representation
    return np.where((margin[ix]>=0)==baseline[ix],margin[ix]+x[ix]@w>=0,baseline[ix])


def run():
    _,y,g,rows,oldfold,newfold,champion,gate,cold,_,_,old,new=aligned()
    raw,all_y,groups,fold=data()
    names,half,full=banks(raw,all_y,fold)
    ref=latest_reference(all_y,groups,fold,names,half,full)
    attributes=passenger_vectors(raw)
    left=design(old,attributes[rows])
    right=design(new,attributes[rows])
    all_right=design(full[:,[names.index(k) for k in ['mitra96','v44_without_name','tabicl']]],attributes)
    common_errors=((full[rows]>=.5)!=y[:,None]).all(1) & (champion!=y)
    assert common_errors.sum()==201
    additional=~np.isin(np.arange(len(all_y)),rows)
    report={}
    for mode in ['historical_only','eightfold_only','intersection','all_eligible']:
        pred_old,pred_new=champion.copy(),gate.copy()
        extra=ref.copy()
        training=[]
        for f in np.unique(newfold):
            for h in np.unique(oldfold):
                cell=np.flatnonzero((newfold==f)&(oldfold==h))
                for a,b in GroupKFold(4,shuffle=True,random_state=462).split(cell,groups=g[cell]):
                    q=cell[b]
                    allowed=~np.isin(g,g[q])
                    # Each donor is used only under a protocol whose base models
                    # excluded the query labels. The other protocol's donor score is ignored.
                    w,nl,nr=train(left,right,(oldfold==h)&allowed,(newfold==f)&allowed,
                                   mode,y,gate)
                    training.append([nl,nr])
                    pred_old[q]=predict(champion,left,q,w)
                    pred_new[q]=predict(gate,right,q,w)
            q=np.flatnonzero(additional&(fold==f))
            allowed=~np.isin(g,groups[q])
            # All historic score sources were fitted on the original development roster.
            w,_,_=train(left,right,allowed,(newfold==f)&allowed,mode,y,gate)
            extra[q]=predict(ref,all_right,q,w)
        def describe(p,base,truth,grp,splits):
            d=(p==truth).astype(int)-(base==truth)
            return dict(fixed=int((d==1).sum()),harmed=int((d==-1).sum()),
                        accuracy=float((p==truth).mean()),fold_net=[int(d[splits==f].sum()) for f in np.unique(splits)],
                        ci_pp=interval(d,grp))
        report[mode]=dict(historical=describe(pred_old,champion,y,g,oldfold),
                          eightfold_common=describe(pred_new,gate,y,g,newfold),
                          additional=describe(extra[additional],ref[additional],all_y[additional],
                                              groups[additional],fold[additional]),
                          cold=describe(pred_old[cold],champion[cold],y[cold],g[cold],oldfold[cold]),
                          average_training_records=np.mean(training,axis=0).tolist(),
                          paired_common_fixed_both=int((common_errors&(pred_old==y)&(pred_new==y)).sum()),
                          corrected_both=int(((champion!=y)&(gate!=y)&(pred_old==y)&(pred_new==y)).sum()))
        for label, mask in [('additional_new_orders',additional & ~np.isin(groups,g)),
                            ('additional_shared_orders',additional & np.isin(groups,g))]:
            report[mode][label]=dict(rows=int(mask.sum()),**describe(
                extra[mask],ref[mask],all_y[mask],groups[mask],fold[mask]))
        print(mode,[(k,report[mode][k]['fixed'],report[mode][k]['harmed'])
                    for k in ['historical','eightfold_common','additional']],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/dual_protocol_calibration.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
