"""Choose among heterogeneous corrections using booking-separated calibration inside each held-out fold."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from certified_tree_release_n25 import AUDIT,HISTORY,structural_inputs
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_six_failure_archive_n24 import prediction_sets
from production_anchor_cpu_replay_n25 import certified
from full_roster_disagreement_calibration_n24 import stats


def select(rivals,p,inc,y,groups,fold):
    free=(p>=.5)==inc
    mixes=.75*p[:,None]+.25*rivals
    flips=free[:,None]&((mixes>=.5)!=inc[:,None])
    sign=np.where(inc==y,-1,1)
    out={'all_mean':free&(((.75*p+.25*rivals.mean(1))>=.5)!=inc),
         'all_median':free&(((.75*p+.25*np.median(rivals,axis=1))>=.5)!=inc),
         'majority_edits':flips.sum(1)>flips.shape[1]/2}
    for key in ('winner','two_panel_winner','reliability_average','abstaining_vote'):
        out[key]=np.zeros(len(y),bool)
    trace=[]
    for level in np.unique(fold):
        ids=np.flatnonzero(fold==level)
        parts=[ids[v] for _,v in GroupKFold(3,shuffle=True,random_state=215).split(ids,groups=groups[ids])]
        for i,q in enumerate(parts):
            a,b=parts[(i+1)%3],parts[(i+2)%3]
            t=np.r_[a,b]
            assert not np.isin(groups[t],groups[q]).any() and not np.isin(groups[a],groups[b]).any()
            net_a=sign[a]@flips[a].astype(int)
            net_b=sign[b]@flips[b].astype(int)
            total=net_a+net_b
            winner=int(np.argmax(total))
            if total[winner]>0:
                out['winner'][q]=flips[q,winner]
            confirmed=np.where((net_a>0)&(net_b>0),total,0)
            if confirmed.max()>0:
                out['two_panel_winner'][q]=flips[q,int(confirmed.argmax())]
            # Shrink precision toward chance using a neutral prior of 12 proposed edits.
            weights=(total/(12+flips[t].sum(0))).clip(0)
            if weights.sum()>0:
                vote=np.average(rivals[q],axis=1,weights=weights)
                out['reliability_average'][q]=free[q]&(((.75*p[q]+.25*vote)>=.5)!=inc[q])
                out['abstaining_vote'][q]=(flips[q]@weights)>weights.sum()*.5
            trace.append(dict(held_fold=int(level),passengers=len(q),chosen=winner if total[winner]>0 else None,
                              confirmed=int(confirmed.argmax()) if confirmed.max()>0 else None))
    return out,trace


def run():
    raw,y,g,fold=data()
    _,oy,og,rows,of,_,_,_,_,_,_,old,_=aligned()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    fields=structural_inputs(pd.concat([raw,test],ignore_index=True))
    hc=certified(old[:,0],*[v[rows] for v in fields])
    with np.load(AUDIT/'oof.npz') as z:
        np.testing.assert_array_equal(z['y'],y)
        np.testing.assert_array_equal(z['fold'],fold)
        cert=z['certified_v131_equals_v96']
        p=z['v96'][cert]
        inc=z['certified_nine'][cert].astype(bool)
    with np.load(HISTORY/'historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],rows)
        prior=z['nine'][hc]
    names,panels=prediction_sets(raw,y,g,fold,rows,of)
    fam=np.array([k.split('/')[0] for k in names])
    unique=np.unique(fam)
    result={'family_order':['response','projections',*unique.tolist()]}
    for (tag,mask,ids,f,anchor,base),panel in zip([
         ('full',cert,np.flatnonzero(cert),fold[cert],p,inc),
         ('historical',hc,rows[hc],of[hc],old[hc,0],prior)],panels,strict=True):
        six,bank=[x[mask] for x in panel]
        rivals=np.column_stack([np.median(six[:,:3],1),np.median(six[:,3:],1),
                                *[np.median(bank[:,fam==k],1) for k in unique]])
        votes,trace=select(rivals,anchor,base,y[ids],g[ids],f)
        entry={}
        new=~np.isin(g[ids],og)
        for k,v in votes.items():
            pred=base^v
            score=stats(pred,base,y[ids],g[ids],f)
            if new.any():
                score['new_orders']=stats(pred[new],base[new],y[ids][new],g[ids][new],f[new])
            entry[k]=score
        result[tag]=dict(policies=entry,selection=trace)
        print(tag,[(k,v['corrected'],v['harmed'],v.get('new_orders',{}).get('net')) for k,v in entry.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/crossfit_certified_committee.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
