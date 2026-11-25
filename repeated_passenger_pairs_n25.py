"""Repeat a pairwise outcome model on fresh reference bookings and adjudicate its unique errors."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from passenger_pair_outcomes_n25 import fit,MODES,TAG,protocol,design,HISTORY
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_selected_tree_replay_n24 import prepared_vote,original_prediction
from full_roster_six_failure_archive_n24 import prediction_sets


def run():
    raw,y,g,f=data()
    names,h,b=banks(raw,y,f)
    ref=latest_reference(y,g,f,names,h,b)
    _,oy,og,rows,of,_,_,_,_,_,_,old,_=aligned()
    ix=np.arange(len(y))
    nine=b[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    inc=prepared_vote(original_prediction(ix,f,g,y,'full'),ref,nine)
    with np.load(HISTORY/'historical.npz') as z:
        np.testing.assert_array_equal(z['y'],oy)
        past=z['nine']
    x=design(raw)
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    result={}
    for (tag,ids,fold,base,p),panel in zip([('full',ix,f,inc,b[:,names.index('mitra96')]),
                                            ('historical',rows,of,past,old[:,0])],panels,strict=True):
        original=protocol(x,y,g,ids,fold,tag,fit,MODES,TAG)
        fresh=protocol(x,y,g,ids,fold,tag,lambda *args:fit(*args,seed=1273),MODES,TAG+'_repeat')
        yq=y[ids]
        first=(original[:,1]>=.5)!=base
        second=(fresh[:,1]>=.5)!=base
        archive=np.column_stack(panel)
        unanimity=((archive>=.5)==base[:,None]).all(1)
        offered=first|second
        both=first&second
        models={'first':first,'second':second,'both':both,'both_and_all_others_agree':both&unanimity,
                'quarter_mix':((.75*p+.125*(original[:,1]+fresh[:,1])>=.5)!=base)&((p>=.5)==base)}
        sign=np.where(base,-1.,1.)
        anchor=sign*logit(p.clip(.002,.998))
        # All underlying scores share the given outer holdout. The correctness selector
        # trains only on separate booking groups within it.
        shared=np.column_stack([base,anchor])
        detail=np.column_stack([sign[:,None]*logit(a.clip(.002,.998)) for a in (original,fresh)])
        opinions=sign[:,None]*logit(np.quantile(archive,[.1,.5,.9],axis=1).T.clip(.002,.998))
        for k,features in [('anchor',shared),('pair_evidence',np.column_stack([shared,detail,unanimity,opinions]))]:
            flip=np.zeros(len(yq),bool)
            for level in np.unique(fold):
                outer=np.flatnonzero(fold==level)
                for t0,q0 in GroupKFold(4,shuffle=True,random_state=213).split(outer,groups=g[ids][outer]):
                    t,q=outer[t0],outer[q0]
                    t,q=t[offered[t]],q[offered[q]]
                    assert not np.isin(g[ids][t],g[ids][q]).any()
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=.08,max_iter=500))
                    model.fit(features[t],yq[t]!=base[t])
                    flip[q]=model.predict(features[q])
            models[k]=flip
        out={}
        for k,flip in models.items():
            pred=base^flip
            val=stats(pred,base,yq,g[ids],fold)
            val['unanimous_fixed']=int((flip&unanimity&(base!=yq)).sum())
            val['unanimous_harmed']=int((flip&unanimity&(base==yq)).sum())
            new=~np.isin(g[ids],og)
            if new.any():
                val['new_orders']=stats(pred[new],base[new],yq[new],g[ids][new],fold[new])
            out[k]=val
        result[tag]=dict(unanimous_errors=int((unanimity&(base!=yq)).sum()),policies=out)
        print(tag,[(k,v['corrected'],v['harmed'],v['unanimous_fixed'],v['unanimous_harmed']) for k,v in out.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/repeated_passenger_pairs.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
