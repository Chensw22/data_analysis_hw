"""Learn passenger outcomes directly from matched positive and negative reference neighborhoods."""
import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import FAMILIES
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_mistake_influence_n24 import design
from full_roster_selected_tree_replay_n24 import prepared_vote,original_prediction
from full_roster_six_failure_archive_n24 import prediction_sets
from historical_nine_release_n25 import OUT as HISTORY
from passenger_pair_outcomes_n25 import classifier

MODES=('raw_matched','reference_only','raw_and_reference','shuffled_reference')
TAG='paired_reference_choice'


def contrasts(z,ids,bank,labels):
    parts=[]
    for side in (False,True):
        donors=bank[labels==side]
        distance,index=cKDTree(z[donors]).query(z[ids],k=8)
        gap=z[ids,None,:]-z[donors[index]]
        parts.append(np.column_stack([distance[:,0],distance.mean(1),abs(gap).mean(1),gap.mean(1)]))
    return parts[0]-parts[1]


def fit(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    output=[]
    for i,(a,b) in enumerate(GroupKFold(2,shuffle=True,random_state=387).split(t,groups=g[t])):
        bank,train=t[a],t[b]
        assert not np.isin(g[bank],g[train]).any()
        z=((x-x[bank].mean(0))/x[bank].std(0).clip(.2)).clip(-6,6).astype('float32')
        ids=np.r_[train,q]
        real=contrasts(z,ids,bank,y[bank])
        null=contrasts(z,ids,bank,np.random.default_rng(713+i).permutation(y[bank]))
        vals=[z[ids],real,np.column_stack([z[ids],real]),np.column_stack([z[ids],null])]
        output.append(np.column_stack([classifier().fit(a[:len(train)],y[train]).predict_proba(a[len(train):])[:,1] for a in vals]))
    return np.mean(output,axis=0)


def run():
    raw,y,g,f=data()
    names,h,b=banks(raw,y,f)
    ref=latest_reference(y,g,f,names,h,b)
    _,oy,og,rows,of,_,_,_,_,_,_,old,_=aligned()
    ix=np.arange(len(y))
    nine=b[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    inc=prepared_vote(original_prediction(ix,f,g,y,'full'),ref,nine)
    with np.load(HISTORY/'historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],rows)
        past=z['nine']
    with np.load(ROOT/'artifacts/nov25_production_anchor_cpu/oof.npz') as z:
        cert=z['certified_v131_equals_v96']
        certified_nine=z['certified_nine'][cert].astype(bool)
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    result={}
    for (tag,ids,fold,base,p),panel in zip([('full',ix,f,inc,b[:,names.index('mitra96')]),
                                           ('historical',rows,of,past,old[:,0])],panels,strict=True):
        predictions=protocol(design(raw),y,g,ids,fold,tag,fit,MODES,TAG)
        unanimous=((np.column_stack(panel)>=.5)==base[:,None]).all(1)
        truth=y[ids]
        new=~np.isin(g[ids],og)
        entries={}
        for j,name in enumerate(MODES):
            for w in (1.,.25):
                score=(1-w)*p+w*predictions[:,j]
                out=(score>=.5) if w==1 else np.where((p>=.5)==base,score>=.5,base)
                v=stats(out,base,truth,g[ids],fold)
                v['unanimous_fixed']=int((unanimous&(base!=truth)&(out==truth)).sum())
                v['unanimous_harmed']=int((unanimous&(base==truth)&(out!=truth)).sum())
                if new.any():
                    v['new_orders']=stats(out[new],base[new],truth[new],g[ids][new],fold[new])
                    pred=(score[cert]>=.5) if w==1 else np.where((p[cert]>=.5)==certified_nine,score[cert]>=.5,certified_nine)
                    v['certified']=stats(pred,certified_nine,truth[cert],g[cert],fold[cert])
                entries[name+'_'+str(w)]=v
        result[tag]=entries
        print(tag,[(k,v['corrected'],v['harmed'],v['unanimous_fixed']) for k,v in entries.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/paired_reference_choice.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
