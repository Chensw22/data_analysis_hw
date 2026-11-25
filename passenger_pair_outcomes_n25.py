"""Predict whether two passengers share an outcome, then classify against labelled reference passengers."""
import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
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
from certified_tree_release_n25 import AUDIT

TAG='passenger_pair_outcomes'
MODES=('raw_tree','near_pairs','random_pairs','near_and_random')


def classifier():
    return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,min_samples_leaf=24,
                                          l2_regularization=3,early_stopping=False,random_state=636)


def references(z,truth,bank,ids,seed):
    rng=np.random.default_rng(seed)
    pairs=[]
    for side in (False,True):
        donor=bank[truth[bank]==side]
        _,near=cKDTree(z[donor]).query(z[ids],k=4)
        random=rng.choice(donor,size=(len(ids),4))
        pairs.append(np.column_stack([donor[near],random]))
    return np.stack(pairs,axis=1)


def features(z,ids,donors):
    a=z[ids,None,None,:]
    b=z[donors]
    return np.concatenate([abs(a-b),(a+b)/2],axis=-1).reshape(-1,2*z.shape[1])


def fit(x,y,g,t,q,seed=831):
    assert not np.isin(g[t],g[q]).any()
    z=((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-6,6).astype('float32')
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.45,random_state=seed).split(t,groups=g[t]))
    bank,examples=t[a],t[b]
    rng=np.random.default_rng(seed-196)
    examples=rng.choice(examples,min(1400,len(examples)),replace=False)
    assert not np.isin(g[examples],g[bank]).any()
    tr=references(z,y,bank,examples,seed-138)
    rel=(y[examples,None,None]==y[tr]).ravel()
    model=classifier().fit(features(z,examples,tr),rel)
    raw=classifier().fit(z[t],y[t]).predict_proba(z[q])[:,1]
    donor=references(z,y,bank,q,seed-137)
    scores=[]
    for i in range(0,len(q),128):
        sl=slice(i,i+128)
        sim=model.predict_proba(features(z,q[sl],donor[sl]))[:,1].reshape(-1,2,8)
        scores.append((1-sim[:,0,:]+sim[:,1,:])/2)
    score=np.vstack(scores)
    return np.column_stack([raw,score[:,:4].mean(1),score[:,4:].mean(1),score.mean(1)])


def run():
    raw,y,g,f=data()
    names,h,full=banks(raw,y,f)
    ref=latest_reference(y,g,f,names,h,full)
    _,oy,og,rows,of,_,oldbase,_,_,_,_,old,_=aligned()
    ix=np.arange(len(y))
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    inc=prepared_vote(original_prediction(ix,f,g,y,'full'),ref,nine)
    with np.load(HISTORY/'historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],rows)
        past=z['nine']
    with np.load(AUDIT/'oof.npz') as z:
        cert=z['certified_v131_equals_v96']
        trusted=z['certified_nine'][cert].astype(bool)
    x=design(raw)
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    report={}
    for (tag,ids,fold,base,p),panel in zip([
            ('full',ix,f,inc,full[:,names.index('mitra96')]),('historical',rows,of,past,old[:,0])],panels,strict=True):
        scores=protocol(x,y,g,ids,fold,tag,fit,MODES,TAG)
        common=(base!=y[ids])&((np.column_stack(panel)>=.5)==base[:,None]).all(1)
        new=~np.isin(g[ids],og)
        entry={}
        for j,name in enumerate(MODES):
            for w in (1.,.25):
                s=(1-w)*p+w*scores[:,j]
                pred=(s>=.5) if w==1 else np.where((p>=.5)==base,s>=.5,base)
                val=stats(pred,base,y[ids],g[ids],fold)
                val['common_fixed']=int((common&(pred==y[ids])).sum())
                if tag=='full':
                    zpred=(s[cert]>=.5) if w==1 else np.where((p[cert]>=.5)==trusted,s[cert]>=.5,trusted)
                    val['certified']=stats(zpred,trusted,y[cert],g[cert],f[cert])
                    val['new_orders']=stats(pred[new],base[new],y[ids][new],g[ids][new],fold[new])
                entry[name+'_'+str(w)]=val
        report[tag]=entry
        print(tag,[(k,v['corrected'],v['harmed'],v['common_fixed']) for k,v in entry.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/passenger_pair_outcomes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
