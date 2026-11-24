"""Replicate third-order neural exceptions with independent initializations and batch orders."""
import json

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import compact_features
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_factor_interactions_n24 import encode, train_order
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_gate_stability_n24 import fit_probability


def run():
    torch.set_num_threads(2)
    raw,y,group,fold=data()
    names,half,full=banks(raw,y,fold)
    ref=latest_reference(y,group,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,y,group,fold)
    frame=compact_features(raw)
    _,ix,size=np.unique(group,return_inverse=True,return_counts=True)
    frame['BookingSize']=size[ix]
    frame['Position']=raw.PassengerId.str.split('_').str[1].astype(int)
    directory=ROOT/'artifacts/nov24_field_interactions'
    seeds=[426,1653,5201,7217,9648]
    preds=np.full((len(y),len(seeds)),np.nan)
    with np.load(directory/'oof.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        preds[:,0]=z['triple']
    for f in np.unique(fold):
        t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        assert not np.isin(group[t],group[q]).any()
        ids,weights,width=encode(frame,t)
        for j,seed in enumerate(seeds[1:],1):
            preds[q,j]=train_order(ids,weights,width,y,t,q,3,seed+int(f))
        print('replicated fold',f,flush=True)
    assert np.isfinite(preds).all()
    np.savez_compressed(directory/'third_order_seeds.npz',seeds=seeds,scores=preds,fold=fold,y=y,
                        passenger=raw.PassengerId.to_numpy(str))
    dissent=(preds>=.5)!=ref[:,None]
    votes=dissent.sum(1)
    shared=((full>=.5)!=y[:,None]).all(1)
    same_right=((full>=.5)==y[:,None]).all(1)
    initial=shared & dissent[:,0]
    active=votes>0
    neural=np.where(ref[:,None],1-preds,preds)
    xx=np.column_stack([neural.mean(1),neural.min(1),neural.max(1),neural.std(1),votes/len(seeds),
                        np.where(ref,1-gate,gate),ref,((full>=.5)!=ref[:,None]).mean(1)])
    unanimous_votes=(full>=.5).all(1)|(full<.5).all(1)
    policies={'neural_mean':preds.mean(1)>=.5,
              'quarter_blend':.75*gate+.25*preds.mean(1)>=.5,
              'all_seeds_dissent':ref^(votes==len(seeds)),
              'unanimous_against_unanimous':ref^((votes==len(seeds))&unanimous_votes),
              'calibrated':ref.copy()}
    # "All original experts agree" depends only on their votes, not on validation labels.
    np.testing.assert_array_equal(shared|same_right,unanimous_votes)
    for f in np.unique(fold):
        outer=np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            t,q=outer[a],outer[b]
            t,q=t[active[t]],q[active[q]]
            assert not np.isin(group[t],group[q]).any()
            corrected=fit_probability(xx,y!=ref,t,xx[q])>=.5
            policies['calibrated'][q]^=corrected
    repeat={}
    for minimum in [1,3,5]:
        selected=votes>=minimum
        repeat[str(minimum)]=dict(uniquely_correct=int((selected&shared).sum()),
                                  unanimous_correct_harmed=int((selected&same_right).sum()),
                                  original_28_retained=int((selected&initial).sum()))
    result={}
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    for k,prediction in policies.items():
        delta=(prediction==y).astype(int)-(ref==y)
        result[k]=dict(accuracy=float((prediction==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                       common_fixed=int((delta[shared]==1).sum()),additional_net=int(delta.sum()-delta[dev].sum()),
                       fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(delta,group))
    report=dict(seeds=seeds,repeat=repeat,results=result)
    (ROOT/'reports/nov23_matched_learning_curve/factor_replicates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    run()
