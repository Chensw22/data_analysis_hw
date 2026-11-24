"""Test when to accept a contrary higher-order neural interaction on unseen bookings."""
import json

import numpy as np
import torch
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import compact_features
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_factor_interactions_n24 import encode, train_order
from full_roster_gate_probability_n23 import gate_probabilities, proposal_design, EVIDENCE
from full_roster_gate_stability_n24 import fit_probability


def evidence(p, scores, nets):
    pick = p>=.5
    sign = np.where(pick,-1.,1.)[:,None]
    higher = logit(nets.clip(.002,.998))*sign
    base_evidence = logit(p.clip(.002,.998))*sign[:,0]
    opposite = (nets>=.5) != pick[:,None]
    votes = ((scores>=.5) != pick[:,None]).sum(axis=1)
    margin = higher[:,2]-higher[:,0]
    x = np.column_stack([base_evidence, higher, votes/19, pick, margin, margin*base_evidence])
    return x, opposite.any(1)


def run():
    torch.set_num_threads(2)
    raw,y,group,fold=data()
    names,half,full=banks(raw,y,fold)
    ref=latest_reference(y,group,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,y,group,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    directory=ROOT/'artifacts/nov24_field_interactions'
    with np.load(directory/'oof.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        nets=np.column_stack([z[k] for k in ('additive','pair','triple')])
    frame=compact_features(raw)
    _,ix,size=np.unique(group,return_inverse=True,return_counts=True)
    frame['BookingSize']=size[ix]
    frame['Position']=raw.PassengerId.str.split('_').str[1].astype(int)
    small=np.zeros_like(nets)
    for h in range(2):
        t,q=np.flatnonzero(fold%2!=h),np.flatnonzero(fold%2==h)
        assert not np.isin(group[t],group[q]).any()
        ids,weights,width=encode(frame,t)
        for i in range(3):
            small[q,i]=train_order(ids,weights,width,y,t,q,i+1,426+h)
    np.savez_compressed(directory/'half.npz',passenger=raw.PassengerId.to_numpy(str),panel=fold%2,y=y,scores=small)
    ids=[names.index(k) for k in EVIDENCE]
    hdis,hx=proposal_design(half[:,:5],half[:,ids])
    fx,active=evidence(gate,full,nets)
    base=half[:,4]>=.5
    policies={k:ref.copy() for k in ('half_calibration','matched_calibration','agreed','stacked')}
    for f in np.unique(fold):
        train=(fold!=f)&(fold%2==f%2)
        outer=np.flatnonzero(fold==f)
        # Each small-model reference on a training passenger also excludes that passenger's fold.
        hp=half[:,4].copy()
        for h in np.unique(fold[train]):
            t=np.flatnonzero(train & (fold!=h) & hdis)
            q=np.flatnonzero(train & (fold==h) & hdis)
            assert not np.isin(group[t],group[q]).any()
            prob=fit_probability(hx,y!=base,t,hx[q])
            hp[q]=np.where(base[q],1-prob,prob)
        learn_x,learn_active=evidence(hp,half,small)
        t=np.flatnonzero(train&learn_active)
        target=y!=(hp>=.5)
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            donors,q=outer[a],outer[b]
            donors,q=donors[active[donors]],q[active[q]]
            assert not np.isin(group[np.r_[t,donors]],group[q]).any()
            earlier=fit_probability(learn_x,target,t,fx[q])
            matched=fit_probability(fx,y!=ref,donors,fx[q])
            x=np.vstack([learn_x[t],fx[donors]])
            z=np.r_[target[t],y[donors]!=ref[donors]]
            combined=fit_probability(x,z,np.arange(len(z)),fx[q])
            for k,s in [('half_calibration',earlier),('matched_calibration',matched),
                        ('agreed',np.minimum(earlier,matched)),('stacked',combined)]:
                policies[k][q]^=s>=.5
    unanimous_wrong=((full>=.5)!=y[:,None]).all(1)
    unanimous_right=((full>=.5)==y[:,None]).all(1)
    oracle=((nets>=.5)==y[:,None]).any(1)
    reachable=unanimous_wrong&oracle
    comparable=unanimous_right&((nets>=.5)!=y[:,None]).any(1)
    report=dict(eligible=int(active.sum()), eligible_reference_errors=int((active&(y!=ref)).sum()),
                common_errors_coverable=int(reachable.sum()), unanimous_correct_challenged=int(comparable.sum()),
                result={})
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    for k,pred in policies.items():
        delta=(pred==y).astype(int)-(ref==y)
        report['result'][k]=dict(accuracy=float((pred==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                                common_fixed=int((delta[unanimous_wrong]==1).sum()),unanimous_harmed=int((delta[unanimous_right]==-1).sum()),
                                additional_net=int(delta.sum()-delta[dev].sum()),fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
                                ci_pp=interval(delta,group))
    (ROOT/'reports/nov23_matched_learning_curve/factor_dissent.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    run()
