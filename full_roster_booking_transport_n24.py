"""Predict from nearby labeled bookings with or without joint passenger matching."""
import json

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_cohort_prevalence_n23 import fraction
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities


def parties(x, group):
    unique, mapping = np.unique(group,return_inverse=True)
    members = np.split(np.argsort(mapping,kind='stable'),np.cumsum(np.bincount(mapping))[:-1])
    projection = np.random.default_rng(785).choice([-1.,1.],size=(x.shape[1],24))/np.sqrt(x.shape[1])
    projected = x @ projection
    features = [np.r_[np.quantile(projected[m],[.2,.5,.8],axis=0).ravel(),np.log(len(m))] for m in members]
    assert len(unique) == len(members)
    return members, mapping, np.stack(features)


def coupling(cost):
    # Uniform source and target mass makes each passenger count equally, even for different party sizes.
    kernel = np.exp(-np.minimum((cost-cost.min(axis=(1,2),keepdims=True))/.85,18.))
    rows, cols = cost.shape[1:]
    v = np.ones((len(cost),cols))
    for step in range(250):
        u = 1/(rows*np.einsum('bij,bj->bi',kernel,v))
        v = 1/(cols*np.einsum('bij,bi->bj',kernel,u))
        if step in (9,29,79,149,249):
            plan = u[:,:,None]*kernel*v[:,None,:]
            if np.max(abs(plan.sum(2)-1/rows))<1.e-5:
                break
    assert np.max(abs(plan.sum(1)-1/cols))<1.e-6
    return plan, float(np.max(abs(plan.sum(2)-1/rows)))


def vote(x, y, group, fold, count=48, panel=None):
    scalar = y.ndim == 1
    outcomes = y[:, None] if scalar else y
    if panel is None:
        panel = np.zeros_like(fold)
    members, group_id, summary = parties(x,group)
    sizes = np.array([len(m) for m in members])
    first = np.array([m[0] for m in members])
    ordered_fold = fold[first]
    assert np.array_equal(ordered_fold[group_id],fold)
    keys = ['individual','booking_rate','independent','transport']
    out = {k:np.full(outcomes.shape,np.nan) for k in keys}
    max_error = 0.
    for f in np.unique(fold):
        q = np.flatnonzero(fold == f)
        assert np.all(panel[q] == panel[q[0]])
        donors = (fold != f) & (panel == panel[q[0]])
        t = np.flatnonzero(donors)
        td, qd = np.flatnonzero(donors[first]), np.flatnonzero(ordered_fold == f)
        assert not np.isin(group[t],group[q]).any()
        dist = cdist(x[q],x[t],'sqeuclidean')
        closest = np.argpartition(dist,count-1,axis=1)[:,:count]
        d = np.take_along_axis(dist,closest,axis=1)
        w = softmax(-d/.85,axis=1)
        out['individual'][q] = np.einsum('ij,ijk->ik', w, outcomes[t[closest]])
        distance = cdist(summary[qd],summary[td],'sqeuclidean')
        near = td[np.argpartition(distance,count-1,axis=1)[:,:count]]
        source = np.repeat(qd,count)
        target = near.ravel()
        # Store a separate match for every target passenger and candidate order.
        cost_out = np.zeros((len(qd),count))
        prob = {k:np.zeros((len(y),count,outcomes.shape[1])) for k in keys[1:]}
        source_size,target_size = sizes[source],sizes[target]
        for a,b in np.unique(np.column_stack([source_size,target_size]),axis=0):
            edges = np.flatnonzero((source_size==a)&(target_size==b))
            for offset in range(0,len(edges),2048):
                idx = edges[offset:offset+2048]
                src = np.stack([members[k] for k in source[idx]])
                dst = np.stack([members[k] for k in target[idx]])
                left,right = x[src],x[dst]
                costs = np.maximum(0,(left**2).sum(2)[:,:,None]+(right**2).sum(2)[:,None,:]
                                   -2*np.matmul(left,right.transpose(0,2,1)))
                plan,error = coupling(costs)
                max_error = max(max_error,error)
                cost_out[idx//count,idx%count] = (plan*costs).sum(axis=(1,2))
                labels = outcomes[dst].astype(float)
                prob['booking_rate'][src,(idx%count)[:,None]] = labels.mean(1)[:,None,:]
                independent = softmax(-costs/.85,axis=2)
                prob['independent'][src,(idx%count)[:,None]] = independent @ labels
                conditional = plan/plan.sum(axis=2,keepdims=True)
                prob['transport'][src,(idx%count)[:,None]] = conditional @ labels
        order_weight = softmax(-cost_out/.85,axis=1)
        lookup = np.searchsorted(qd,group_id[q])
        for k in keys[1:]:
            out[k][q] = np.einsum('ijk,ij->ik', prob[k][q],order_weight[lookup])
        print('matched fold',int(f),'max marginal error',max_error,flush=True)
    for p in out.values():
        assert np.isfinite(p).all() and p.min()>=-1.e-8 and p.max()<=1+1.e-8
    return ({k: p[:, 0] for k, p in out.items()} if scalar else out), max_error


def run():
    raw,y,group,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,group,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,group,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    scores, error = vote(passenger_vectors(raw),y,group,fold)
    directory = ROOT/'artifacts/nov24_booking_transport'
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/'oof.npz',passenger=raw.PassengerId.to_numpy(str),y=y,fold=fold,**scores)
    _,ids,size = np.unique(group,return_inverse=True,return_counts=True)
    multi = size[ids]>1
    shared = ((full>=.5)!=y[:,None]).all(1)
    correct_all = ((full>=.5)==y[:,None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    result={}
    for key,p in scores.items():
        learned=np.empty(len(y))
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
                t,q=outer[a],outer[b]
                assert not np.isin(group[t],group[q]).any()
                alpha=fraction(gate[t],p[t],y[t])
                learned[q]=(1-alpha)*gate[q]+alpha*p[q]
        for k,v in [('direct',p),('quarter',.75*gate+.25*p),('learned',learned)]:
            delta=((v>=.5)==y).astype(int)-(ref==y)
            result[key+'_'+k]=dict(accuracy=float(((v>=.5)==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                               common_fixed=int((delta[shared]==1).sum()),unanimous_harmed=int((delta[correct_all]==-1).sum()),
                               multi_net=int(delta[multi].sum()),additional_net=int(delta.sum()-delta[dev].sum()),
                               net_by_fold=[int(delta[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(delta,group))
    report=dict(number_of_neighbors=48,max_transport_marginal_error=error,results=result)
    (ROOT/'reports/nov23_matched_learning_curve/booking_transport.json').write_text(json.dumps(report,indent=2)+'\n')
    for k,v in result.items(): print(k,round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['common_fixed'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
