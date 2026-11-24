"""Integrate the unobserved outcome of an adult companion when predicting a passenger."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_two_view_reliability_n24 import design,tree

TAG = 'adult_companion_outcome'
MODES = ('focal','direct_pair','integrated_pair','shuffled_outcome')


def edges(raw,groups):
    focal,peer = [],[]
    adult = raw.Age.ge(18).to_numpy()
    for members in pd.Series(np.arange(len(groups))).groupby(groups).indices.values():
        adults = members[adult[members]]
        for i in members:
            for j in adults[adults!=i]:
                focal.append(i)
                peer.append(j)
    return np.asarray(focal),np.asarray(peer)


def predict(inputs,y,g,t,q):
    x,focal,peer,placebo_group = inputs
    assert not np.isin(g[t],g[q]).any()
    train = np.isin(focal,t)&np.isin(peer,t)
    query = np.isin(focal,q)
    f,d = focal[train],peer[train]
    a,b = focal[query],peer[query]
    xx = np.column_stack([x[f],x[d]])
    zz = np.column_stack([x[a],x[b]])
    weight = 1/np.bincount(f,minlength=len(y))[f]
    def model(fields,labels):
        return tree(13).fit(fields,labels,sample_weight=weight)
    point = model(x[f],y[f]).predict_proba(x[a])[:,1]
    direct = model(xx,y[f]).predict_proba(zz)[:,1]

    def integrate(outcome):
        adult = model(xx,outcome[d]).predict_proba(zz)[:,1]
        conditional = model(np.column_stack([xx,outcome[d]]),y[f])
        events = [conditional.predict_proba(np.column_stack([zz,np.full(len(a),v)]))[:,1] for v in (0,1)]
        return (1-adult)*events[0]+adult*events[1]

    real = integrate(y)
    draw = np.random.default_rng(878)
    control = y.copy()
    donors = np.unique(d)
    for level in np.unique(placebo_group[donors]):
        ix = donors[placebo_group[donors]==level]
        control[ix] = draw.permutation(y[ix])
    shuffled = integrate(control)
    counts = np.bincount(a,minlength=len(y))[q]
    scores = []
    for p in (point,direct,real,shuffled):
        sums = np.bincount(a,weights=p,minlength=len(y))[q]
        scores.append(np.divide(sums,counts,out=np.full(len(q),.5),where=counts>0))
    return np.column_stack(scores)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(ref,gate>=.5)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    assert hard.sum()==147
    f,d = edges(raw,g)
    assert (raw.Age.to_numpy()[d]>=18).all() and not (f==d).any()
    np.testing.assert_array_equal(g[f],g[d])
    place = pd.MultiIndex.from_arrays([raw.HomePlanet.fillna('?'),raw.CryoSleep.fillna('missing').astype(str),
                                       raw[BILLS].gt(0).any(axis=1)]).factorize()[0]
    inputs = (np.column_stack(design(raw,g)),f,d,place)
    a = protocol(inputs,y,g,rows,of,'historical',predict,MODES,TAG)
    b = protocol(inputs,y,g,np.arange(len(y)),fold,'full',predict,MODES,TAG)
    eligible = np.isin(np.arange(len(y)),f)
    children = eligible & raw.Age.lt(13).to_numpy()
    both = children & raw.CryoSleep.eq(False).to_numpy() & raw[BILLS].eq(0).all(axis=1).to_numpy()
    unseen = ~np.isin(g,og)
    report = {}
    for name,mask in [('any_passenger',eligible),('under_13',children),('awake_child_zero',both)]:
        report[name] = dict(eligible=int(mask.sum()), base_errors=int((mask & (ref!=y)).sum()),
                            eligible_unseen=int((mask & unseen).sum()), eligible_hard=int((mask & hard).sum()))
        for j,mode in enumerate(MODES):
            report[name][mode] = {}
            for weight in (1.,.25):
                pred = np.where(mask,weight*b[:,j]+(1-weight)*gate>=.5,ref)
                old_mask = mask[rows] & ((weight==1.) | ((old[:,0]>=.5)==champ))
                past = np.where(old_mask,weight*a[:,j]+(1-weight)*old[:,0]>=.5,champ)
                fixed = hard & (pred==y)
                report[name][mode][str(weight)] = dict(full=stats(pred,ref,y,g,fold),
                     historical=stats(past,champ,oy,og,of),unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                     remaining_fixed=int(fixed.sum()),repeated=int((fixed[rows] & (champ!=oy) & (past==oy)).sum()),
                     fixed_ids=raw.loc[fixed,'PassengerId'].tolist())
            r = report[name][mode]['1.0']
            print(name,mode,[r[k]['net'] for k in ('full','historical','unseen')],r['remaining_fixed'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
