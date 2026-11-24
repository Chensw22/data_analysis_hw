"""Marginalize local residual risk along the actual cabin numbers of each deck-side."""
import json
from itertools import product

import numpy as np
from scipy.special import softmax

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, BILLS, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_kernel_repair_overlap_n24 import from_cache

SETTINGS = [(0.,1.), *product((.35,.75), (1.,8.,32.,128.))]


def geometry(raw):
    cabin = raw.Cabin.str.extract(r'^([^/]+)/(\d+)/([PS])$')
    valid = cabin.notna().all(axis=1).to_numpy()
    lane = np.full(len(raw),-1,int)
    lane[valid] = np.unique((cabin.loc[valid,0]+cabin.loc[valid,2]).to_numpy(),return_inverse=True)[1]
    position = np.zeros(len(raw),int)
    position[valid] = cabin.loc[valid,1].astype(int)
    shuffled = position.copy()
    rng = np.random.default_rng(193)
    for j in np.unique(lane[valid]):
        use = lane==j
        positions,inv = np.unique(position[use],return_inverse=True)
        shuffled[use] = rng.permutation(positions)[inv]
    return lane,position,shuffled


def messages(log_plus, log_minus, rho):
    # Each column is an independent cabin lane and parameter setting.
    # State means are +1/-1 expectations; emission factors are relative to the fixed base.
    n,width = log_plus.shape
    pre = np.empty((n,width))
    m = np.zeros(width)
    evidence = np.zeros(width)
    delta = .5*(log_plus-log_minus)
    for j in range(n):
        pre[j] = m
        evidence += np.logaddexp(np.log1p(m)+log_plus[j],np.log1p(-m)+log_minus[j])-np.log(2.)
        m = rho*np.tanh(np.arctanh(m)+delta[j])
    posterior = np.empty_like(pre)
    m.fill(0)
    for j in range(n-1,-1,-1):
        posterior[j] = np.tanh(np.arctanh(pre[j])+delta[j]+np.arctanh(m))
        m = rho*np.tanh(np.arctanh(m)+delta[j])
    return posterior,evidence


def local_effect(lane,position,t,y,p):
    valid = lane>=0
    t = t[valid[t]]
    lanes = lane.max()+1
    n = position[valid].max()+1
    shape = (n,lanes,len(SETTINGS))
    log_plus, log_minus = np.zeros(shape),np.zeros(shape)
    flat = position[t]*lanes+lane[t]
    # If y=1 the relative likelihood factor is 1 +/- a*(1-p);
    # if y=0 it is 1 -/+ a*p. At no evidence the marginal returns p exactly.
    for k,(a,_) in enumerate(SETTINGS):
        residual = y[t]-p[t]
        for sign,arr in [(1,log_plus),(-1,log_minus)]:
            arr[:,:,k] = np.bincount(flat, weights=np.log1p(sign*a*residual), minlength=n*lanes).reshape(n,lanes)
    amp,length = np.array(SETTINGS).T
    rho = np.tile(np.exp(-1/length),lanes)
    posterior, evidence = messages(log_plus.reshape(n,-1),log_minus.reshape(n,-1),rho)
    prior = np.r_[.5,np.full(len(SETTINGS)-1,.5/(len(SETTINGS)-1))]
    weight = softmax(evidence.reshape(lanes,-1)+np.log(prior),axis=1)
    influence = np.sum(posterior.reshape(shape)*amp[None,None,:]*weight[None,:,:],axis=2)
    out = np.zeros(len(lane))
    out[valid] = influence[position[valid],lane[valid]]
    return out


def run():
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    raw_idx = names.index('mitra96')
    own = full[:,raw_idx]
    source = half[:,raw_idx]
    lane,pos,shuffled = geometry(raw)
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    options = dict(real=(pos,False),swapped=(shuffled,False),real_by_activity=(pos,True),
                   swapped_by_activity=(shuffled,True))
    tilt = {key:np.full(len(y),np.nan) for key in options}
    for f in np.unique(fold):
        q = np.flatnonzero(fold==f)
        t = np.flatnonzero((fold%2==f%2) & (fold!=f))
        assert not np.isin(g[t],g[q]).any()
        for key,(loc,stratified) in options.items():
            channel = np.where(lane<0,-1,2*lane+paid) if stratified else lane
            tilt[key][q] = local_effect(channel,loc,t,y,source)[q]
    _, rivals = from_cache('full',np.arange(len(y)),fold,y,g)
    common = ((full>=.5)!=y[:,None]).all(1)
    missed = common & ((rivals>=.5)!=y[:,None]).all(1)
    hard = common[rows] & (champ!=oy)
    assert missed.sum()==243 and common.sum()==257
    unseen = ~np.isin(g,og)
    report = {}
    for key,change in tilt.items():
        assert np.isfinite(change).all() and np.max(np.abs(change))<=.75
        report[key] = {}
        for w in (1.,.25):
            p = np.where((own>=.5)==ref,own+w*own*(1-own)*change>=.5,ref)
            oldp = old[:,0]
            a = np.where((oldp>=.5)==champ,oldp+w*oldp*(1-oldp)*change[rows]>=.5,champ)
            v = dict(full=stats(p,ref,y,g,fold), historical=stats(a,champ,oy,og,ofold),
                     unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                     common_fixed=int((common & (p==y)).sum()),unreached_fixed=int((missed & (p==y)).sum()),
                     persistent_both=int((hard & (p[rows]==oy) & (a==oy)).sum()))
            report[key][str(w)] = v
            print(key,w,[v[s]['net'] for s in ('historical','full','unseen')],v['unreached_fixed'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/cabin_markov.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
