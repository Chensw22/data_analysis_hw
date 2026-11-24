"""Compare independent evidence, fitted dependency trees and an exact average over trees."""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import expit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_solo_coverage_n24 import cached

MODES = ('independent','common_tree','class_tree','tree_average')


def encode(raw,g):
    cabin = raw.Cabin.str.split('/',expand=True)
    size = raw.groupby(g).PassengerId.transform('size')
    cats = [raw[k] for k in ('HomePlanet','Destination','VIP','CryoSleep')]+[cabin[0],cabin[2]]
    values = [pd.factorize(s.astype('string').fillna('?'),sort=True)[0] for s in cats]
    for s,cuts in [(raw.Age,[5,13,18,25,35,50]),(size,[2,3,5]),
                   (pd.to_numeric(cabin[1]),[100,300,700,1200,1600])]+[
                       (raw[k],[1,50,250,700,1500,3000]) for k in BILLS]:
        values.append(np.where(s.isna(),0,1+np.digitize(s,cuts)))
    return np.column_stack(values)


def edge_partition(weights):
    # Any principal cofactor of the weighted graph Laplacian sums the tree weights.
    lap = -weights.copy()
    idx = np.arange(weights.shape[-1])
    lap[...,idx,idx] = weights.sum(-1)
    sign,val = np.linalg.slogdet(lap[...,:-1,:-1])
    assert np.all(sign>0)
    return val


def check_normalization():
    # On three binary fields the full outcome space and all three trees can be enumerated.
    cells = np.array(list(itertools.product((0,1),repeat=3)))
    mass = np.arange(1,9)/36
    probs = np.column_stack([np.bincount(cells[:,i],weights=mass)[cells[:,i]] for i in range(3)])
    edge = np.ones((8,3,3))
    for i,j in itertools.combinations(range(3),2):
        k = 2*cells[:,i]+cells[:,j]
        edge[:,i,j] = edge[:,j,i] = np.bincount(k,weights=mass)[k]/(probs[:,i]*probs[:,j])
    w = np.array([[0.,1,2],[1,0,3],[2,3,0]])
    predicted = probs.prod(1)*np.exp(edge_partition(w*edge)-edge_partition(w))
    brute = probs.prod(1)*(2*edge[:,0,1]*edge[:,0,2]+3*edge[:,0,1]*edge[:,1,2]+6*edge[:,0,2]*edge[:,1,2])/11
    np.testing.assert_allclose(predicted,brute)
    np.testing.assert_allclose(predicted.sum(),1.)


def fit(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    width = x.shape[1]
    dims = x.max(0)+1
    prior = [(np.bincount(x[t,j],minlength=dims[j])+1)/(len(t)+dims[j]) for j in range(width)]
    classes = []
    for label in (False,True):
        rows = t[y[t]==label]
        n = len(rows)
        alpha = 24.
        marg = [(np.bincount(x[rows,j],minlength=dims[j])+alpha*prior[j])/(n+alpha) for j in range(width)]
        log_base = np.full(len(q),np.log((n+1)/(len(t)+2)))
        for j in range(width):
            log_base += np.log(marg[j][x[q,j]])
        assoc = np.zeros((width,width))
        ratio = np.ones((len(q),width,width))
        for i,j in itertools.combinations(range(width),2):
            codes = x[rows,i]*dims[j]+x[rows,j]
            counts = np.bincount(codes,minlength=dims[i]*dims[j]).reshape(dims[i],dims[j])
            joint = (counts+alpha*np.outer(prior[i],prior[j]))/(n+alpha)
            rel = joint/np.outer(marg[i],marg[j])
            # First-order finite-sample correction to empirical mutual information.
            mi = (joint*np.log(rel)).sum()-((counts.sum(0)>0).sum()-1)*((counts.sum(1)>0).sum()-1)/(2*n)
            assoc[i,j] = assoc[j,i] = mi
            ratio[:,i,j] = ratio[:,j,i] = rel[x[q,i],x[q,j]]
        classes.append((n,log_base,assoc,ratio))
    common = sum(n*mi for n,_,mi,_ in classes)/len(t)
    likelihood = []
    for n,base,mi,ratio in classes:
        logs = [base]
        for structure in (common,mi):
            # Strictly positive costs keep measured zero-dependence edges available.
            e = minimum_spanning_tree(1+structure.max()-structure).tocoo()
            assert len(e.row)==width-1
            logs.append(base+np.log(ratio[:,e.row,e.col]).sum(1))
        weights = np.exp(np.clip(8*mi,-5,5))
        np.fill_diagonal(weights,0)
        logs.append(base+edge_partition(weights*ratio)-edge_partition(weights))
        likelihood.append(np.column_stack(logs))
    return expit(likelihood[1]-likelihood[0])


def run():
    check_normalization()
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    result = {}
    for tag,ix,f,base,p in [('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                            ('historical',rows,of,champ,old[:,0])]:
        s = protocol(encode(raw,g),y,g,ix,f,tag,fit,MODES,'dependency_tree')
        v = np.column_stack([cached(k,tag,ix,f,g,y) for k in ('response_channels','broad_error_transport_projection')])
        six_miss = (base!=y[ix]) & ((v>=.5)==base[:,None]).all(1)
        policies = {}
        for j,name in enumerate(MODES):
            for w in (1.,.25):
                pred = s[:,j]>=.5 if w==1 else np.where((p>=.5)==base,(1-w)*p+w*s[:,j]>=.5,base)
                entry = stats(pred,base,y[ix],g[ix],f)
                entry.update(six_fixed=int((six_miss & (pred==y[ix])).sum()),
                             hard_fixed=int((hard[ix] & (base!=y[ix]) & (pred==y[ix])).sum()))
                if tag=='full':
                    new = ~np.isin(g[ix],og)
                    entry['new_net'] = int((pred[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
                policies[f'{name}_{w}'] = entry
                print(tag,name,w,entry['corrected'],entry['harmed'],entry['six_fixed'],entry['hard_fixed'],flush=True)
        result[tag] = policies
    (ROOT/'reports/nov23_matched_learning_curve/dependency_tree.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
