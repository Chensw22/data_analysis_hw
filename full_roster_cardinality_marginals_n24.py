"""Learn a booking count distribution and integrate its uncertainty in passenger-level decisions."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_disagreement_calibration_n24 import stats
from full_roster_order_coverage_n24 import comparison,profile


@lru_cache(maxsize=None)
def states(n):
    return ((np.arange(1<<n)[:,None]>>np.arange(n))&1).astype(float)


def system(orders,p,raw):
    width = max(map(len,orders))+1
    features = np.zeros((len(orders),width,9))
    log_prior = np.full((len(orders),width),-np.inf)
    given_count = np.zeros((len(p),width))
    index = np.zeros(len(p),int)
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    asleep = raw.CryoSleep.eq(True).to_numpy()
    for i,ix in enumerate(orders):
        n = len(ix)
        prob = p[ix].clip(1e-5,1-1e-5)
        outcomes = states(n)
        k = outcomes.sum(1).astype(int)
        joint = np.exp(outcomes @ np.log(prob)+(1-outcomes) @ np.log1p(-prob))
        pmf = np.bincount(k,weights=joint)
        log_prior[i,:n+1] = np.log(pmf)
        for c in range(n+1):
            present = k==c
            given_count[ix,c] = (joint[present]/pmf[c]) @ outcomes[present]
        index[ix] = i
        u = (np.arange(n+1)-prob.sum())/np.sqrt(n)
        attrs = [1,prob.mean()-.5,prob.std(),np.log(n),paid[ix].mean(),asleep[ix].mean()]
        features[i,:n+1] = np.column_stack([u[:,None]*attrs,u*u,(np.arange(n+1)==0)&(n>1),
                                            (np.arange(n+1)==n)&(n>1)])
    np.testing.assert_allclose((np.exp(log_prior[index])*given_count).sum(1),p.clip(1e-5,1-1e-5),atol=1e-10)
    return features,log_prior,given_count,index


def fit(sys,ords,people,labels,counts,mode):
    x,prior,risk,idx = sys
    lookup = np.searchsorted(ords,idx[people])
    np.testing.assert_array_equal(ords[lookup],idx[people])
    z0,xx = prior[ords],x[ords]
    obs = xx[np.arange(len(ords)),counts[ords]].sum(0)
    marg = risk[people]
    def objective(w):
        z = z0+xx@w
        pmf = softmax(z,axis=1)
        if mode=='count':
            loss = (logsumexp(z,axis=1)-z[np.arange(len(ords)),counts[ords]]).sum()
            gradient = np.einsum('ik,ikj->j',pmf,xx)-obs
        else:
            q = (pmf[lookup]*marg).sum(1).clip(1e-9,1-1e-9)
            loss = -np.where(labels[people],np.log(q),np.log1p(-q)).sum()
            coeff = (q-labels[people])/(q*(1-q))
            by_person = coeff[:,None]*pmf[lookup]*(marg-q[:,None])
            by_order = np.zeros_like(pmf)
            np.add.at(by_order,lookup,by_person)
            gradient = np.einsum('ik,ikj->j',by_order,xx)
        return loss+10*w@w,gradient+20*w
    fitted = minimize(objective,np.zeros(xx.shape[-1]),jac=True,method='L-BFGS-B')
    assert fitted.success,fitted.message
    return fitted.x


def assign(orders,k,prob):
    pred = np.zeros(len(prob),bool)
    for ix,count in zip(orders,k,strict=True):
        rank = np.argsort(-prob[ix],kind='stable')
        pred[ix[rank[:count]]] = True
    return pred


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    j = names.index('mitra96')
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    orders = list(raw.groupby(g).indices.values())
    # groupby.indices preserves the input row order inside each booking.
    sizes = np.array([len(ix) for ix in orders])
    of = np.array([fold[ix[0]] for ix in orders])
    assert all(np.unique(fold[ix]).size==1 for ix in orders)
    true_count = np.array([y[ix].sum() for ix in orders])
    original = np.array([ref[ix].sum() for ix in orders])
    prior = system(orders,h[:,j],raw)
    full_sys = system(orders,gate,raw)
    x,lp,risk,index = full_sys
    baseline_count_hard = assign(orders,original,gate)
    np.testing.assert_array_equal(baseline_count_hard,ref)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    target = profile(pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId','Age']))[0]
    cells = profile(raw)[0]
    multi = sizes[index]>1
    report = dict(base_correct=int((ref==y).sum()),local_fixed_count_ceiling=int(len(y)-np.abs(original-true_count).sum()),
                  multi_bookings=int((sizes>1).sum()),multi_passengers=int(multi.sum()),
                  multi_baseline_count_correct=int(((sizes>1)&(original==true_count)).sum()),
                  multi_base_hard=int((hard&multi).sum()))
    def record(name,pred):
        d = (pred==y).astype(int)-(ref==y)
        k = np.bincount(index,weights=pred).astype(int)
        r = dict(full=stats(pred,ref,y,g,fold),multi=stats(pred[multi],ref[multi],y[multi],g[multi],fold[multi]),
                 multi_count_correct=int(((sizes>1)&(k==true_count)).sum()),
                 solo_hard_fixed=int((hard & ~multi & (pred==y)).sum()),multi_hard_fixed=int((hard & multi & (pred==y)).sum()),
                 weighted_pp=comparison(cells,target,np.ones(len(y),bool),{'net':d})['results']['net']['covered_weighted_pp'])
        report[name] = r
        print(name,r['full']['net'],r['multi']['corrected'],r['multi']['harmed'],r['multi_count_correct'],flush=True)
    record('oracle_count_prefix',assign(orders,true_count,gate))
    record('oracle_count_marginal',risk[np.arange(len(y)),true_count[index]]>=.5)
    for variant in ('half_source','within_outer'):
        for mode in ('count','passenger'):
            post = np.full(lp.shape,np.nan)
            for f in np.unique(of):
                query = np.flatnonzero(of==f)
                if variant=='half_source':
                    splits = [(np.flatnonzero((of%2==f%2)&(of!=f)),query)]
                    source = prior
                else:
                    splits = [(query[a],query[b]) for a,b in GroupKFold(4,shuffle=True,random_state=404).split(query,groups=query)]
                    source = full_sys
                for a,b in splits:
                    train = np.flatnonzero(np.isin(index,a))
                    assert not np.isin(index[train],b).any()
                    w = fit(source,a,train,y,true_count,mode)
                    post[b] = softmax(lp[b]+x[b]@w,axis=1)
            np.testing.assert_allclose(post.sum(1),1.)
            prefix = assign(orders,(post.cumsum(1)<.5).sum(1),gate)
            marginal = (post[index]*risk).sum(1)>=.5
            for method,pred in [('median_prefix',prefix),('marginal',marginal)]:
                record(f'{variant}_{mode}_{method}',pred)
    path = ROOT/'reports/nov23_matched_learning_curve/cardinality_marginals.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
