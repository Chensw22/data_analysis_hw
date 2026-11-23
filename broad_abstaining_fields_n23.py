"""Fit four field-specific experts that learn when to decide or defer to another field."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT/'reports/nov23_abstaining_fields'
BLOCKS = [('CryoSleep','Age','HomePlanet'),('Destination','VIP','BookingSize'),
          ('Deck','CabinNumber','Side'),tuple(BILLS)]
FORWARD = (0,1,2,3)
REVERSE = FORWARD[::-1]


def matrices(frame, t, q):
    train, query = [], []
    for fields in BLOCKS:
        columns = {}
        for key in fields:
            v = frame[key]
            if pd.api.types.is_numeric_dtype(v):
                x = v.to_numpy(float)
                selected = x[t]
                seen = selected[np.isfinite(selected) & ((selected>0) if key in BILLS else True)]
                edges = np.unique(np.quantile(seen, np.arange(1,8)/8)) if len(seen) else []
                state = np.searchsorted(edges,x).astype(str)
                state[~np.isfinite(x)] = 'm'
                if key in BILLS:
                    state[x==0] = 'z'
                columns[key] = state
            else:
                columns[key] = v.astype(str).to_numpy()
        def records(ids):
            for i in ids:
                record = {k:columns[k][i] for k in fields}
                for a,b in combinations(fields,2):
                    record[a+'|'+b] = record[a]+'|'+record[b]
                record['constant'] = 1.
                yield record
        encoder = DictVectorizer()
        train.append(encoder.fit_transform(records(t)))
        query.append(encoder.transform(records(q)))
    return train, query


def ordered_probability(accept, value, orders):
    # Average over the specified first-to-last orders; if all defer, assign .5.
    p = np.zeros(len(accept))
    da = np.zeros_like(accept)
    dv = np.zeros_like(accept)
    for order in orders:
        prefix = {}
        weight = np.ones(len(p))
        for j in order:
            prefix[j] = weight
            weight = weight*(1-accept[:,j])
        later = np.full(len(p), .5)
        for j in reversed(order):
            da[:,j] += prefix[j]*(value[:,j]-later)
            dv[:,j] += prefix[j]*accept[:,j]
            later = accept[:,j]*value[:,j]+(1-accept[:,j])*later
        p += later
    return p/len(orders), da/len(orders), dv/len(orders)


def fit_selective(train, query, target, orders):
    splits = np.cumsum([2*x.shape[1] for x in train])[:-1]
    def evaluate(coef, design):
        pairs = [x@w.reshape(-1,2) for x,w in zip(design,np.split(coef,splits))]
        accept = expit(np.column_stack([p[:,0] for p in pairs])-.6)
        value = expit(np.column_stack([p[:,1] for p in pairs]))
        prob, da, dv = ordered_probability(accept,value,orders)
        return prob, da*accept*(1-accept), dv*value*(1-value)
    def loss(coef):
        prob, da, dv = evaluate(coef,train)
        p = np.clip(prob,1e-12,1-1e-12)
        residual = (p-target)/(p*(1-p))
        grad = np.concatenate([(x.T @ np.column_stack([residual*da[:,i],residual*dv[:,i]])).ravel()
                               for i,x in enumerate(train)])
        nll = -np.where(target,np.log(p),np.log1p(-p)).sum()
        return nll + .5*(coef@coef), grad + coef
    start = np.zeros(sum(2*x.shape[1] for x in train))
    fitted = minimize(loss,start,method='L-BFGS-B',jac=True,options={'maxiter':1200,'ftol':1e-8})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return evaluate(fitted.x,query)[0]


def main():
    row,y,frame,g,_ = load()
    wrong,right,hard = cohorts(row,y)
    plans, history = {}, {}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as data:
            np.testing.assert_array_equal(data['row'], row)
            np.testing.assert_array_equal(data['y'], y)
            fold,ref,base = data['fold'],data['ref'],data['base']
        bank = {k:np.full(len(row),np.nan) for k in ('additive','personal_first','bill_first','averaged_order')}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            train,query = matrices(frame,t,q)
            lr = LogisticRegression(C=1, max_iter=500).fit(hstack(train),y[t])
            bank['additive'][q] = lr.predict_proba(hstack(query))[:,1]
            for key,orders in [('personal_first',[FORWARD]),('bill_first',[REVERSE]),
                               ('averaged_order',[FORWARD,REVERSE])]:
                bank[key][q] = fit_selective(train,query,y[t],orders)
        hits = {'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            hits[k] = (p>=.5)==y
            hits['quarter_'+k] = np.where(ref==(base>=.5),.75*base+.25*p>=.5,ref)==y
        plans[seed] = {k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        directory = ART/'nov23_abstaining_fields'
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**bank)
        print(seed,{k:m['fixed']-m['harmed'] for k,m in plans[seed].items()},flush=True)
    baseline = np.stack(history['reference'])
    stable = {}
    for k,parts in history.items():
        hit = np.stack(parts)
        stable[k] = dict(hard_stable=int((hard & hit.all(0)).sum()),hard_ever=int((hard & hit.any(0)).sum()),
                         common_stable=int((wrong & hit.all(0)).sum()),stable_harms=int((right & ~hit.any(0)).sum()),
                         hard_solo_stable=int((hard & hit.all(0) & (frame.BookingSize==1)).sum()),
                         gain95_pp=interval((hit.astype(float)-baseline).mean(0),g))
    write(REPORT/'results.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
