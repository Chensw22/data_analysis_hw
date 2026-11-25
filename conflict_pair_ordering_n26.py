"""Compare pointwise and pairwise supervision on the fixed disjoint shared-error pairs."""
import hashlib
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from cross_order_conflict_cover_n26 import relations
from deployment_booking_mix_n23 import passenger_vectors
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import OUT,ROOT,data
from price_implied_deck_n26 import FEES

METHODS = ['spending_rank','full_point','full_rank','shuffled_rank']
REPORT = ROOT/'reports/nov23_matched_learning_curve/conflict_pair_ordering.json'
CACHE = ROOT/'artifacts/nov26_conflict_pair_ordering'


def features(raw,g):
    fees = raw[FEES].to_numpy(float)
    small = np.column_stack([pd.get_dummies(raw[['HomePlanet','CryoSleep']].astype(str),dtype=float),
                             raw.Age.fillna(-1),np.nan_to_num(np.log1p(fees),nan=-1)])
    individual = passenger_vectors(raw)
    members = pd.DataFrame(individual).groupby(g)
    group = pd.Series(g)
    full = np.column_stack([individual,members.transform('mean'),members.transform('max')-members.transform('min'),
                            group.map(group.value_counts()), raw.PassengerId.str.split('_').str[1].astype(int)])
    return small,full


def comparisons(a,b,y):
    # Select high-coverage disjoint pairs with opposite outcomes. This is used
    # only for reporting conditional ordering, never for a deployable action.
    left,right = np.flatnonzero(~y),np.flatnonzero(y)
    i = np.searchsorted(left,np.where(y[a],b,a))
    j = np.searchsorted(right,np.where(y[a],a,b))
    graph = csr_matrix((np.ones(len(i),bool),(i,j)),shape=(len(left),len(right)))
    match = maximum_bipartite_matching(graph,perm_type='column')
    used = match>=0
    return np.column_stack([right[match[used]],left[used]])


def training_pairs(a,b,dist,y,g,eligible):
    # Nearby pairs receive priority; cap each passenger at six partners so dense
    # neighborhoods cannot dominate via thousands of copies of the same endpoint.
    active = np.flatnonzero(eligible[a]&eligible[b]&(y[a]!=y[b]))
    order = active[np.lexsort((b[active],a[active],dist[active]))]
    degree = np.zeros(len(y),int)
    pos,neg = [],[]
    for e in order:
        u,v = a[e],b[e]
        if degree[u]<6 and degree[v]<6:
            degree[u]+=1
            degree[v]+=1
            pos.append(u if y[u] else v)
            neg.append(v if y[u] else u)
    pos,neg = np.array(pos),np.array(neg)
    assert y[pos].all() and not y[neg].any() and (g[pos]!=g[neg]).all()
    rng = np.random.default_rng(531)
    sham = rng.permutation(neg)
    for idx in np.flatnonzero(g[pos]==g[sham]):
        partner = np.flatnonzero((g[pos]!=g[sham[idx]])&(g[sham]!=g[pos[idx]]))
        j = rng.choice(partner)
        sham[idx],sham[j] = sham[j],sham[idx]
    assert (g[pos]!=g[sham]).all()
    np.testing.assert_array_equal(np.sort(neg),np.sort(sham))
    rows = np.flatnonzero(eligible)
    return [np.searchsorted(rows,np.column_stack([pos,end])) for end in (neg,sham)]


def fit_predict(x,y,g,a,b,d,eligible,q):
    t = np.flatnonzero(eligible)
    assert not np.isin(g[t],g[q]).any()
    observed,shuffled = training_pairs(a,b,d,y,g,eligible)
    params = dict(iterations=115,depth=4,learning_rate=.07,l2_leaf_reg=8,verbose=False,
                  thread_count=2,allow_writing_files=False,random_seed=3505)
    result = []
    for name in METHODS:
        design = x[0] if name=='spending_rank' else x[1]
        if name=='full_point':
            model = CatBoostClassifier(**params).fit(design[t],y[t].astype(int))
            result.append(model.predict(design[q],prediction_type='RawFormulaVal'))
        else:
            edges = shuffled if name=='shuffled_rank' else observed
            pool = Pool(design[t],label=y[t].astype(int),pairs=edges)
            model = CatBoostRanker(loss_function='PairLogit',**params).fit(pool)
            result.append(model.predict(design[q]))
    return np.column_stack(result)


def sides(raw,y,g,f,cur):
    a,b,d = relations(raw,g,False)
    pairs = {}
    for name,full in [('spending',False),('registration',True)]:
        l,r,_ = relations(raw,g,True) if full else (a,b,d)
        ok = cur['common'][l]&cur['common'][r]&(y[l]!=y[r])&(cur['parent'][l]==cur['parent'][r])
        pairs[name] = comparisons(l[ok],r[ok],y)
        assert len(np.unique(pairs[name]))==pairs[name].size
    assert [len(v) for v in pairs.values()]==[896,330]
    diff = {name:np.full((len(edges),len(METHODS)),np.nan) for name,edges in pairs.items()}
    oof = np.full((len(y),len(METHODS)),np.nan)
    x = features(raw,g)
    for k,h in combinations_with_replacement(np.unique(f),2):
        selected = {}
        for name,edges in pairs.items():
            lo,hi = np.sort(f[edges],axis=1).T
            selected[name] = (lo==k)&(hi==h)
        if k==h:
            q = np.flatnonzero(f==k)
        else:
            q = np.unique(np.concatenate([edges[selected[name]].ravel() for name,edges in pairs.items()]))
        if len(q)==0:
            continue
        pred = fit_predict(x,y,g,a,b,d,(f!=k)&(f!=h),q)
        if k==h:
            oof[q] = pred
        for name,edges in pairs.items():
            subset = selected[name]
            i,j = np.searchsorted(q,edges[subset]).T
            diff[name][subset] = pred[i]-pred[j]
    assert np.isfinite(oof).all() and all(np.isfinite(v).all() for v in diff.values())
    return pairs,diff,oof


def calibrate(scores,strong,y,g,fold):
    names = ['confidence_only',*METHODS]
    decisions = {k:np.zeros(len(y),bool) for k in names}
    for k in np.unique(fold):
        rows = np.flatnonzero(fold==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=169).split(rows,groups=g[rows]):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            for j,name in enumerate(names):
                xs = strong[:,None] if j==0 else np.column_stack([strong,scores[:,j-1]])
                model = LogisticRegression(C=.5,max_iter=350).fit(xs[t],y[t])
                decisions[name][q] = model.predict(xs[q])
    return decisions


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    pairs,diff,oof = sides(raw,y,g,f,cur)
    report = {'ordering':{},'deployment':{}}
    for name,v in diff.items():
        report['ordering'][name] = {}
        for j,method in enumerate(METHODS):
            z = v[:,j]
            report['ordering'][name][method] = dict(pairs=len(z),correct=int((z>0).sum()),
                                                     tied=int((z==0).sum()),concordance=float(((z>0)+.5*(z==0)).mean()))
    with np.load(OUT/'mitra96.npz') as file:
        for field,expected in [('passenger',raw.PassengerId.to_numpy(str)),('fold',f),('y',y)]:
            np.testing.assert_array_equal(file[field],expected)
        strong = logit(file['p'].clip(.003,.997))
    decisions = calibrate(oof,strong,y,g,f)
    parent = cur['parent']
    control = decisions['confidence_only']
    for scope,m in [('complete',cur['known']),('all_eleven_agree',cur['common']),('new_orders',cur['known']&cur['new'])]:
        report['deployment'][scope] = {}
        for name,v in decisions.items():
            challenge = v!=parent
            extra = challenge&(control==parent)
            report['deployment'][scope][name] = dict(versus_parent=summary(challenge,y!=parent,m,f),
                extra_challenge=summary(extra,y!=parent,m,f),versus_same_calibration=summary(v!=control,y!=control,m,f))
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE/'held.npz',passenger=raw.PassengerId.to_numpy(str),y=y,fold=f,scores=oof,
                        methods=METHODS,**{name+'_pairs':edges for name,edges in pairs.items()},
                        **{name+'_differences':v for name,v in diff.items()})
    report['cached_predictions_sha256'] = hashlib.sha256((CACHE/'held.npz').read_bytes()).hexdigest()
    report['scope'] = ('Ranking diagnostics use disjoint pairs defined with observed outcomes. Each comparison scores both '
       'passengers using one model trained outside both booking folds. Deployable predictions separately score each '
       'passenger in eight whole-booking folds. Calibration uses separate bookings inside each held fold.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['ordering'],flush=True)
    print({k:v['extra_challenge']['net'] for k,v in report['deployment']['all_eleven_agree'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
