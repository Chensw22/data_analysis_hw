"""Compare all eight combinations of route, cabin, and booking attributes on the same conflicts."""
import json
from itertools import combinations_with_replacement
from math import factorial

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from threadpoolctl import threadpool_limits

from conflict_pair_ordering_n26 import CACHE,REPORT as RANK_REPORT,features,training_pairs
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,OUT,data
from kaggle_workflow import sha256
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BLOCKS = ['destination_and_vip','cabin','companions']
REPORT = ROOT/'reports/nov23_matched_learning_curve/conflict_field_blocks.json'


def designs(raw,g):
    base,_ = features(raw,g)
    route = pd.get_dummies(raw[['Destination','VIP']].astype('string').fillna('?'),dtype=float)
    split = raw.Cabin.str.split('/',expand=True)
    location = pd.get_dummies(split[[0,2]].astype('string').fillna('?'),dtype=float)
    number = pd.to_numeric(split[1]).to_numpy(float)
    cabin = np.column_stack([location,np.nan_to_num(number,nan=-1)])
    counts = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    # Keep other members' spending/age information separate from the route and cabin blocks.
    total = pd.DataFrame(base).groupby(g).transform('sum').to_numpy()
    others = np.divide(total-base,(counts-1)[:,None],out=np.zeros_like(total),where=(counts>1)[:,None])
    companions = np.column_stack([counts,raw.PassengerId.str.split('_').str[1].astype(int),others])
    blocks = [route,cabin,companions]
    return [np.column_stack([base,*[block for j,block in enumerate(blocks) if mask&(1<<j)]])
            for mask in range(8)]


def shapley(values):
    result = {}
    for j,name in enumerate(BLOCKS):
        increments = []
        for mask in range(8):
            if mask&(1<<j):
                continue
            k = mask.bit_count()
            weight = factorial(k)*factorial(2-k)/factorial(3)
            increments.append(weight*(values[mask|(1<<j)]-values[mask]))
        result[name] = float(sum(increments))
    np.testing.assert_allclose(sum(result.values()),values[-1]-values[0],atol=1e-9)
    return result


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    cache = CACHE/'held.npz'
    old = json.loads(RANK_REPORT.read_text())
    assert sha256(cache)==old['cached_predictions_sha256']
    with np.load(cache) as z:
        for key,expected in [('passenger',raw.PassengerId.to_numpy(str)),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[key],expected)
        pairs = {k:z[k+'_pairs'] for k in ('spending','registration')}
        previous = {k:z[k+'_differences'][:,0] for k in pairs}
        prev_oof = z['scores'][:,0]
    a,b,d = relations(raw,g,False)
    matrices = designs(raw,g)
    diff = {n:np.full((len(e),8),np.nan) for n,e in pairs.items()}
    oof = np.full((len(y),8),np.nan)
    for k,h in combinations_with_replacement(np.unique(f),2):
        t = np.flatnonzero((f!=k)&(f!=h))
        selected = {n:(np.sort(f[e],axis=1)==[k,h]).all(1) for n,e in pairs.items()}
        q = np.flatnonzero(f==k) if k==h else np.unique(np.concatenate([e[selected[n]].ravel() for n,e in pairs.items()]))
        if not len(q):
            continue
        assert not np.isin(g[t],g[q]).any()
        real,_ = training_pairs(a,b,d,y,g,(f!=k)&(f!=h))
        batch = np.zeros((len(q),8))
        # Baseline votes have already been calculated under the identical model and split.
        for mask in range(1,8):
            x = matrices[mask]
            train = Pool(x[t],label=y[t].astype(int),pairs=real)
            model = CatBoostRanker(loss_function='PairLogit',iterations=115,depth=4,learning_rate=.07,
                   l2_leaf_reg=8,verbose=False,thread_count=2,allow_writing_files=False,random_seed=3505).fit(train)
            batch[:,mask] = model.predict(x[q])
        if k==h:
            oof[q] = batch
        for name,e in pairs.items():
            loc = selected[name]
            u,v = np.searchsorted(q,e[loc]).T
            diff[name][loc] = batch[u]-batch[v]
    oof[:,0] = prev_oof
    report = dict(blocks=BLOCKS,ordering={},deployment={})
    for name,z in diff.items():
        z[:,0] = previous[name]
        assert np.isfinite(z).all()
        right = (z>0)+.5*(z==0)
        r = right.mean(0)
        report['ordering'][name] = dict(pairs=len(z),correct_equivalents_per_mask=right.sum(0).tolist(),
                                        concordance_per_mask=r.tolist(),shapley_pp=shapley(100*r))
    assert np.isfinite(oof).all()
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        strong = logit(z['p'].clip(.003,.997))
    votes = np.zeros_like(oof,dtype=bool)
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        for u,v in GroupKFold(4,shuffle=True,random_state=169).split(rows,groups=g[rows]):
            t,q = rows[u],rows[v]
            assert not np.isin(g[t],g[q]).any()
            for j in range(8):
                x = np.column_stack([strong,oof[:,j]])
                votes[q,j] = LogisticRegression(C=.5,max_iter=350).fit(x[t],y[t]).predict(x[q])
    for name,mask in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
        r = [summary(pred!=cur['parent'],y!=cur['parent'],mask,f) for pred in votes.T]
        report['deployment'][name] = dict(net_per_mask=[v['net'] for v in r],
                   versus_tabnet=r,shapley_net_people=shapley([v['net'] for v in r]))
    report['interpretation'] = ('Each field contribution averages its change across every insertion order in the '
        'eight refitted ranking models. Ordering uses known opposed outcomes for diagnostics. Deployment counts '
        'instead use calibrated passenger decisions on complete held-out bookings. Mask bit order is given by blocks.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['ordering'],flush=True)
    print({k:v['net_per_mask'] for k,v in report['deployment'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
