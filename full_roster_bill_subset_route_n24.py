"""Estimate whether registration dissent is useful from predictions on all 32 bill subsets."""
import json
from math import comb

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_solo_coverage_n24 import cached
from full_roster_two_view_reliability_n24 import design,tree

SUBSETS = np.arange(32)
MASK = (SUBSETS[:,None] & (1<<np.arange(5))) != 0


def predict(inputs,y,g,t,q):
    reg,bills = inputs
    assert not np.isin(g[t],g[q]).any()
    def build(ids,codes):
        visible = MASK[codes]
        return np.column_stack([reg[ids], np.where(visible,bills[ids],-2)])
    rng = np.random.default_rng(891)
    masks = np.column_stack([np.zeros(len(t),int),np.full(len(t),31),rng.integers(1,31,(len(t),5))])
    train = np.repeat(t,masks.shape[1])
    model = tree(19).fit(build(train,masks.ravel()),y[train])
    # -2 denotes hidden by the experiment; -1 denotes an originally missing bill.
    query = build(np.repeat(q,32),np.tile(SUBSETS,len(q)))
    return model.predict_proba(query)[:,1].reshape(len(q),32)


def effects(curve):
    sizes = MASK.sum(1)
    diffs = []
    for j in range(5):
        left = SUBSETS[~MASK[:,j]]
        d = curve[:,left+(1<<j)]-curve[:,left]
        w = np.array([1/(5*comb(4,n)) for n in sizes[left]])
        diffs.append(d@w)
    phi = np.column_stack(diffs)
    np.testing.assert_allclose(phi.sum(1),curve[:,-1]-curve[:,0],atol=1e-8)
    sign = (curve>=.5)==(curve[:,0,None]>=.5)
    summary = np.column_stack([sign[:,sizes==k].mean(1) for k in range(1,5)])
    return np.column_stack([phi, summary, curve.min(1),curve.max(1)])


def route(curve,bills,six,registration,base,p,y,g,fold):
    conflict = (registration>=.5)!=base
    raw = np.column_stack([base, p, registration, bills, six.mean(1), six.std(1)])
    end = np.column_stack([raw,curve[:,0],curve[:,-1]])
    designs = dict(raw=raw,ends=end,subsets=np.column_stack([end,effects(curve)]))
    out = {k:np.zeros(len(y),bool) for k in designs}
    for f in np.unique(fold):
        rows = np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=318).split(rows,groups=g[rows]):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            t,q = t[conflict[t]],q[conflict[q]]
            for name,x in designs.items():
                learner = make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=400))
                learner.fit(x[t],y[t]!=base[t])
                out[name][q] = learner.predict(x[q])
    return out


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('y',y),('fold',fold),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    bills = np.log1p(raw[BILLS]).fillna(-1).to_numpy()
    inputs = (design(raw,g)[0],bills)
    report = {}
    for tag,ix,f,base,p in [('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                            ('historical',rows,of,champ,old[:,0])]:
        curve = protocol(inputs,y,g,ix,f,tag,predict,SUBSETS,'bill_subset_route')
        reg = cached('two_view_reliability',tag,ix,f,g,y)[:,0]
        six = np.column_stack([cached(k,tag,ix,f,g,y) for k in ('response_channels','broad_error_transport_projection')])
        agree = ((six>=.5)==base[:,None]).all(1)
        result = {}
        votes = route(curve,bills[ix],six,reg,base,p,y[ix],g[ix],f)
        for k,flip in votes.items():
            for scope,mask in [('full',np.ones(len(ix),bool)),('consensus',agree)]:
                pred = base^(mask&flip)
                r = stats(pred,base,y[ix],g[ix],f)
                r['six_fixed'] = int((agree & (base!=y[ix]) & (pred==y[ix])).sum())
                r['hard_fixed'] = int((hard[ix] & (base!=y[ix]) & (pred==y[ix])).sum())
                if tag=='full':
                    new = ~np.isin(g[ix],og)
                    r['new_net'] = int((pred[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
                result[k+'_'+scope] = r
                print(tag,k,scope,r['corrected'],r['harmed'],r['six_fixed'],r['hard_fixed'],flush=True)
        report[tag] = result
    (ROOT/'reports/nov23_matched_learning_curve/bill_subset_route.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
