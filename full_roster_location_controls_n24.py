"""Compare real and permuted booking/cabin coordinates at the same selection budget."""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities


def spatial(y,g,fold,pools,xy,valid,half_score):
    direct = np.empty(len(y))
    area = np.empty(len(y))
    residual = np.zeros(len(y))
    for f in np.unique(fold):
        q = np.flatnonzero(fold==f)
        t = np.flatnonzero(fold!=f)
        assert not np.isin(g[t],g[q]).any()
        for key in np.unique(pools[q]):
            ask = q[pools[q]==key]
            donors = t[pools[t]==key]
            rate = (y[donors].sum()+2)/(len(donors)+4) if len(donors) else y[t].mean()
            direct[ask] = area[ask] = rate
            ask = ask[valid[ask]]
            donors = donors[valid[donors]]
            if not len(ask) or len(donors)<6:
                continue
            def neighbours(source,n):
                dist, idx = cKDTree(xy[source]).query(xy[ask],k=min(n,len(source)))
                return source[idx],np.exp(-dist/(dist[:,-1:]+.25))
            idx,w = neighbours(donors,24)
            direct[ask] = ((w*y[idx]).sum(1)+5*rate)/(w.sum(1)+5)
            # Half-roster scores for donors must exclude all target-fold orders.
            honest = donors[fold[donors]%2==f%2]
            if len(honest)<8:
                continue
            idx,w = neighbours(honest,32)
            grad = y[honest]-half_score[honest]
            hess = half_score[honest]*(1-half_score[honest])
            shift = grad.sum()/(hess.sum()+12)
            local_hess = half_score[idx]*(1-half_score[idx])
            residual[ask] = (w*(y[idx]-half_score[idx]-shift*local_hess)).sum(1)/((w*local_hess).sum(1)+5)
    return direct,area,residual


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    half_score = h[:,names.index('mitra96')]
    _,_,og,*_ = aligned()
    unseen = ~np.isin(g,og)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    assert hard.sum()==147
    cabin = raw.Cabin.str.extract(r'^([^/]+)/(\d+)/([PS])$')
    deck = cabin[0].fillna('?').to_numpy(str)
    lane = (cabin[0].fillna('?')+cabin[2].fillna('?')).to_numpy(str)
    pos = pd.to_numeric(cabin[1]).to_numpy()
    valid = np.isfinite(pos)
    xy = np.column_stack([g.astype(float)/5,pos])
    shuffled = xy.copy()
    rng = np.random.default_rng(6785)
    for k in np.unique(lane):
        idx = np.flatnonzero(lane==k)
        shuffled[idx] = xy[rng.permutation(idx)]
    paid = raw[BILLS].gt(0).any(axis=1)
    state = np.char.add(raw.CryoSleep.astype(str).to_numpy(str),np.where(paid,'P','Z'))
    # Compare to assigning exactly the same flip counts within each observable stratum.
    groups = pd.MultiIndex.from_arrays([fold,ref,np.digitize(np.abs(gate-.5),[.1,.2,.3,.4]),state])
    strata = groups.factorize()[0]
    sizes = np.bincount(strata)
    error_share = np.bincount(strata,weights=ref!=y)/sizes
    hard_share = np.bincount(strata,weights=hard)/sizes
    report = {}
    union = np.zeros(len(y),bool)
    for name,pools in [('lane',lane),('lane_activity',np.char.add(lane,state)),
                       ('mirrored_activity',np.char.add(deck,state))]:
        for kind,location in [('real',xy),('permuted',shuffled)]:
            direct,area,residual = spatial(y,g,fold,pools,location,valid,half_score)
            base_logit = logit(gate.clip(.001,.999))
            lift = logit(direct.clip(.01,.99))-logit(area.clip(.01,.99))
            if kind=='real':
                union |= (direct>=.5)==y
            for method,pred in [('direct',direct>=.5),('relative_rate',base_logit+lift>=0),
                                ('relative_residual',base_logit+residual>=0)]:
                changes = np.bincount(strata,weights=ref!=pred,minlength=len(sizes))
                report[f'{name}_{kind}_{method}'] = dict(full=stats(pred,ref,y,g,fold),
                    unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                    hard_fixed=int((hard & (pred==y)).sum()),
                    expected_hard_fixed=float(changes@hard_share),
                    expected_all_fixed=float(changes@error_share), active_strata=int((changes>0).sum()))
    report['real_direct_union_among_147'] = int((hard & union).sum())
    path = ROOT/'reports/nov23_matched_learning_curve/location_controls.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for tag in ('lane_real_direct','lane_permuted_direct','lane_real_relative_residual','lane_permuted_relative_residual'):
        r = report[tag]
        print(tag,r['full']['corrected'],r['full']['harmed'],r['hard_fixed'],r['expected_hard_fixed'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
