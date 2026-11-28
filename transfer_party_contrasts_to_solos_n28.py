"""Test whether the within-party ordering of passengers transfers to observationally similar solos."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from conditional_party_allocation_n27 import conditional_weights, passenger_matrix
from conditional_party_placebo_n27 import shuffled_bookings
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import ROOT, inputs


def recenter(margin, difference, strata):
    # Preserve the total expected number transported for each query population.
    out = margin.copy()
    for rows in pd.Series(strata).groupby(strata, sort=False).indices.values():
        z = margin[rows] + difference[rows]
        target = expit(margin[rows]).mean()
        offset = brentq(lambda b: expit(z+b).mean()-target, -np.max(z)-60., 60.-np.min(z))
        out[rows] = z + offset
        np.testing.assert_allclose(expit(out[rows]).mean(), target, atol=1e-9)
    return out


def run():
    raw, frame, y, groups, folds, first, changed, _, scores, _ = inputs()
    n = len(y)
    parent = first ^ changed
    wrong = parent[:n] != y
    _, _, known, fresh = cohort(raw,y,folds)
    solo = ~frame.PassengerId.str[:4].duplicated(keep=False).to_numpy()
    bill = frame[FEES].gt(0).any(axis=1).to_numpy()
    common = ((scores[:n]>=.5)==parent[:n,None]).all(axis=1)
    scopes = dict(all=known, fresh=known&fresh, common=known&common, solo_common_paid=known&common&solo[:n]&bill[:n],
                  solo_common_unpaid=known&common&solo[:n]&~bill[:n])
    assert (wrong & scopes['solo_common_paid']).sum()==329 and (wrong & scopes['solo_common_unpaid']).sum()==196
    anchor = np.where(parent,1.,-1.) * np.maximum(1e-6,abs(logit(np.median(scores[:,1:4],axis=1).clip(.002,.998))))
    # Separate levels of passenger demand from within-level contrasts.
    strata = (frame.HomePlanet.fillna('?')+'|'+frame.CryoSleep.astype('string').fillna('?')+'|'+np.where(bill,'pay','zero')).to_numpy(str)
    x = passenger_matrix(frame)
    orders = list(frame.groupby(frame.PassengerId.str[:4],sort=False).indices.values())
    ordered = np.zeros(n,int)
    for k,chunk in enumerate(np.array_split(np.unique(groups),8)):
        ordered[np.isin(groups,chunk)] = k
    report, archive = {}, {}
    with np.load(ART/'nov27_conditional_party_allocation/aligned.npz') as bank:
        np.testing.assert_array_equal(bank['passenger'], frame.PassengerId)
        np.testing.assert_array_equal(bank['fold'],folds)
        np.testing.assert_array_equal(bank['y'],y)
        for scheme, partition in [('random',folds),('ordered',ordered)]:
            ordinary, actual = bank[scheme+'_scores'].T
            flags = np.zeros((len(frame),3),bool)
            for k in range(9):
                train = np.r_[partition!=k,np.zeros(len(frame)-n,bool)]
                q = np.flatnonzero((np.r_[partition==k,np.full(len(frame)-n,k==8)]) & solo)
                bookings = [r for r in orders if len(r)>1 and train[r].all()]
                artificial = shuffled_bookings(bookings,strata,np.random.default_rng(741))
                w,_ = conditional_weights(x,y,artificial)
                placebo = x[q] @ w
                for j, difference in enumerate([actual[q]-ordinary[q],placebo-ordinary[q],actual[q]-placebo]):
                    estimate = recenter(anchor[q],difference,strata[q])
                    flags[q,j] = (estimate>=0)!=parent[q]
            actions = dict(actual=flags[:,0], placebo=flags[:,1], unique=flags[:,0]&~flags[:,1],
                           corroborated=flags[:,0]&flags[:,2]&~flags[:,1])
            report[scheme]={}
            for key, flag in actions.items():
                archive[scheme+'_'+key]=flag
                delta=flag[:n]*np.where(wrong,1,-1)
                report[scheme][key]=dict(scopes={s:count(flag[:n],v,wrong) for s,v in scopes.items()},
                    accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),
                    by_fold=[int(delta[known&(partition==i)].sum()) for i in range(8)],test=int(flag[n:].sum()))
    for key in actions:
        np.testing.assert_array_equal(archive['random_'+key][n:],archive['ordered_'+key][n:])
    dest=ART/'nov28_solo_party_contrasts'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=folds,**archive)
    (ROOT/'reports/nov23_matched_learning_curve/solo_party_contrasts.json').write_text(json.dumps(report,indent=2)+'\n')
    for tag in report:
        print(tag, report[tag]['corroborated'])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
