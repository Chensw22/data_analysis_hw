"""Estimate local service spending from disjoint paying bookings and test it on unbilled solos."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT
from full_roster_two_view_reliability_n24 import design
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs


def borrow(train, query, lanes, positions, groups, residual, shifted):
    result = np.zeros((len(query), residual.shape[1]))
    available = np.zeros(len(query), bool)
    for lane in np.unique(lanes[query]):
        if lane == '?':
            continue
        bank = train[lanes[train] == lane]
        if len(bank) < 8:
            continue
        order = bank[np.argsort(positions[bank], kind='stable')]
        for j in np.flatnonzero(lanes[query] == lane):
            center = positions[query[j]]
            if shifted:
                index = np.searchsorted(positions[order], center)
                center = positions[order[(index + len(order)//2) % len(order)]]
            rank = np.argsort(np.abs(positions[order] - center), kind='stable')
            closest = order[rank]
            # Each paying booking supplies at most one neighbor.
            first = np.unique(groups[closest], return_index=True)[1]
            selected = closest[np.sort(first)[:12]]
            distance = np.abs(positions[selected] - center)
            weights = np.exp(-distance/max(2., np.median(distance)))
            result[j] = np.average(residual[selected], axis=0, weights=weights)
            available[j] = True
    return result, available


def spatial_proxy(frame, groups, registration, paid):
    cabin = frame.Cabin.str.split('/', expand=True)
    lanes = (cabin[0] + cabin[2]).fillna('?').to_numpy(str)
    positions = pd.to_numeric(cabin[1]).to_numpy(float)
    bills = np.log1p(frame[FEES].to_numpy(float))
    complete = paid & np.isfinite(bills).all(axis=1)
    split = np.empty(len(frame), int)
    for k, (_, q) in enumerate(GroupKFold(3, shuffle=True, random_state=1843).split(frame, groups=groups)):
        split[q] = k
    expected = np.empty_like(bills)
    real, displaced = np.zeros_like(bills), np.zeros_like(bills)
    supported = np.zeros(len(frame),bool)
    for k in range(3):
        ref = np.flatnonzero(complete & (split == (k+1)%3))
        donors = np.flatnonzero(complete & (split == (k+2)%3))
        q = np.flatnonzero(split == k)
        assert not np.isin(groups[q],groups[np.r_[ref,donors]]).any()
        residual = np.full_like(bills, np.nan)
        for j in range(len(FEES)):
            model = HistGradientBoostingRegressor(max_iter=75, max_leaf_nodes=12, min_samples_leaf=25,
                                                   l2_regularization=4., early_stopping=False)
            model.fit(registration[ref], bills[ref,j])
            expected[q,j] = model.predict(registration[q])
            residual[donors,j] = bills[donors,j] - model.predict(registration[donors])
        real[q], supported[q] = borrow(donors,q,lanes,positions,groups,residual,False)
        displaced[q], other = borrow(donors,q,lanes,positions,groups,residual,True)
        np.testing.assert_array_equal(supported[q],other)
    held = complete & supported
    skill = {}
    for name, increment in [('baseline',0.), ('corridor',real), ('displaced',displaced)]:
        error = (bills - expected - increment)[held]
        skill[name] = dict(n=int(held.sum()), mse=float(np.mean(error**2)),
                           mse_by_service=dict(zip(FEES, (error**2).mean(0).tolist())))
    return real, displaced, supported, skill


def run():
    raw, frame, y, g, fold, first, corrections, _, ordinary, _ = inputs()
    n = len(y)
    parent = first ^ corrections
    wrong = parent[:n] != y
    _,_,known,fresh = cohort(raw,y,fold)
    unanimous = ((ordinary >= 0.5) == parent[:,None]).all(1)
    groups = frame.PassengerId.str.split('_').str[0].to_numpy()
    alone = ~pd.Series(groups).duplicated(keep=False).to_numpy()
    paid = frame[FEES].gt(0).any(axis=1).to_numpy()
    unpaid = ~paid
    eligible = alone & unpaid
    masks = dict(solo_unpaid=known & eligible[:n], fresh=known & fresh & eligible[:n],
                 shared=known & eligible[:n] & unanimous[:n])
    assert (masks['shared'] & wrong).sum() == 196
    registration, spend = design(frame,groups)
    real, displaced, supported, proxy = spatial_proxy(frame,groups,registration,paid)
    base = np.column_stack([registration, spend])
    variants = [base,np.column_stack([base,real,supported]),np.column_stack([base,displaced,supported])]
    contiguous = np.empty(n,int)
    for k,chunk in enumerate(np.array_split(np.unique(g),8)):
        contiguous[np.isin(g,chunk)] = k
    report = dict(payer_blind_bill_prediction=proxy,known_target=int(masks['solo_unpaid'].sum()),
                  shared_errors_with_local_donors=int((masks['shared'] & wrong & supported[:n]).sum()))
    archived = {}
    odds = lambda p: logit(np.clip(p, .002, .998))
    previous = np.where(parent,1.,-1.) * np.abs(odds(np.median(ordinary[:,1:4],axis=1)))
    for tag, partition in [('random',fold),('ordered',contiguous)]:
        scores = np.zeros((len(frame),3))
        for k in range(9):
            train = np.flatnonzero((partition != k) & unpaid[:n])
            q = np.flatnonzero(partition == k) if k < 8 else np.arange(n,len(frame))
            assert not np.isin(groups[train],groups[q]).any()
            for j,x in enumerate(variants):
                model = HistGradientBoostingClassifier(max_iter=110, max_leaf_nodes=12, min_samples_leaf=28,
                                                        l2_regularization=4., early_stopping=False, random_state=112)
                model.fit(x[train], y[train])
                scores[q,j] = model.predict_proba(x[q])[:,1]
        votes = scores >= .5
        reviews = (votes != parent[:,None]) & eligible[:,None] & supported[:,None]
        increment = ((previous + odds(scores[:,1]) - odds(scores[:,0]) >= 0) != parent) & reviews[:,1]
        rules = dict(real=reviews[:,1], displaced=reviews[:,2],
                     only_real=reviews[:,1] & ~reviews[:,0] & ~reviews[:,2],
                     supported_increment=increment & ~reviews[:,2])
        archived[tag] = scores
        report[tag] = dict(accuracy={key:float(np.mean(votes[:n,j][masks['solo_unpaid']]==y[masks['solo_unpaid']]))
                      for j,key in enumerate(('base','real','displaced'))},rules={})
        for key, flag in rules.items():
            d = flag[:n]*np.where(wrong,1,-1)
            report[tag]['rules'][key] = dict(scopes={k:count(flag[:n],v,wrong) for k,v in masks.items()},
                 fold_net=[int(d[known & (partition==j)].sum()) for j in range(8)],test=int(flag[n:].sum()))
            archived[tag+'_'+key] = flag
    np.testing.assert_allclose(archived['random'][n:],archived['ordered'][n:])
    out=ART/'nov27_solo_corridor_receipts'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=fold,**archived)
    (ROOT/'reports/nov23_matched_learning_curve/solo_corridor_receipts.json').write_text(json.dumps(report,indent=2)+'\n')
    for tag in ('random','ordered'):
        print(tag,report[tag]['accuracy'],report[tag]['rules']['supported_increment'],flush=True)
    print(proxy,report['shared_errors_with_local_donors'])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
