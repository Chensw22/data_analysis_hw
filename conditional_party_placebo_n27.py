"""Replace actual training companions with passengers in comparable slots of other training bookings."""
import json
from collections import defaultdict

import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from conditional_party_allocation_n27 import ROOT, conditional_weights, passenger_matrix, transfer
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs


def shuffled_bookings(bookings, strata, rng):
    # Matching position and sizes keeps each order's observable slot profile while scrambling identity.
    banks = defaultdict(list)
    for i, rows in enumerate(bookings):
        for slot, row in enumerate(rows):
            banks[(len(rows), slot, strata[row])].append((i, slot, row))
    out = [np.empty_like(rows) for rows in bookings]
    for group in banks.values():
        permuted = rng.permutation([row for _, _, row in group])
        for (i, slot, _), replacement in zip(group, permuted):
            out[i][slot] = replacement
    np.testing.assert_array_equal(np.sort(np.concatenate(out)), np.sort(np.concatenate(bookings)))
    return out


def run():
    raw, frame, y, g, fold, first, change, _, opinions, _ = inputs()
    parent = first ^ change
    n = len(y)
    wrong = parent[:n] != y
    _, _, known, fresh = cohort(raw, y, fold)
    shared = known & ((opinions[:n] >= 0.5) == parent[:n,None]).all(1)
    masks = dict(all=known, fresh=known&fresh, seven=shared)
    groups = frame.PassengerId.str.split('_').str[0]
    bookings = list(frame.groupby(groups,sort=False).indices.values())
    x = passenger_matrix(frame)
    paid = frame[FEES].gt(0).any(axis=1).to_numpy()
    strata = (frame.HomePlanet.fillna('?') + '|' + frame.CryoSleep.astype('string').fillna('?')
              + '|' + np.where(paid, 'paid', 'unpaid')).to_numpy()
    odds = np.where(parent, 1., -1.) * np.maximum(1e-6, abs(logit(np.median(opinions[:,1:4],axis=1).clip(.002,.998))))
    prior = expit(odds)
    ordered = np.zeros(n,int)
    for k, ids in enumerate(np.array_split(np.unique(g),8)):
        ordered[np.isin(g,ids)] = k
    report = {}
    with np.load(ART/'nov27_conditional_party_allocation/aligned.npz') as bank:
        np.testing.assert_array_equal(bank['passenger'],frame.PassengerId)
        np.testing.assert_array_equal(bank['y'],y)
        np.testing.assert_array_equal(bank['fold'],fold)
        for scheme, partition in [('random',fold),('ordered',ordered)]:
            original = bank[scheme+'_supported_increment']
            ordinary = bank[scheme+'_scores'][:,0]
            controls = []
            for seed in (741, 1823, 9059):
                scores = np.empty(len(frame))
                for k in range(9):
                    train_flag = np.r_[partition != k, np.zeros(len(frame)-n,bool)]
                    orders = [rows for rows in bookings if len(rows)>1 and train_flag[rows[0]]]
                    query = np.flatnonzero(partition==k) if k<8 else np.arange(n,len(frame))
                    assert all(train_flag[rows].all() for rows in orders)
                    fake = shuffled_bookings(orders, strata, np.random.default_rng(seed))
                    weights, _ = conditional_weights(x,y,fake)
                    scores[query] = x[query]@weights
                conditioned = transfer(prior,scores,bookings)
                incremental = transfer(prior, odds+scores-ordinary, bookings)
                controls.append(((conditioned>=.5)!=parent) & ((incremental>=.5)!=parent))
            votes = np.sum(controls,axis=0)
            rules = {f'shuffled_{i}':flag for i,flag in enumerate(controls)}
            rules.update(actual=original, actual_and_majority=original&(votes>=2),
                         actual_without_majority=original&(votes<2), shuffled_majority_only=(votes>=2)&~original)
            report[scheme]={}
            for name,flag in rules.items():
                report[scheme][name]=dict(scopes={k:count(flag[:n],v,wrong) for k,v in masks.items()},
                                       test=int(flag[n:].sum()))
            print(scheme,report[scheme],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/conditional_party_placebo.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
