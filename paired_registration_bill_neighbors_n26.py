"""Test whether finding the same historical passenger in two views improves shared decisions."""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, sources, ticker


def coordinates(frame, categorical, numerical, train):
    columns = []
    for name in categorical:
        values = frame[name].astype('string').fillna('missing')
        columns.append(pd.get_dummies(values).to_numpy(dtype=float) / np.sqrt(2))
    for name in numerical:
        x = frame[name].to_numpy(dtype=float)
        observed = np.sort(x[train][np.isfinite(x[train])])
        ranks = (np.searchsorted(observed, x, 'left') + np.searchsorted(observed, x, 'right')) / (2 * len(observed))
        columns.extend([np.where(np.isfinite(x), ranks, .5)[:, None], np.isnan(x)[:, None]])
    return np.column_stack(columns)


def kernel(x, train, query):
    squared = cdist(x[query], x[train], 'sqeuclidean')
    # Keep all tied neighbors at their true distances, including identically empty receipts.
    radius = np.partition(squared, 127, axis=1)[:, 127]
    return np.exp(-squared / np.maximum(radius[:, None], .0025))


def estimates(raw, y, groups, train, query, seed):
    assert not np.isin(groups[train], groups[query]).any()
    frame = raw.copy()
    parts = raw.Cabin.str.split('/', expand=True)
    frame['deck'], frame['berth'], frame['side'] = parts[0], pd.to_numeric(parts[1]), parts[2]
    frame['party_size'] = raw.groupby(groups).PassengerId.transform('size')
    registration = coordinates(frame, ['HomePlanet', 'Destination', 'VIP', 'deck', 'side', 'CryoSleep'],
                               ['Age', 'berth', 'party_size'], train)
    for field in FEES:
        frame[field + '_positive'] = frame[field].gt(0)
    bills = coordinates(frame, [f + '_positive' for f in FEES], FEES, train)
    # Destroy person-level matching but retain the outcome and sleep class of every training receipt.
    rng = np.random.default_rng(seed)
    permuted = np.arange(len(train))
    sleep = frame.CryoSleep.astype('string').fillna('missing').to_numpy()[train]
    for value in np.unique(sleep):
        for target in (False, True):
            idx = np.flatnonzero((sleep == value) & (y[train] == target))
            permuted[idx] = rng.permutation(idx)
    prior = y[train].mean()
    def posterior(weights):
        return (weights @ y[train] + 2 * prior) / (weights.sum(axis=1) + 2)
    chunks = []
    for q in np.array_split(query, (len(query) + 95) // 96):
        left, right = kernel(registration, train, q), kernel(bills, train, q)
        chunks.append(np.column_stack([posterior(left), posterior(right), posterior(left * right),
                                       posterior(left * right[:, permuted])]))
    return np.concatenate(chunks)


def run():
    raw, y, g, folds, views, _ = projection(include_full=True)
    nine, known, _, _ = champions()
    historical = sources()[-1]['historical']['ids']
    details = {}
    for tag in ('full', 'historical_to_new'):
        valid, parent = views[tag]
        assert known[valid].all()
        ids = np.flatnonzero(valid)
        splits = [(np.flatnonzero(folds != k), ids[folds[ids] == k]) for k in np.unique(folds)] if tag == 'full' else [(historical, ids)]
        out = np.full((len(y), 4), np.nan)
        for k, (train, query) in enumerate(splits):
            out[query] = estimates(raw, y, g, train, query, 9614 + k)
        assert np.isfinite(out[ids]).all()
        left, right, real, shuffled = out[ids].T
        base = parent[ids]
        both = base == nine[ids]
        flip = lambda p: (p >= .5) != base
        separate = flip(left) & flip(right)
        cross = flip(real)
        decisions = {'joint': cross, 'permuted': flip(shuffled), 'both_views': separate,
                     'joint_only_vs_permuted': cross & ~flip(shuffled),
                     'joint_confirmed_both_views': cross & separate,
                     'joint_confident': cross & (np.abs(real - .5) >= .25),
                     'joint_confident_confirmed': cross & (np.abs(real - .5) >= .25) & separate}
        details[tag] = {key: {name: ticker(mask, base != y[ids], rule)
                              for name, mask in [('all', np.ones(len(ids), bool)), ('common', both)]}
                        for key, rule in decisions.items()}
        print(tag, [(k, v['common']['fixed'], v['common']['harmed']) for k, v in details[tag].items()], flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/paired_registration_bill_neighbors.json'
    path.write_text(json.dumps(details, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
