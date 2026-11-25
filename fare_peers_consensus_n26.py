"""Test exact-fare evidence against predictions for entirely held-out bookings."""
import json

import numpy as np
from scipy.special import logit

from cross_deck_budget_multiples_n26 import fields
from full_roster_booking_validation_n23 import ROOT, data
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits


def peer_probabilities(cost, amount, pool, query):
    # One vote per independent order. Select its closest amount before reading its label.
    delta = cost[pool] - cost[query]
    near = np.flatnonzero(abs(delta) <= 480)
    near = near[np.argsort(abs(delta[near]), kind='stable')]
    shifted = 2 * (((amount[query] + 23) % 64) - amount[query] % 64)
    diffs = delta[near]
    weight = np.column_stack([np.exp(-.5 * (diffs / 42.)**2),
                              np.exp(-.5 * (diffs / 2.4)**2),
                              np.exp(-.5 * ((diffs - shifted) / 2.4)**2)])
    return near, weight


def predict(raw, y, groups, src, dst, t, q, target):
    total, count, eligible, origin, deck = fields(raw)
    cost = 2 * total + count
    donors, queries = src['ids'][t], dst['ids'][q]
    assert not np.isin(groups[donors], groups[queries]).any()
    train_y = y[donors] if target == 'outcome' else src['wrong'][t]
    out = np.full((len(q), 3), np.nan)
    for j, passenger in enumerate(queries):
        if not eligible[passenger]:
            continue
        mask = eligible[donors] & (origin[donors] == origin[passenger]) & (deck[donors] == deck[passenger])
        if target == 'error':
            mask &= src['anchor'][t] == dst['anchor'][q[j]]
        indexes = np.flatnonzero(mask)
        near, weight = peer_probabilities(cost, total, donors[indexes], passenger)
        if not len(near):
            continue
        _, distinct = np.unique(groups[donors[indexes[near]]], return_index=True)
        labels = train_y[indexes[near[distinct]]]
        weight = weight[distinct]
        background = (labels.sum() + 1) / (len(labels) + 2)
        wide = (weight[:, 0] @ labels + 5 * background) / (weight[:, 0].sum() + 5)
        fine = (weight[:, 1:].T @ labels + 5 * wide) / (weight[:, 1:].sum(axis=0) + 5)
        out[j] = [wide, *fine]
    return out


def run():
    raw, y, groups, _ = data()
    panels = sources()[-1]
    blocks = {}
    for key, src, dst, t, q in splits(panels, groups):
        if not (key.startswith('bookingmix') or key == 'historical_to_new'):
            continue
        protocol = 'bookingmix' if key.startswith('bookingmix') else 'historical_to_new'
        with np.load(REVIEW / f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'], dst['ids'][q])
            review = (z['identities_error'] > .5) & (z['identities_outcome'] > .5)
        store = blocks.setdefault(protocol, [])
        anchor = dst['anchor'][q]
        ref = anchor ^ review
        votes = {}
        for target in ['outcome', 'error']:
            p = predict(raw, y, groups, src, dst, t, q, target)
            valid = np.isfinite(p).all(1)
            for i, variant in enumerate(['fine', 'displaced'], start=1):
                # Hold original error odds fixed; add evidence only for the precise amount.
                signed_delta = logit(p[:, i]) - logit(p[:, 0])
                if target == 'outcome':
                    signed_delta *= np.where(anchor, -1, 1)
                native = dst['off'][q]
                votes[target + '_' + variant] = valid & ((native > 0) == review) & (((native + signed_delta) > 0) != review)
        votes['agree_fine'] = votes['error_fine'] & votes['outcome_fine']
        store.append((dst['ids'][q], ref != y[dst['ids'][q]], dst['common'][q], votes))
    report = {}
    for name, entries in blocks.items():
        ids, wrong, common = [np.concatenate([e[j] for e in entries]) for j in range(3)]
        assert len(np.unique(ids)) == len(ids)
        report[name] = {}
        for key in entries[0][3]:
            action = np.concatenate([e[3][key] for e in entries])
            report[name][key] = {n: ticker(m, wrong, action) for n, m in
                                 [('all', np.ones(len(ids), bool)), ('unanimous', common)]}
    (ROOT / 'reports/nov23_matched_learning_curve/fare_peers_consensus.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    run()
