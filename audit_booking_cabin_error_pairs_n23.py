"""Test whether long-standing shared errors co-occur within bookings or across nearby cabins."""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from audit_stable_rescue_placebo_n23 import booking_blocks
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts

REPORT = ROOT / 'reports/nov23_booking_cabin_error_pairs'


def pair_indices(frame, groups, eligible, direction):
    bookings = defaultdict(list)
    cabins = defaultdict(list)
    nearby = []
    for i in np.flatnonzero(eligible):
        bookings[groups[i]].append(i)
        num = frame.CabinNumber.iat[i]
        if not np.isfinite(num):
            continue
        region = (frame.Deck.iat[i], frame.Side.iat[i])
        for n in range(int(num)-3, int(num)+4):
            nearby.extend((j, i) for j in cabins[(*region, n)] if groups[j] != groups[i])
        cabins[(*region, int(num))].append(i)
    same = [p for members in bookings.values() for p in combinations(members, 2)]
    edges = {}
    for name, pairs in [('same_booking', same), ('near_cabin_3', nearby)]:
        a, b = np.array(pairs, dtype=int).T
        mask = direction[a] == direction[b]
        edges[name] = (a, b)
        edges[name+'_same_prediction'] = (a[mask], b[mask])
    return edges


def permutation_test(edges, hard, buckets, rng):
    batch = np.tile(hard, (50, 1))
    draws = {key: [] for key in edges}
    movable = [ids for ids in buckets if len(ids) > 1]
    for _ in range(40):
        for ids in movable:
            order = np.argsort(rng.random((len(batch), len(ids))), axis=1)
            batch[:, ids] = hard[ids[order]]
        for key, (a, b) in edges.items():
            draws[key].extend(np.sum(batch[:, a] & batch[:, b], axis=1))
    summary = {}
    for key, (a, b) in edges.items():
        hit = hard[a] & hard[b]
        observed = int(hit.sum())
        sims = np.asarray(draws[key])
        summary[key] = dict(pairs=len(a), both_hard=observed,
                            distinct_hard=len(np.unique(np.r_[a[hit], b[hit]])),
                            expected=float(sims.mean()), null95=np.quantile(sims, [.025, .975]).tolist(),
                            upper_tail=float((1 + (sims >= observed).sum())/(1 + len(sims))))
    return dict(swappable=sum(ids.size for ids in movable), edges=summary)


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    eligible = hard | right
    ids = np.flatnonzero(eligible)
    assert (hard.sum(), right.sum()) == (372, 3867)
    # Every historic opinion in these cohorts agrees on direction. Extract it
    # directly from the saved panel; use correctness only to score pairs.
    with np.load(ART/'nov23_release_vote_information/panel.npz') as panel:
        np.testing.assert_array_equal(row, panel['row'])
        np.testing.assert_array_equal(y, panel['y'])
        direction = panel['votes'][0, :, 0]
    np.testing.assert_array_equal(direction[ids] != y[ids], hard[ids])
    base = []
    for seed in SEEDS:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            np.testing.assert_array_equal(row, cache['row'])
            np.testing.assert_array_equal(y, cache['y'])
            base.append(cache['base'])
    mean = np.mean(base, axis=0)
    risk = np.where(direction, 1-mean, mean)
    cuts = np.quantile(risk[ids], np.arange(1, 6)/6)
    size = frame.BookingSize.to_numpy()
    spending = frame[BILLS].gt(0).any(axis=1).to_numpy()
    fields = np.column_stack([direction, np.searchsorted(cuts, risk), np.minimum(size, 3), spending])
    strata = pd.DataFrame(fields[ids]).groupby(list(range(fields.shape[1]))).indices
    passengers = [ids[idx] for idx in strata.values()]
    bookings = booking_blocks(groups, eligible, fields, risk)
    edges = pair_indices(frame, groups, eligible, direction)
    rng = np.random.default_rng(4011)
    tests = {name: permutation_test(edges, hard, blocks, rng)
             for name, blocks in [('passenger', passengers), ('intact_booking', bookings)]}
    membership = pd.DataFrame(dict(group=groups, direction=direction, hard=hard)).iloc[ids]
    count = membership.groupby('group').hard.sum()
    known_peers = membership.groupby(['group', 'direction']).hard.transform('sum') - membership.hard.astype(int)
    optimistic = {}
    for threshold in [1, 2]:
        flip = known_peers >= threshold
        optimistic[str(threshold)] = dict(flagged=int(flip.sum()), fixed=int((flip & membership.hard).sum()),
                                          harmed=int((flip & ~membership.hard).sum()))
    train, test = [pd.read_csv(ROOT/f'data/raw/{part}.csv', usecols=['PassengerId', 'Cabin'])
                   for part in ['train', 'test']]
    result = dict(passengers=len(y), hard=int(hard.sum()), comparison_right=int(right.sum()),
                  hard_bookings=int((count > 0).sum()), multiple_hard_bookings=int((count > 1).sum()),
                  hard_in_multiple=int(count[count > 1].sum()), singleton_hard=int((hard & (size == 1)).sum()),
                  test_rows=len(test), test_known_cabins=int(test.Cabin.notna().sum()),
                  shared_booking_rows=int(test.PassengerId.str.split('_').str[0].isin(
                      train.PassengerId.str.split('_').str[0]).sum()),
                  shared_known_cabin_rows=int(test.Cabin.dropna().isin(train.Cabin.dropna()).sum()),
                  seed=4011, draws_per_mode=2000, risk_sextile_cuts=cuts.tolist(),
                  tests=tests, diagnostic_with_known_peer_errors=optimistic)
    write(REPORT/'pairs.json', result)
    print(result)


if __name__ == '__main__':
    run()
