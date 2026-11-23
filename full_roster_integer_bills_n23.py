"""Test exact-price recurrence and cross-service common divisors on held-out bookings."""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from expanded_training_roster_n23 import fitting
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data

BILLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
REPORT = ROOT / 'reports/nov23_matched_learning_curve/integer_bills.json'


def divisor_features(amounts, index):
    fields = {}
    for left, right in combinations(range(len(BILLS)), 2):
        first, second = amounts[:, left], amounts[:, right]
        both = (first > 0) & (second > 0)
        divisor = np.gcd(first, second)
        denominator = np.maximum(1, divisor)
        pair = f'{left}{right}'
        fields['gcd' + pair] = np.where(both, np.log1p(divisor), -1.)
        fields['units' + pair] = np.where(both, np.log1p(first / denominator + second / denominator), -1.)
    return pd.DataFrame(fields, index=index)


def price_frequencies(amounts, groups, train, index):
    fields = {}
    for j, values in enumerate(amounts.T):
        counts = np.bincount(values[train], minlength=int(values.max()) + 1)
        totals = counts[values]
        # Each training feature excludes every occurrence contributed by its own
        # order, even when two passengers in that order have the same charge.
        source = pd.DataFrame({'group': groups[train], 'value': values[train]})
        totals[train] -= source.groupby(['group', 'value']).value.transform('size').to_numpy()
        assert (totals >= 0).all()
        fields[f'exact{j}'] = np.where(values > 0, np.log1p(totals), -1.)
    return pd.DataFrame(fields, index=index)


def changes(hit, previous, fold):
    fixed = hit & ~previous
    harmed = ~hit & previous
    return dict(fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                net_by_fold=np.bincount(fold, weights=fixed.astype(int) - harmed.astype(int)).astype(int).tolist())


def run():
    raw, y, groups, fold = data()
    old = {}
    for family in (*FAMILIES, 'mitra96'):
        with np.load(OUT / f'{family}.npz') as cache:
            for key, expected in [('passenger', raw.PassengerId), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(cache[key], expected)
            old[family] = cache['p']
    votes = np.column_stack(list(old.values())) >= .5
    hard = (votes != y[:, None]).all(1)
    correct = (votes == y[:, None]).all(1)
    observed = raw[BILLS].to_numpy(float)
    amounts = np.where(np.isfinite(observed), observed, 0).astype(int)
    assert (np.isnan(observed) | (observed == amounts)).all() and (amounts >= 0).all()
    fields = compact_features(raw)
    divisors = divisor_features(amounts, fields.index)
    with_divisors = pd.concat([fields, divisors], axis=1)
    predictions = {name: np.full(len(y), np.nan) for name in ('divisors', 'frequency', 'combined')}
    for f in np.unique(fold):
        train = np.flatnonzero(fold != f)
        query = np.flatnonzero(fold == f)
        assert not np.intersect1d(groups[train], groups[query]).size
        freq = price_frequencies(amounts, groups, train, fields.index)
        for name, x in [('divisors', with_divisors), ('combined', pd.concat([with_divisors, freq], axis=1)),
                        ('frequency', pd.concat([fields, freq], axis=1))]:
            predictions[name][query] = fitting(x, y[train], train, query)
    results = {}
    for name, p in {'raw_cat': old['cat'], **predictions}.items():
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        for key, score, ref in [('single', p, old['cat']),
                                ('half_mitra', .5 * (p + old['mitra96']), .5 * (old['cat'] + old['mitra96']))]:
            hit = (score >= .5) == y
            results.setdefault(name, {})[key] = dict(
                correct=int(hit.sum()), accuracy=float(hit.mean()),
                vs_matched_control=changes(hit, (ref >= .5) == y, fold),
                vs_mitra=changes(hit, (old['mitra96'] >= .5) == y, fold),
                common_repaired=int((hard & hit).sum()), common_harmed=int((correct & ~hit).sum()))
    coverage = {}
    multiple = (amounts > 0).sum(axis=1) >= 2
    common_divisor = divisors.filter(regex='^gcd').max(axis=1).to_numpy()
    for name, mask in [('two_positive', multiple), *[(f'gcd_at_least_{n}', common_divisor >= np.log1p(n))
                                                     for n in (2, 10, 100)]]:
        coverage[name] = dict(rows=int(mask.sum()), common_wrong=int((mask & hard).sum()),
                              common_correct=int((mask & correct).sum()))
    return dict(rows=len(y), bookings=len(np.unique(groups)), fold_count=len(np.unique(fold)), service_order=BILLS,
                shared_models=list(old), common_wrong=int(hard.sum()), common_correct=int(correct.sum()),
                coverage=coverage, results=results)


if __name__ == '__main__':
    result = run()
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, arms in result['results'].items():
        print(name, {arm: row['correct'] for arm, row in arms.items()})
