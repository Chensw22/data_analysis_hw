"""Estimate the expected count in nested portions of each order and recover individual risks."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data
from scale_teacher_exception_training_n26 import CACHE, features
from tabnet_attention_transfer_n26 import RESULT, collect
from target_mix_expert_routing_n26 import ticker

REPORT = RESULT.parent / 'booking_prefix_outcomes.json'
OUT = ROOT / 'artifacts/nov26_booking_prefix_outcomes'


def order_view(fields, opinions, group, reverse, outcome=None):
    probability = opinions.mean(axis=1)
    # Raw manifest order breaks score ties identically at training and evaluation.
    order = np.lexsort((np.arange(len(group)), -probability if reverse else probability, group))
    sorted_group = group[order]
    start = np.r_[0, 1 + np.flatnonzero(sorted_group[1:] != sorted_group[:-1])]
    size = np.diff(np.r_[start, len(group)])
    begin = np.repeat(start, size)
    end = np.repeat(start + size - 1, size)
    count = np.arange(len(group)) - begin + 1

    def prefix_sum(x):
        cumulative = np.vstack([np.zeros(x.shape[1]), np.cumsum(x[order], axis=0)])
        return cumulative[1:] - cumulative[begin]

    # The focal passenger, the prefix and the whole order get the same measured attributes.
    sums = prefix_sum(fields)
    design = np.column_stack([fields[order], sums / count[:, None],
                              sums[end] / (end - begin + 1)[:, None], count, end - begin + 1,
                              np.full(len(group), float(reverse))])
    view = dict(order=order, begin=begin, end=end, count=count, design=design)
    if outcome is not None:
        residual = outcome.astype(float) - probability
        view['prefix_target'] = prefix_sum(residual[:, None])[:, 0] / count
        view['person_target'] = residual[order]
    return view


def regressor():
    return HistGradientBoostingRegressor(max_iter=90, max_leaf_nodes=9, min_samples_leaf=45,
                                         learning_rate=.055, l2_regularization=8,
                                         early_stopping=False, random_state=149)


def fit_views(raw, groups, size, ids, opinions, truth=None):
    fields, _ = features(raw.iloc[ids], size[ids], opinions)
    return [order_view(fields, opinions, groups[ids], direction, truth) for direction in (False, True)]


def recover_counts(view, estimates):
    prefix = estimates * view['count']
    previous = np.r_[0., prefix[:-1]]
    previous[view['count'] == 1] = 0
    difference = prefix - previous
    # The increments telescope to the predicted total residual before clipping.
    np.testing.assert_allclose(np.add.reduceat(difference, np.unique(view['begin'])),
                               prefix[np.unique(view['end'])], atol=1e-9)
    return difference


def score(raw, group, size, ids, opinions, models):
    base = opinions.mean(axis=1)
    result = {key: np.zeros(len(ids)) for key in ('prefix', 'direct', 'count_only')}
    for view in fit_views(raw, group, size, ids, opinions):
        order = view['order']
        prefix = models['prefix_target'].predict(view['design'])
        result['prefix'][order] += recover_counts(view, prefix) / 2
        result['direct'][order] += models['person_target'].predict(view['design']) / 2
        result['count_only'][order] += prefix[view['end']] / 2
    return {key: (base + correction).clip(0, 1) for key, correction in result.items()}


def run():
    raw, y, g, fold = data()
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    scores = {}
    for k in np.unique(fold):
        t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        with np.load(CACHE / f'fold{k}.npz') as z:
            for key, expected in [('t', t), ('q', q), ('train_y', y[t])]:
                np.testing.assert_array_equal(z[key], expected)
            soft, panels, inner = z['soft'], z['outer_single'], z['inner_fold']
        assert not np.isin(g[t], g[q]).any()
        assert pd.DataFrame({'group': g[t], 'inner': inner}).groupby('group').inner.nunique().max() == 1
        train = fit_views(raw, g, size, t, soft, y[t])
        design = np.vstack([view['design'] for view in train])
        models = {target: regressor().fit(design, np.concatenate([v[target] for v in train]))
                  for target in ['prefix_target', 'person_target']}
        held = [score(raw, g, size, q, p, models) for p in panels]
        for key in held[0]:
            scores.setdefault(key, np.full(len(y), np.nan))[q] = np.mean([v[key] for v in held], axis=0)
        scores.setdefault('teachers', np.full(len(y), np.nan))[q] = panels.mean(axis=(0, 2))
        print('fold', k, flush=True)
    assert all(np.isfinite(v).all() for v in scores.values())
    # With no companion the two count-based estimates are identically the same.
    np.testing.assert_allclose(scores['prefix'][size == 1], scores['count_only'][size == 1], atol=1e-10)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'scores.npz', ids=raw.PassengerId.to_numpy(str), fold=fold, **scores)
    hard = {key: values >= .5 for key, values in scores.items()}
    report = {'accuracy': {key: float(np.mean(values == y)) for key, values in hard.items()}}
    for tag, items in collect()[0].items():
        ids, common, parent = (np.concatenate([batch[i] for batch in items]) for i in range(3))
        assert len(np.unique(ids)) == len(ids)
        common &= ~np.concatenate([batch[3]['actual_and_scores'] for batch in items])
        wrong = y[ids] != parent
        edits = {key: values[ids] != parent for key, values in hard.items()}
        edits['prefix_and_direct'] = edits['prefix'] & edits['direct']
        edits['unique_to_prefix'] = edits['prefix'] & ~edits['direct'] & ~edits['count_only']
        report[tag] = {}
        for key, action in edits.items():
            result = ticker(common, wrong, action)
            delta = common * action * np.where(wrong, 1, -1)
            result['net_by_fold'] = [int(delta[fold[ids] == k].sum()) for k in range(8)]
            result['multiple_member'] = ticker(common & (size[ids] > 1), wrong, action)
            report[tag][key] = result
        print(tag, report[tag]['unique_to_prefix'], flush=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
