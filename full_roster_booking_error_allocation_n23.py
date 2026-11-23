"""Separate booking-wide errors from errors that a change of passenger rank can fix."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from broad_booking_canonical_n23 import edge_table
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from matched_consensus_errors_n23 import roster_fields


def clustering(groups, scores, wrong):
    size = np.bincount(groups)[groups]
    votes = scores >= .5
    eligible = (votes == votes[:, :1]).all(axis=1) & (size > 1)
    # Shuffle only among unanimous calls of similar direction, confidence and order size.
    frame = pd.DataFrame(dict(size=size, direction=votes[:, 0],
                              confidence=np.minimum((abs(scores.mean(1)-.5)*20).astype(int), 9)))
    strata = list(frame[eligible].groupby(list(frame)).groups.values())
    def pairs(flags):
        totals = np.bincount(groups[flags])
        return int((totals * (totals-1)).sum() // 2)
    rng = np.random.default_rng(SEED)
    null = []
    shuffled = np.zeros(len(groups), bool)
    for _ in range(1000):
        for ids in strata:
            shuffled[ids] = rng.permutation(wrong[ids])
        null.append(pairs(shuffled))
    counts = np.bincount(groups[wrong], minlength=groups.max()+1)
    return dict(solo_passengers=int((size == 1).sum()), solo_misses=int(wrong[size == 1].sum()),
                multi_passengers=int((size > 1).sum()), multi_misses=int(wrong[size > 1].sum()),
                multi_orders_with_misses=int(((counts > 0) & (np.bincount(groups) > 1)).sum()),
                multi_orders_with_multiple_misses=int((counts >= 2).sum()),
                multi_misses_in_these_orders=int(counts[counts >= 2].sum()),
                paired_misses=pairs(wrong), shuffled_mean=float(np.mean(null)),
                shuffled_interval=np.quantile(null, [.025, .975]).tolist(),
                one_sided_p=(1+sum(n >= pairs(wrong) for n in null))/1001)


def partner_rank(raw, groups, fold, y):
    ids = np.arange(len(y))
    fields, _ = roster_fields(ids, raw.PassengerId.str.split('_').str[0].astype(int).to_numpy())
    left, right, _ = edge_table(groups, ids)
    x = np.column_stack([fields[left]-fields[right], (fields[left]+fields[right])/2]).astype('float32')
    probability = np.zeros(len(left))
    assert np.array_equal(fold[left], fold[right])
    training_pairs = []
    for f in np.unique(fold):
        train = (fold[left] != f) & (y[left] != y[right])
        query = fold[left] == f
        train_groups = groups[left[train]]
        assert not np.isin(train_groups, groups[left[query]]).any()
        weights = 1 / np.bincount(train_groups)[train_groups]
        model = CatBoostClassifier(iterations=180, depth=4, learning_rate=.04, verbose=False,
                                   thread_count=3, random_seed=SEED, allow_writing_files=False)
        model.fit(x[train], y[left[train]].astype(int), sample_weight=weights/weights.mean())
        probability[query] = model.predict_proba(x[query])[:, 1]
        training_pairs.append(int(train.sum()))
    # Give each passenger half credit from each orientation of every comparison.
    wins = np.bincount(left, probability, minlength=len(y)) + np.bincount(right, 1-probability, minlength=len(y))
    return wins / np.maximum(2*np.bincount(left, minlength=len(y)), 1), training_pairs


def assign(order_rows, counts, score):
    pred = np.zeros(len(score), bool)
    for rows, count in zip(order_rows, counts, strict=True):
        rank = np.argsort(-score[rows], kind='stable')
        pred[rows[rank[:count]]] = True
    return pred


def run():
    raw, y, bookings, fold = data()
    _, groups = np.unique(bookings, return_inverse=True)
    columns = []
    for family in (*FAMILIES, 'mitra96'):
        with np.load(OUT / f'{family}.npz') as bank:
            for key, expected in [('passenger', raw.PassengerId), ('fold', fold), ('y', y)]:
                np.testing.assert_array_equal(bank[key], expected)
            columns.append(bank['p'])
    scores = np.column_stack(columns)
    wrong = ((scores >= .5) != y[:, None]).all(1)
    baseline = scores[:, -1] >= .5
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    rank, training_pairs = partner_rank(raw, groups, fold, y)
    ordered = np.split(np.argsort(groups, kind='stable'), np.cumsum(np.bincount(groups))[:-1])
    def totals(flags):
        return np.bincount(groups[flags], minlength=len(ordered))
    predicted, actual = totals(baseline), totals(y)
    oracle_errors = abs(actual-predicted).sum()
    np.testing.assert_array_equal(baseline, assign(ordered, predicted, scores[:, -1]))
    def summary(candidate):
        repairs = ~ (baseline == y) & (candidate == y)
        harms = (baseline == y) & (candidate != y)
        delta = repairs.astype(int)-harms
        return dict(accuracy=float((candidate == y).mean()), repaired=int(repairs.sum()),
                    newly_wrong=int(harms.sum()), shared_errors_repaired=int((repairs & wrong).sum()),
                    historic_errors_repaired=int((repairs & hard).sum()),
                    historic_newly_wrong=int((harms & hard).sum()),
                    fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)])
    reassigned = assign(ordered, predicted, rank)
    np.testing.assert_array_equal(totals(reassigned), predicted)
    fp, fn = baseline & ~y, ~baseline & y
    fp_total, fn_total = totals(fp), totals(fn)
    paired_shared = (np.minimum(totals(wrong & fp), fn_total) +
                     np.minimum(totals(wrong & fn), fp_total)).sum()
    result = dict(n=len(y), baseline_correct=int((baseline == y).sum()),
                  baseline_accuracy=float((baseline == y).mean()), shared_errors=int(wrong.sum()),
                  clustering=clustering(groups, scores, wrong), training_directed_pairs=training_pairs,
                  rank_model=summary(reassigned),
                  diagnostic_known_totals_mitra_rank=summary(assign(ordered, actual, scores[:, -1])),
                  diagnostic_best_fixed_total_errors=int(oracle_errors),
                  diagnostic_best_fixed_total_gain=int((baseline != y).sum()-oracle_errors),
                  diagnostic_max_shared_errors_repaired_without_harm_at_fixed_total=int(paired_shared))
    path = ROOT / 'reports/nov23_matched_learning_curve/full_roster_booking_allocation.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(path.read_text())


if __name__ == '__main__':
    run()
