"""Count nearby independent training bookings and rebalance training toward rare passenger profiles."""
import json

import numpy as np
from catboost import CatBoostClassifier
from scipy.spatial.distance import cdist

from broad_booking_policy_replay_n23 import interval
from broad_local_profile_fit_n23 import coordinates
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_metric_adjudication_n23 import load_scores


def booking_distances(x, y, groups, train):
    donors = train[np.argsort(groups[train], kind='stable')]
    keys, starts, counts = np.unique(groups[donors], return_index=True, return_counts=True)
    order_rate = np.add.reduceat(y[donors].astype(float), starts) / counts
    def batches(rows):
        for start in range(0, len(rows), 128):
            batch = rows[start:start+128]
            distance = np.minimum.reduceat(cdist(x[batch], x[donors], 'sqeuclidean'), starts, axis=1)
            distance[groups[batch, None] == keys] = np.inf
            kth = np.partition(distance, 19, axis=1)[:, 19]
            yield batch, distance, kth
    # Each training order excludes all its own passengers when determining the radius.
    radius = float(np.median(np.concatenate([v for _, _, v in batches(train)])))
    count = np.empty(len(y), int)
    rate = np.empty(len(y))
    for rows, distance, kth in batches(np.arange(len(y))):
        count[rows] = (distance <= radius).sum(1)
        neighbors = distance <= kth[:, None]
        assert (neighbors.sum(1) >= 20).all()
        rate[rows] = (neighbors*order_rate).sum(1) / neighbors.sum(1)
    return count, rate, radius


def weighted_prediction(x, y, train, query, weights):
    w = weights[train]
    w = np.clip(w, *np.quantile(w, [.05, .95]))
    model = CatBoostClassifier(iterations=350, depth=6, learning_rate=.065, verbose=False,
                               random_seed=1923, thread_count=3, allow_writing_files=False)
    model.fit(x.iloc[train], y[train].astype(int), cat_features=CATEGORICAL, sample_weight=w/w.mean())
    return model.predict_proba(x.iloc[query])[:, 1]


def run():
    raw, y, groups, fold = data()
    x = compact_features(raw)
    all_scores = {k: load_scores(raw, y, fold, OUT / f'{k}.npz', ['p'])[:, 0] for k in (*FAMILIES, 'mitra96')}
    cols = np.column_stack(list(all_scores.values()))
    ref = all_scores['mitra96'] >= .5
    unanimous = ((cols >= .5) == ref[:, None]).all(1)
    common = unanimous & (ref != y)
    query_counts, query_rates = np.empty(len(y), int), np.empty(len(y))
    probabilities = {k: np.empty(len(y)) for k in ['density', 'exception']}
    radii = []
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        assert not np.isin(groups[t], groups[q]).any()
        count, rate, radius = booking_distances(coordinates(x, t)[0], y, groups, t)
        query_counts[q], query_rates[q] = count[q], rate[q]
        probabilities['density'][q] = weighted_prediction(x, y, t, q, 1/np.sqrt(1+count))
        local_agreement = np.where(y[t], rate[t], 1-rate[t])
        weights = np.ones(len(y))
        weights[t] = 1/np.sqrt(.10+local_agreement)
        probabilities['exception'][q] = weighted_prediction(x, y, t, q, weights)
        radii.append(radius)
        print('completed outer fold', f, flush=True)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as bank:
        dev = np.sort(bank['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    bands = (20*np.where(ref[:, None], cols, 1-cols).mean(1)).astype(int) + 21*ref
    expected = np.zeros(len(y))
    for b in np.unique(bands[unanimous]):
        members = unanimous & (bands == b)
        expected[members] = common[members].mean()
    def group(mask):
        return dict(rows=int(mask.sum()), unanimous=int((mask & unanimous).sum()), errors=int((mask & common).sum()),
                    expected_at_matched_confidence=float(expected[mask].sum()), historical=int((mask & hard).sum()))
    agreement = np.where(ref, query_rates, 1-query_rates)
    report = dict(radius_squared=radii,
        coverage={k: group(m) for k, m in [('0_to_4',query_counts <= 4),('5_to_19',(query_counts >= 5) & (query_counts < 20)),
            ('20_to_99',(query_counts >= 20) & (query_counts < 100)),('100_plus',query_counts >= 100)]},
        nearby_agreement={k: group(m) for k, m in [('opposite',agreement < .5), ('same_below_80',(agreement >= .5) & (agreement < .8)),
                                                    ('same_at_least_80',agreement >= .8)]}, models={})
    def compare(pred, baseline):
        fixed = (pred == y) & (baseline != y)
        harmed = (pred != y) & (baseline == y)
        delta = fixed.astype(int)-harmed
        return dict(accuracy=float((pred == y).mean()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                    common_fixed=int((fixed & common).sum()), historical_fixed=int((fixed & hard).sum()),
                    sparse_fixed=int((fixed & (query_counts <= 4)).sum()), sparse_harmed=int((harmed & (query_counts <= 4)).sum()),
                    additional_net=int(delta.sum()-delta[dev].sum()), fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
                    interval_pp=interval(delta,groups))
    plain = all_scores['cat']
    mitra = all_scores['mitra96']
    for name, p in probabilities.items():
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        blend = p+mitra >= 1
        report['models'][name] = dict(standalone_vs_cat=compare(p >= .5, plain >= .5),
                blend_vs_plain_blend=compare(blend, plain+mitra >= 1), blend_vs_mitra=compare(blend, ref))
    report['models']['sparse_density_only'] = compare(np.where(query_counts <= 4, probabilities['density']+mitra >= 1, ref), ref)
    dest = ROOT/'reports/nov23_matched_learning_curve/support_density.json'
    dest.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
