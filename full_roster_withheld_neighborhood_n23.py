"""Contrast withholding nearby bookings with removing the same booking sizes and outcome counts."""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_local_profile_fit_n23 import coordinates
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import compact_features
from expanded_training_roster_n23 import fitting
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from full_roster_metric_adjudication_n23 import load_scores


def matched_removal(books, nearby, rng):
    control = []
    for _, ids in books.groupby(['size', 'positive']).groups.items():
        n = int(np.isin(ids, nearby).sum())
        if n:
            # Prefer controls outside the neighborhood. Rare strata can force overlap.
            outside = np.setdiff1d(ids, nearby)
            inside = np.intersect1d(ids, nearby)
            controls = np.r_[rng.permutation(outside), rng.permutation(inside)]
            control.extend(controls[:n])
    control = np.asarray(control)
    for keys in [nearby, control]:
        assert len(np.unique(keys)) == len(nearby)
    np.testing.assert_array_equal(books.loc[nearby].value_counts().sort_index().to_numpy(),
                                  books.loc[control].value_counts().sort_index().to_numpy())
    assert books.loc[nearby].value_counts().sort_index().index.equals(books.loc[control].value_counts().sort_index().index)
    return control


def run():
    raw, y, groups, fold = data()
    frame = compact_features(raw)
    cached = {k: load_scores(raw, y, fold, OUT/f'{k}.npz', ['p'])[:, 0] for k in (*FAMILIES, 'mitra96')}
    scores = np.column_stack(list(cached.values()))
    common = ((scores >= .5) != y[:, None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    held, random = np.full(len(y), np.nan), np.full(len(y), np.nan)
    coverage = []
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        assert not np.isin(groups[q], groups[t]).any()
        x, _ = coordinates(frame, t)
        partition = KMeans(n_clusters=8, n_init=4, random_state=SEED).fit(x[t])
        cluster = partition.predict(x)
        books = pd.DataFrame({'group': groups[t], 'y': y[t]}).groupby('group').y.agg(size='size', positive='sum')
        for k in np.unique(cluster[q]):
            rows = q[cluster[q] == k]
            nearby = np.unique(groups[t[cluster[t] == k]])
            other = matched_removal(books, nearby, np.random.default_rng(SEED+8*int(f)+int(k)))
            a, b = t[~np.isin(groups[t], nearby)], t[~np.isin(groups[t], other)]
            assert len(a) == len(b) and y[a].sum() == y[b].sum()
            held[rows] = fitting(frame, y[a], a, rows)
            random[rows] = fitting(frame, y[b], b, rows)
            coverage.append(dict(fold=int(f), cluster=int(k), query=len(rows), orders_removed=len(nearby),
                                 rows_removed=len(t)-len(a), mandatory_overlap=len(np.intersect1d(nearby, other))))
        print('completed outer fold', f, flush=True)
    assert np.isfinite(held).all() and np.isfinite(random).all()
    def compare(candidate, ref):
        fixed, harm = (candidate == y) & (ref != y), (candidate != y) & (ref == y)
        delta = fixed.astype(int)-harm
        return dict(accuracy=float((candidate == y).mean()), fixed=int(fixed.sum()), harmed=int(harm.sum()),
                    common_fixed=int((fixed & common).sum()), common_harmed=int((harm & common).sum()),
                    historical_fixed=int((fixed & hard).sum()), historical_harmed=int((harm & hard).sum()),
                    fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta, groups))
    cat, mitra = cached['cat'], cached['mitra96']
    result = dict(coverage=coverage, cat_accuracy=float(((cat >= .5) == y).mean()),
                  removed_near_vs_full=compare(held >= .5, cat >= .5),
                  removed_other_vs_full=compare(random >= .5, cat >= .5),
                  retained_near_vs_removed_near=compare(random >= .5, held >= .5),
                  mean_near_removed_with_mitra=compare(held+mitra >= 1, mitra >= .5))
    path = ROOT/'reports/nov23_matched_learning_curve/withheld_neighborhood.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print({k: v for k, v in result.items() if k != 'coverage'})


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
