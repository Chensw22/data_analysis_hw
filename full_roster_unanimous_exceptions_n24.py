"""Learn separate exception probabilities for unanimous positives and unanimous negatives."""
import json
from math import ceil

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats


def same_side(scores, side):
    return ((scores >= .5) == side).all(axis=1)


def profile(raw, eligible, exceptions):
    result = {}
    for name in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']:
        values = raw[name].astype('string').fillna('missing').to_numpy()
        result[name] = {k: dict(passengers=int((eligible & (values == k)).sum()),
                               exceptions=int((exceptions & eligible & (values == k)).sum()))
                        for k in np.unique(values[eligible])}
    return result


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    _, old_y, old_group, rows, _, _, champion, _, _, _, _, _, _ = aligned()
    unseen = ~np.isin(group, old_group)
    x = passenger_vectors(raw)
    common = ((full >= .5) != y[:, None]).all(1)
    modes = ['expert_mean', 'opinion_logistic', 'passenger_logistic', 'combined_tree']
    decisions = {name: reference.copy() for name in modes}
    summary = {}
    for side in [False, True]:
        actual = y != side
        qmask = same_side(full, side)
        hmask = same_side(half, side)
        assert np.all(reference[qmask] == side)
        oriented_h = 1-half if side else half
        oriented_q = 1-full if side else full
        h = logit(oriented_h.clip(.0001, .9999))
        z = logit(oriented_q.clip(.0001, .9999))
        scores = {k: np.full(len(y), np.nan) for k in modes}
        sizes = []
        for f in np.unique(fold):
            t = np.flatnonzero((fold%2 == f%2) & (fold != f) & hmask)
            q = np.flatnonzero((fold == f) & qmask)
            assert not np.isin(group[t], group[q]).any()
            sizes.append(dict(training_rows=len(t), training_exceptions=int(actual[t].sum()), query_rows=len(q)))
            scores['expert_mean'][q] = oriented_q[q].mean(1)
            for mode in modes[1:]:
                if mode == 'combined_tree':
                    model = HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=5, min_samples_leaf=25,
                                                           l2_regularization=6, early_stopping=False, random_state=314)
                    train, test = np.column_stack([h[t], x[t]]), np.column_stack([z[q], x[q]])
                else:
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=400))
                    train, test = (h[t], z[q]) if mode == 'opinion_logistic' else (x[t], x[q])
                scores[mode][q] = model.fit(train, actual[t]).predict_proba(test)[:, 1]
        side_summary = dict(n=int(qmask.sum()), exceptions=int(actual[qmask].sum()),
                            baserate=float(actual[qmask].mean()), by_fold=sizes,
                            observed=profile(raw, qmask, actual), models={})
        for k, p in scores.items():
            assert np.array_equal(np.isfinite(p), qmask)
            adopted = qmask & (p > .5)
            decisions[k][adopted] = ~reference[adopted]
            # A label-free review budget within each held-out fold for ranking comparisons.
            top = []
            for f in np.unique(fold):
                q = np.flatnonzero(qmask & (fold == f))
                top.extend(q[np.argsort(-p[q], kind='stable')[:ceil(.10*len(q))]])
            hit = int(actual[adopted].sum())
            side_summary['models'][k] = dict(auc=float(roc_auc_score(actual[qmask], p[qmask])),
                average_precision=float(average_precision_score(actual[qmask], p[qmask])),
                reviewed=len(top), review_hits=int(actual[top].sum()), adopted=int(adopted.sum()),
                fixed=hit, harmed=int(adopted.sum())-hit)
        summary[str(side)] = side_summary
    assert sum(v['exceptions'] for v in summary.values()) == common.sum() == 257
    report = dict(sides=summary, deployed={})
    for mode, p in decisions.items():
        report['deployed'][mode] = dict(full=stats(p, reference, y, group, fold),
              unseen=stats(p[unseen], reference[unseen], y[unseen], group[unseen], fold[unseen]),
              changed_common_that_v131_also_missed=int((common[rows] & (champion != old_y) & (p[rows] == old_y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/unanimous_exceptions.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    for side, v in summary.items():
        for mode, m in v['models'].items():
            print(side, mode, m['review_hits'], m['reviewed'], m['fixed'], m['harmed'], round(m['auc'], 3))


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
