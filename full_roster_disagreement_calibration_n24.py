"""Adjudicate dissent on held-out orders with separately fitted routes and calibration."""
import json

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_joint_raw_dissent_n24 import load_maxout

FAMILIES = ('engineered_trees', 'maxout', 'spending_likelihood')


def subspace_predictions(rows, folds, groups, y, tag):
    from full_roster_permutation_selection_n24 import OUT

    predictions = np.empty((len(rows), 5))
    for fold in np.unique(folds):
        q = rows[folds == fold]
        t = rows[(folds != fold) & ~np.isin(groups[rows], groups[q])]
        with np.load(OUT / f'{tag}_{fold}.npz') as cache:
            for key, expected in [('train', t), ('query', q), ('labels', y[t])]:
                np.testing.assert_array_equal(cache[key], expected)
            predictions[folds == fold] = cache['prob']
    return predictions


def venn_bounds(calibration, outcomes, query):
    """Pool exact ties before each of the two isotonic fits at a query score."""
    values, bins = np.unique(calibration, return_inverse=True)
    counts = np.bincount(bins)
    hits = np.bincount(bins, weights=outcomes)
    bounds = np.empty((len(query), 2))
    for row, score in enumerate(query):
        index = np.searchsorted(values, score)
        if index < len(values) and values[index] == score:
            weights = counts.copy()
            sums = hits.copy()
            weights[index] += 1
        else:
            weights = np.insert(counts, index, 1)
            sums = np.insert(hits, index, 0.)
        for target in (0, 1):
            sums[index] += target
            bounds[row, target] = isotonic_regression(sums / weights, weights=weights).x[index]
    assert (bounds[:, 0] <= bounds[:, 1] + 1e-12).all()
    return bounds.T


def decisions(base, families, raw_fields, labels, group, folds, champion):
    direction = base >= .5
    opposition = np.where(direction[:, None], 1 - families, families)
    oppose = (families >= .5) != direction[:, None]
    conflict = oppose.any(axis=1)
    reference_risk = np.minimum(base, 1 - base)
    relative = logit(opposition.clip(.0001, .9999))
    simple = np.column_stack([direction, logit(reference_risk.clip(.0001, .5)), relative])
    # Identical risk inputs for the two routes. The second additionally observes passenger fields.
    designs = {'opinions': simple, 'opinions_and_registration': np.column_stack([simple, raw_fields])}
    error = labels != direction
    votes = {'one_family': conflict, 'two_families': oppose.sum(axis=1) >= 2}
    seen = np.zeros(len(labels), dtype=int)
    for fold in np.unique(folds):
        ix = np.flatnonzero(folds == fold)
        parts = [ix[q] for _, q in GroupKFold(4, shuffle=True, random_state=281).split(ix, groups=group[ix])]
        for part in range(4):
            held, calib = parts[part], parts[(part + 1) % 4]
            fit = np.r_[parts[(part + 2) % 4], parts[(part + 3) % 4]]
            assert not (set(group[fit]) & set(group[calib]) or set(group[calib]) & set(group[held])
                        or set(group[fit]) & set(group[held]))
            fit, calib, held = [ids[conflict[ids]] for ids in (fit, calib, held)]
            seen[held] += 1
            for name, x in designs.items():
                route = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=300))
                route.fit(x[fit], error[fit])
                score = route.predict_proba(x[held])[:, 1]
                lo, hi = venn_bounds(route.predict_proba(x[calib])[:, 1], error[calib], score)
                for kind, accept in [('raw', score > .5), ('regularized', hi/(1-lo+hi) > .5),
                                     ('both_labels', lo > .5)]:
                    votes.setdefault(name+'_'+kind, np.zeros(len(labels), dtype=bool))[held] = accept
    np.testing.assert_array_equal(seen, conflict)
    free = direction == champion
    predictions = {key: champion ^ (vote & free) for key, vote in votes.items()}
    return predictions, oppose & free[:, None]


def stats(pred, baseline, y, group, folds):
    change = (pred == y).astype(int) - (baseline == y)
    return dict(corrected=int((change == 1).sum()), harmed=int((change == -1).sum()),
                accuracy=float(np.mean(pred == y)), net=int(change.sum()), ci_pp=interval(change, group),
                fold_net=[int(change[folds == k].sum()) for k in np.unique(folds)])


def run():
    _, old_y, old_group, rows, oldfold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    raw_fields = passenger_vectors(raw)
    old_trees = subspace_predictions(rows, oldfold, group, y, 'historical')
    all_rows = np.arange(len(y))
    trees = subspace_predictions(all_rows, fold, group, y, 'full')
    with np.load(ROOT/'artifacts/nov24_zero_truncated/truncated.npz') as z:
        for k, v in [('rows', rows), ('oldfold', oldfold), ('fold', fold),
                     ('passenger', raw.PassengerId.to_numpy(str))]:
            np.testing.assert_array_equal(z[k], v)
        a = np.column_stack([old_trees.mean(1), load_maxout(rows, oldfold, group, y, 'historical'), z['historical']])
        b = np.column_stack([trees.mean(1), load_maxout(all_rows, fold, group, y, 'eightfold'), z['full']])
    pa, da = decisions(old[:, 0], a, raw_fields[rows], old_y, old_group, oldfold, champion)
    pb, db = decisions(full[:, names.index('mitra96')], b, raw_fields, y, group, fold, reference)
    common = ((full >= .5) != y[:, None]).all(axis=1)
    persistent = common[rows] & (champion != old_y)
    unseen = ~np.isin(group, old_group)
    assert common.sum() == 257 and persistent.sum() == 201 and unseen.sum() == 1044
    report = {'coverage': {}, 'decisions': {}}
    for i, name in enumerate((*FAMILIES, 'any_family')):
        old_dis = da[:, i] if i < len(FAMILIES) else da.any(1)
        new_dis = db[:, i] if i < len(FAMILIES) else db.any(1)
        report['coverage'][name] = dict(correctable_all=int((new_dis & (reference != y)).sum()),
            correctable_common=int((new_dis & common & (reference != y)).sum()),
            persistent_both=int((persistent & old_dis & new_dis[rows] & (reference[rows] != old_y)).sum()),
            at_risk_all=int((new_dis & (reference == y)).sum()))
    for k, p in pb.items():
        report['decisions'][k] = dict(historical=stats(pa[k], champion, old_y, old_group, oldfold),
            full=stats(p, reference, y, group, fold), unseen=stats(p[unseen], reference[unseen], y[unseen], group[unseen], fold[unseen]),
            common_repaired=int((common & (reference != y) & (p == y)).sum()),
            persistent_repaired_both=int((persistent & (pa[k] == old_y) & (reference[rows] != old_y) & (p[rows] == old_y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/disagreement_calibration.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(report['coverage'])
    for k, v in report['decisions'].items():
        print(k, [v[s]['net'] for s in ('historical', 'full', 'unseen')], v['common_repaired'])


if __name__ == '__main__':
    run()
