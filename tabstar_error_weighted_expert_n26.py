"""Learn on common mistakes of two inner-fold teachers, with a known selection-odds correction."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from original_tabstar_probe_n26 import ART
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def fields(raw, group):
    frame = raw.drop(columns=['PassengerId', 'Cabin']).copy()
    cabin = raw.Cabin.str.split('/', expand=True)
    frame['deck'], frame['berth'], frame['side'] = cabin[0], pd.to_numeric(cabin[1]), cabin[2]
    frame['party_size'] = raw.groupby(group).PassengerId.transform('size')
    frame['total'] = frame[FEES].sum(axis=1, min_count=1)
    frame['active'] = frame[FEES].gt(0).sum(axis=1)
    for fee in [*FEES, 'total']:
        frame[fee] = np.log1p(frame[fee])
    return pd.get_dummies(frame, dummy_na=True).to_numpy(dtype=float)


def tree():
    return HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=15, min_samples_leaf=32,
                                          early_stopping=False, random_state=5062)


def teachers(registration, hidden, y, train, query):
    scaler = StandardScaler().fit(hidden[train])
    proj = LogisticRegression(C=.025, max_iter=400).fit(scaler.transform(hidden[train]), y[train])
    physical = tree().fit(registration[train], y[train])
    return np.column_stack([physical.predict_proba(registration[query])[:, 1],
                            proj.predict_proba(scaler.transform(hidden[query]))[:, 1]])


def fit_fold(k, raw, reg, y, groups, fold):
    t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    with np.load(ART / f'base_embedding{k}.npz') as z:
        ids = z['ids']
        np.testing.assert_array_equal(np.sort(ids), np.arange(len(y)))
        hidden = z['embedding'][np.argsort(ids)]
    # Projection uses no outcomes. Both outer query and inner teacher query are excluded
    # from their teachers' supervised training, including all members of those bookings.
    pc = PCA(n_components=48, svd_solver='randomized', random_state=6179).fit(hidden[t])
    hidden = pc.transform(hidden)
    predictions = np.full((len(y), 2), np.nan)
    for source, held in GroupKFold(3, shuffle=True, random_state=1529).split(t, groups=groups[t]):
        a, b = t[source], t[held]
        assert not np.isin(groups[a], groups[b]).any()
        predictions[b] = teachers(reg, hidden, y, a, b)
    assert not np.isin(groups[q], groups[t]).any()
    predictions[q] = teachers(reg, hidden, y, t, q)
    assert np.isfinite(predictions).all()
    votes = predictions >= .5
    agreement = votes[:, 0] == votes[:, 1]
    preferred = votes[:, 0]
    weights = np.where(agreement[t] & (preferred[t] != y[t]), 3., 1.)
    x = np.column_stack([reg, hidden, predictions])
    baseline = tree().fit(x[t], y[t]).predict_proba(x[q])[:, 1]
    concentrate = tree().fit(x[t], y[t], sample_weight=weights).predict_proba(x[q])[:, 1]
    # In unanimous-teacher rows the minority label was oversampled threefold.
    # Undo its log-odds shift when applying the specialist to an unweighted population.
    adjustment = np.where(agreement[q], np.where(preferred[q], -np.log(3.), np.log(3.)), 0.)
    corrected = expit(logit(concentrate.clip(1.e-5, 1-1.e-5)) - adjustment)
    return q, np.column_stack([baseline, concentrate, corrected])


def run():
    raw, y, groups, fold, views, _ = projection(include_full=True)
    nine, known, _, _ = champions()
    reg = fields(raw, groups)
    pred = np.full((len(y), 3), np.nan)
    for k in np.unique(fold):
        q, scores = fit_fold(k, raw, reg, y, groups, fold)
        pred[q] = scores
        print('fold', k, 'rows', len(q), flush=True)
    assert np.isfinite(pred).all()
    basic, tilted, corrected = (pred >= .5).T
    results = {}
    for tag, (valid, parent) in views.items():
        assert known[valid].all()
        common = valid & (parent == nine)
        rules = dict(unweighted=basic != parent, weighted_raw=tilted != parent,
                     weighted_corrected=corrected != parent,
                     only_added_by_weighting=(corrected != parent) & (basic == parent),
                     only_removed_by_weighting=(corrected == parent) & (basic != parent),
                     corrected_confirmed=(corrected != parent) & (basic != parent))
        results[tag] = {n: {s: ticker(m, y != parent, r) for s, m in [('all', valid), ('common', common)]}
                        for n, r in rules.items()}
    print(results['full'], flush=True)
    dest = ROOT / 'reports/nov23_matched_learning_curve/tabstar_error_weighted_expert.json'
    dest.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
