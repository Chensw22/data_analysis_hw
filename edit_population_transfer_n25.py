"""Audit local gains after matching the populations actually changed in three scored submissions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from frozen_joint_corroboration_n25 import decisions
from kaggle_workflow import sha256
from mitra_four_outcomes_n25 import FEES

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports'
NAMES = ('proposal_n25_frozen_binary', 'proposal_n25_frozen_binary_confirmed', 'proposal_n25_frozen_binary_matched_support')


def scored(name, test):
    folder = REPORT / name
    path = ART / name / 'submission.csv'
    receipt, = (folder / 'online_attempts').glob('*.json')
    record = json.loads(receipt.read_text())
    assert record['file_sha256'] == sha256(path) and record['evaluation']['Status'] == 'COMPLETE'
    pred = pd.read_csv(path)
    np.testing.assert_array_equal(pred.PassengerId, test.PassengerId)
    return pred.Transported.to_numpy(), float(record['evaluation']['Public Score'])


def load_predictions(tag, raw, ids, fold, inc, prev):
    n = len(ids)
    native, matched, scores = np.empty(n), np.empty(n), np.empty((n, 4))
    for k in np.unique(fold):
        chosen = fold == k
        part = 'test' if tag == 'test' else f'fold{k}'
        with np.load(ART / 'nov25_mitra_four_outcomes' / (part + '.npz')) as a, np.load(
            ART / 'nov25_mitra_stratified_support' / (part + '.npz')) as b:
            for z in (a, b):
                np.testing.assert_array_equal(z['q'], ids[chosen])
                np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[ids[chosen]])
            scores[chosen], native[chosen], matched[chosen] = a['scores'], a['native'], b['scores'][:, 0]
    with np.load(ART / 'nov25_mitra_token_trees/predictions.npz') as z:
        index = z['model_names'].tolist().index('joint')
        if tag == 'test':
            np.testing.assert_array_equal(z['test_ids'], raw.PassengerId.iloc[ids])
            tree = z['test_by_fold'].mean(0)[:, index]
        else:
            np.testing.assert_array_equal(z['rows'], ids)
            tree = z['historical'][:, index]
    free = ((native >= .5) == prev) & (prev == inc)
    policies = decisions(scores, tree, native, prev, inc)
    edits = np.column_stack([free & ((scores[:, 0] >= .5) != inc), policies['binary_confirmed'],
                             free & ((.5 * (scores[:, 0] + matched) >= .5) != inc)])
    evidence = np.column_stack([inc, native, scores[:, :3], matched, tree])
    return edits, evidence


def domain_weights(x, domain, groups):
    odds = np.full(len(x), np.nan)
    score = odds.copy()
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=889)
    for t, q in cv.split(x, domain, groups):
        assert not np.intersect1d(groups[t], groups[q]).size
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.08, max_iter=500))
        model.fit(x[t], domain[t])
        p = model.predict_proba(x[q])[:, 1].clip(.001, .999)
        score[q] = p
        odds[q] = p / (1 - p) * ((~domain[t]).sum() / domain[t].sum())
    assert np.isfinite(score).all()
    return odds[~domain].clip(.25, 4.), float(roc_auc_score(domain, score))


def describe(delta, weights, groups, target_rate):
    _, codes = np.unique(groups, return_inverse=True)
    by_group = np.bincount(codes, weights=delta * weights)
    mass = np.bincount(codes, weights=weights)
    rng = np.random.default_rng(1422)
    draws = rng.integers(len(mass), size=(1200, len(mass)))
    boot = by_group[draws].sum(1) / mass[draws].sum(1)
    return dict(fraction_beneficial=float(np.average(delta > 0, weights=weights)),
                expected_accuracy_delta_pp=float(100 * target_rate * np.average(delta, weights=weights)),
                order_bootstrap_pp=(100 * target_rate * np.quantile(boot, [.025, .975])).tolist(),
                effective_passengers=float(weights.sum() ** 2 / (weights @ weights)))


def run():
    train = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda c: c != 'Name')
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=lambda c: c != 'Name')
    raw = pd.concat([train.drop(columns='Transported'), test], ignore_index=True)
    group = raw.PassengerId.str.split('_').str[0]
    assert not np.intersect1d(group[:len(train)], group[len(train):]).size
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        rows, fold, inc, prev = [z[k] for k in ('rows', 'fold', 'nine', 'v131')]
        np.testing.assert_array_equal(z['y'], train.Transported.iloc[rows])
    historical_edits, hist_scores = load_predictions('historical', raw, rows, fold, inc, prev)
    ids = np.arange(len(train), len(raw))
    best, best_score = scored('proposal_n24_selected_dissent', test)
    previous, _ = scored('proposal_v131_party_consensus', test)
    test_edits, test_scores = load_predictions('test', raw, ids, np.zeros(len(test), int), best, previous)
    bills = raw[FEES]
    summary = np.column_stack([raw.Age.fillna(28) / 40, raw.Age.isna(), raw.CryoSleep.eq(True), raw.Cabin.isna(),
                               bills.gt(0).sum(axis=1), bills.isna().sum(axis=1), np.log1p(bills.fillna(0)),
                               np.log(group.groupby(group).transform('size')),
                               pd.get_dummies(raw[['HomePlanet','Destination']].astype('string').fillna('?'), dtype=float)])
    effects = np.where(inc == train.Transported.iloc[rows].to_numpy(), -1, 1)
    report = {}
    for j, name in enumerate(NAMES):
        hard, public = scored(name, test)
        h, q = historical_edits[:, j], test_edits[:, j]
        np.testing.assert_array_equal(hard != best, q)
        domain = np.r_[np.zeros(h.sum(), bool), np.ones(q.sum(), bool)]
        g = group.iloc[np.r_[rows[h], ids[q]]].to_numpy()
        x = np.vstack([hist_scores[h], test_scores[q]])
        physical = summary[np.r_[rows[h], ids[q]]]
        rate = q.mean()
        entry = dict(history_changes=int(h.sum()), test_changes=int(q.sum()), public_delta_pp=round(100 * (public - best_score), 5),
                     training_wrong=int((effects[h] > 0).sum()), by_direction={})
        weights = np.ones(h.sum())
        for side in (False, True):
            old = inc[h] == side
            expected = (best[q] == side).mean()
            weights[old] = expected / old.mean()
            entry['by_direction'][str(side)] = dict(history=int(old.sum()), test=int((best[q] == side).sum()),
                                                   history_fixed=int((effects[h][old] > 0).sum()))
        entry['unweighted'] = describe(effects[h], np.ones(h.sum()), g[~domain], rate)
        entry['direction_matched'] = describe(effects[h], weights, g[~domain], rate)
        for label, features in [('model_scores', x), ('passenger_and_scores', np.column_stack([x, physical]))]:
            w, auc = domain_weights(features, domain, g)
            entry[label] = dict(domain_auc=auc, **describe(effects[h], w, g[~domain], rate))
        report[name] = entry
        print(name, {k:round(entry[k]['expected_accuracy_delta_pp'],3)
                     for k in ('unweighted', 'direction_matched', 'model_scores', 'passenger_and_scores')}, flush=True)
    output = dict(method='Retrospective diagnostic. Target covariates are unlabeled. Intervals condition on estimated weights, target edits and target edit rate.',
                  results=report)
    (REPORT / 'nov23_matched_learning_curve/edit_population_transfer.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    run()
