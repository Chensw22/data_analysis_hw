"""Estimate the predictive contrast between recorded awake and asleep states at comparable manifests."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from frozen_queue_shared_errors_n26 import paired
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def design(raw, groups):
    cabin = raw.Cabin.str.split('/', expand=True)
    table = raw[['Age', 'Destination', 'HomePlanet', 'VIP']].copy()
    table['deck'], table['side'], table['berth'] = cabin[0], cabin[2], pd.to_numeric(cabin[1])
    table['booking_size'] = raw.groupby(groups).PassengerId.transform('size')
    table['slot'] = raw.PassengerId.str.split('_').str[1].astype(int)
    for col in FEES:
        table[col + '_missing'] = raw[col].isna()
    peer_fields = {'age': raw.Age, 'sleep': raw.CryoSleep.map({True: 1., False: 0.}),
                   'active': raw[FEES].gt(0).any(axis=1).astype(float),
                   'spend': np.log1p(raw[FEES].sum(axis=1, min_count=1))}
    for name, val in peer_fields.items():
        obs = val.notna().astype(int)
        counts = obs.groupby(groups).transform('sum') - obs
        sums = val.fillna(0).groupby(groups).transform('sum') - val.fillna(0)
        table['other_' + name] = sums / counts.replace(0, np.nan)
    return pd.get_dummies(table, dummy_na=True).to_numpy(float)


def classifier():
    return HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=12, min_samples_leaf=32,
                                           l2_regularization=3, early_stopping=False, random_state=6415)


def nuisance(x, y, sleep, t, q):
    return np.column_stack([classifier().fit(x[t], target[t]).predict_proba(x[q])[:,1] for target in (y, sleep)])


def fold_estimates(x, y, w, g, t, q, seed):
    xs = np.column_stack([x, w])
    direct = classifier().fit(xs[t], y[t]).predict_proba(xs[q])[:,1]
    inner = np.full((len(t), 2), np.nan)
    for a, b in GroupKFold(3, shuffle=True, random_state=seed).split(t, groups=g[t]):
        assert not np.isin(g[t[a]], g[t[b]]).any()
        inner[b] = nuisance(x, y, w, t[a], t[b])
    assert np.isfinite(inner).all() and not np.isin(g[t], g[q]).any()
    m, e = nuisance(x, y, w, t, q).T
    # The contrast is predictive, restricted to the recorded no-positive-bill cohort.
    resid = w[t].astype(float) - inner[:,1].clip(.02, .98)
    weight = resid**2
    response = (y[t].astype(float) - inner[:,0]) / resid
    adjustment = w[q].astype(float) - e
    rng = np.random.default_rng(seed)
    sham = response.copy()
    bins = np.minimum((inner[:,1] * 10).astype(int), 9)
    strata = bins * 2 + w[t].astype(int)
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        sham[rows] = response[rng.permutation(rows)]
    estimates = [direct, m + adjustment * np.average(response, weights=weight)]
    for label in [response, sham]:
        model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=8, min_samples_leaf=36,
                                               l2_regularization=5, early_stopping=False, random_state=6415)
        model.fit(x[t], label, sample_weight=weight)
        estimates.append(m + adjustment * model.predict(x[q]).clip(-1, 1))
    return np.column_stack(estimates).clip(0, 1)


def run():
    raw, y, groups, folds, known, versions = paired()
    x = design(raw, groups)
    sleep = raw.CryoSleep.eq(True).to_numpy()
    eligible = raw.CryoSleep.notna().to_numpy() & ~raw[FEES].gt(0).any(axis=1).to_numpy()
    scores = np.full((len(y), 4), np.nan)
    for k in np.unique(folds):
        t, q = [np.flatnonzero(eligible & m) for m in (folds != k, folds == k)]
        scores[q] = fold_estimates(x, y, sleep, groups, t, q, 3951 + k)
    assert np.isfinite(scores[eligible]).all()
    archive = ROOT/'artifacts/nov26_manifest_sleep_contrast'
    archive.mkdir(exist_ok=True)
    np.savez_compressed(archive/'probabilities.npz', scores=scores, passenger=raw.PassengerId.to_numpy(str),
                        eligible=eligible, folds=folds)
    parent = versions[:,-1]
    unanimous = known & (versions == parent[:,None]).all(1)
    pairwise = known & (versions[:,0] == parent)
    masks = dict(all_known_state_no_positive=known & eligible, both_releases=pairwise & eligible,
                 all_versions=unanimous & eligible, sleeper_false_alarms=unanimous & eligible & sleep & parent)
    predictions = scores >= .5
    real = predictions[:,2] != parent
    rules = {n: predictions[:,j] != parent for j,n in enumerate(['direct','constant','learned','permuted'])}
    rules.update(new_vs_constant=real & ~rules['constant'], new_vs_permuted=real & ~rules['permuted'],
                 independently_confirmed=real & rules['direct'])
    report = {}
    for n, rule in rules.items():
        report[n] = {s: ticker(mask, y != parent, rule) for s, mask in masks.items()}
        m = masks['sleeper_false_alarms']
        report[n]['sleeper_fold_net'] = [int(np.sum(np.where(parent != y,1,-1) * rule * m * (folds == k)))
                                         for k in np.unique(folds)]
    assert report['direct']['sleeper_false_alarms']['errors'] == 388
    print({k:v['sleeper_false_alarms'] for k,v in report.items()}, flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/manifest_sleep_contrast.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
