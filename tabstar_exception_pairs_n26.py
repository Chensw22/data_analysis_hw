"""Learn within-manifest exception rankings and calibrate them on separate bookings."""
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from original_tabstar_probe_n26 import ART
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def challenge(x, strata, target, groups, t, q, seed):
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    a, c = next(inner.split(t, groups=groups[t]))
    fit_ids, cal = t[a], t[c]
    assert not np.isin(groups[cal], groups[fit_ids]).any() and not np.isin(groups[t], groups[q]).any()
    scale = StandardScaler().fit(x[fit_ids])
    features = scale.transform(x).clip(-5, 5)
    rng = np.random.default_rng(seed)
    pair_diffs = []
    # Compare exceptions and ordinary outcomes with matching origin and deck when possible.
    for key in np.unique(strata[fit_ids]):
        subset = fit_ids[strata[fit_ids] == key]
        cases, others = subset[target[subset]], subset[~target[subset]]
        for case in cases:
            eligible = others[groups[others] != groups[case]]
            if len(eligible):
                controls = rng.choice(eligible, size=min(4, len(eligible)), replace=False)
                pair_diffs.extend(features[case] - features[controls])
    deltas = np.stack(pair_diffs)
    pair_model = LogisticRegression(fit_intercept=False, C=.03, max_iter=400)
    pair_model.fit(np.vstack([deltas, -deltas]), np.repeat([1, 0], len(deltas)))
    order_score = pair_model.decision_function(features)
    # The case-control sample intentionally has equal numbers of both labels.
    # Calibrate on untouched bookings, including the prevalence of each origin/deck stratum.
    calibration = np.column_stack([order_score, (strata[:, None] == np.unique(strata[fit_ids])).astype(float)])
    calibrated = LogisticRegression(C=1., max_iter=400).fit(calibration[cal], target[cal])
    direct = LogisticRegression(C=.03, max_iter=400).fit(features[fit_ids], target[fit_ids])
    return direct.predict_proba(features[q])[:, 1], calibrated.predict_proba(calibration[q])[:, 1]


def run():
    raw, y, group, f, views, _ = projection(include_full=True)
    nine, known, _, _ = champions()
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    cohorts = {'awake_paid': raw.CryoSleep.eq(False).to_numpy() & paid,
               'asleep_no_positive': raw.CryoSleep.eq(True).to_numpy() & ~paid}
    stratum = (raw.HomePlanet.fillna('?') + '/' + raw.Cabin.str.split('/').str[0].fillna('?')).to_numpy()
    report = {}
    for name, eligible in cohorts.items():
        usual = name == 'asleep_no_positive'
        target = y != usual
        scores = np.full((len(y), 2), np.nan)
        for k in np.unique(f):
            train = np.flatnonzero((f != k) & eligible)
            query = np.flatnonzero((f == k) & eligible)
            with np.load(ART / f'base_embedding{k}.npz') as z:
                np.testing.assert_array_equal(np.sort(z['ids']), np.arange(len(y)))
                emb = z['embedding'][np.argsort(z['ids'])]
            pca = PCA(n_components=48, random_state=6671, svd_solver='randomized').fit(emb[train])
            x = pca.transform(emb)
            scores[query] = np.column_stack(challenge(x, stratum, target, group, train, query, 6102 + k))
        assert np.isfinite(scores[eligible]).all()
        direct, pairs = (scores >= .5).T
        report[name] = {}
        for key, (valid, parent) in views.items():
            assert known[valid].all()
            same = valid & (parent == nine) & eligible & (parent == usual)
            actions = dict(direct=direct, paired=pairs, agree=pairs & direct, only_pairs=pairs & ~direct)
            report[name][key] = {n: ticker(same, y != parent, action) for n, action in actions.items()}
        print(name, report[name]['full'], flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/tabstar_exception_pairs.json'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
