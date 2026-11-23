"""Rank each unanimous mistake against correct passengers with comparable confidence."""
import json

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from full_roster_metric_adjudication_n23 import load_scores
from half_roster_disagreement_n23 import DEST
from matched_consensus_errors_n23 import roster_fields


def half_bank(raw, y, fold):
    p = {}
    def read(path, keys):
        with np.load(path) as z:
            for k, expected in [('y', y), ('panel', fold % 2), ('passenger', raw.PassengerId)]:
                np.testing.assert_array_equal(z[k], expected)
            return [z[k] for k in keys]
    for k in ['tabicl', 'v44_without_name']:
        p[k], = read(DEST / f'{k}.npz', ['p'])
    p['cat'], p['budget'] = read(DEST / 'others.npz', ['cat', 'budget'])
    p['mitra96'] = np.empty(len(y))
    for h in (0, 1):
        t, q = np.flatnonzero(fold % 2 != h), np.flatnonzero(fold % 2 == h)
        with np.load(DEST / f'mitra_{h}.npz') as z:
            for k, expected in [('train', t), ('query', q), ('labels', y[t]), ('steps', 15)]:
                np.testing.assert_array_equal(z[k], expected)
            p['mitra96'][q] = z['p']
    return np.column_stack([p[k] for k in (*FAMILIES, 'mitra96')])


def features(prob):
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    direction = prob[:, -1] >= .5
    allowed = ((prob >= .5) == direction[:, None]).all(1)
    confidence = np.where(direction[:, None], prob, 1-prob)
    scores = np.column_stack([logit(confidence.clip(.001, .999)), direction])
    # Same predicted class and five-percentage-point bands of mean confidence.
    band = np.minimum((20*confidence.mean(1)).astype(int), 19) + 20*direction
    return direction, allowed, scores, band


def paired_rows(rows, errors, band, groups, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for b in np.unique(band[rows]):
        members = np.flatnonzero(band[rows] == b)
        correct = members[~errors[rows[members]]]
        for case in members[errors[rows[members]]]:
            candidates = correct[groups[rows[correct]] != groups[rows[case]]]
            for neighbor in rng.choice(candidates, min(8, len(candidates)), replace=False):
                pairs.append((case, neighbor))
    return np.array(pairs, dtype=int)


def logistic(x, y, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.2, max_iter=300))
    return model.fit(x, y).predict_proba(test)[:, 1]


def conditional_auc(risk, error, band):
    total, weight = 0., 0
    for b in np.unique(band):
        use = band == b
        n = int(error[use].sum())*int((~error[use]).sum())
        if n:
            total += n*roc_auc_score(error[use], risk[use])
            weight += n
    return total / weight


def summarize(risk, error, allowed, direction, band, y, groups, fold, hard):
    ids = np.flatnonzero(allowed)
    assert np.array_equal(np.isfinite(risk), allowed)
    flip = allowed & (risk >= .5)
    fixed, harmed = flip & error, flip & ~error
    gains = fixed.astype(int)-harmed
    ordered = ids[np.argsort(-risk[ids], kind='stable')]
    cuts = np.r_[np.flatnonzero(np.diff(risk[ordered]) != 0)+1, len(ordered)]
    curve = np.cumsum(2*error[ordered].astype(int)-1)
    oracle_gain = int(max(0, curve[cuts-1].max()))
    budgets = {}
    for percent in [1, 5, 10]:
        top = ordered[:round(len(ordered)*percent/100)]
        budgets[str(percent)] = dict(fixed=int(error[top].sum()), harmed=int((~error[top]).sum()))
    return dict(eligible=len(ids), common_errors=int(error[ids].sum()), accuracy=float(((direction ^ flip) == y).mean()),
                auc=float(roc_auc_score(error[ids], risk[ids])), within_band_auc=conditional_auc(risk[ids], error[ids], band[ids]),
                average_precision=float(average_precision_score(error[ids], risk[ids])),
                fixed=int(fixed.sum()), harmed=int(harmed.sum()), historical_fixed=int((fixed & hard).sum()),
                historical_harmed=int((harmed & hard).sum()), fold_net=[int(gains[fold == f].sum()) for f in np.unique(fold)],
                ci_pp=interval(gains, groups), ranked_budgets=budgets, hindsight_best_threshold_net=oracle_gain)


def run():
    raw, y, g, fold = data()
    full = np.column_stack([load_scores(raw, y, fold, OUT / f'{k}.npz', ['p'])[:, 0] for k in (*FAMILIES, 'mitra96')])
    half = half_bank(raw, y, fold)
    hd, ha, hx, hb = features(half)
    fd, fa, fx, fb = features(full)
    he, fe = hd != y, fd != y
    own, peer = roster_fields(np.arange(len(y)), g.astype(int))
    fields = np.column_stack([own, peer]).astype('float32')
    risk = {k: np.full(len(y), np.nan) for k in ['scores', 'direct_context', 'matched_context']}
    coverage = []
    for f in np.unique(fold):
        t = np.flatnonzero((fold%2 == f%2) & (fold != f) & ha)
        q = np.flatnonzero((fold == f) & fa)
        assert not np.isin(g[t], g[q]).any()
        risk['scores'][q] = logistic(hx[t], he[t], fx[q])
        model = CatBoostClassifier(iterations=180, depth=4, learning_rate=.045, thread_count=3,
                                   verbose=False, random_seed=SEED, allow_writing_files=False)
        model.fit(np.column_stack([hx[t], fields[t]]), he[t].astype(int))
        risk['direct_context'][q] = model.predict_proba(np.column_stack([fx[q], fields[q]]))[:, 1]
        estimates, pair_counts = [], []
        for a, b in GroupKFold(2, shuffle=True, random_state=SEED).split(t, groups=g[t]):
            fit, cal = t[a], t[b]
            assert not np.isin(g[fit], g[cal]).any()
            pairs = paired_rows(fit, he, hb, g, SEED+int(f))
            rank = CatBoostRanker(loss_function='PairLogit', iterations=160, depth=4, learning_rate=.045,
                                  thread_count=3, verbose=False, random_seed=SEED, allow_writing_files=False)
            # Vote direction is observable and lets field effects differ between the two predicted classes.
            rank.fit(Pool(np.column_stack([fields[fit], hd[fit]]), label=he[fit].astype(int), pairs=pairs))
            rcal = rank.predict(np.column_stack([fields[cal], hd[cal]]))
            rtest = rank.predict(np.column_stack([fields[q], fd[q]]))
            estimates.append(logistic(np.column_stack([hx[cal], rcal]), he[cal], np.column_stack([fx[q], rtest])))
            pair_counts.append(len(pairs))
        risk['matched_context'][q] = np.mean(estimates, axis=0)
        coverage.append(dict(training=len(t), train_errors=int(he[t].sum()), query=len(q), training_pairs=pair_counts))
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    result = dict(coverage=coverage, baseline_accuracy=float((fd == y).mean()),
                  methods={k: summarize(v, fe, fa, fd, fb, y, g, fold, hard) for k, v in risk.items()})
    dest = ROOT/'reports/nov23_matched_learning_curve/consensus_ranking.json'
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['methods'], indent=2))


if __name__ == '__main__':
    run()
