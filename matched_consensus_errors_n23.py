"""Learn to distinguish errors from equally confident correct predictions in other bookings."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import panel
from broad_booking_canonical_n23 import load_roster
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ROOT, write
from broad_tree_field_dividends_n23 import hard_cohorts
from expanded_score_frontier_n23 import source_scores
from expanded_training_roster_n23 import eight_fold
from exaone_validation_data import CATEGORICAL


def roster_fields(row, groups):
    frame, booking, size, _ = load_roster()
    np.testing.assert_array_equal(booking[row], groups)
    fees = frame[BILLS]
    own = np.column_stack([pd.get_dummies(frame[CATEGORICAL]), frame.Age.fillna(-1)/40,
                           frame.CabinNumber.fillna(-100)/1000, np.log1p(fees.fillna(0)), fees.gt(0),
                           fees.isna(), np.log1p(size)])
    sums = pd.DataFrame(own).groupby(booking).transform('sum').to_numpy()
    peers = (sums-own) / np.maximum(size-1, 1)[:, None]
    return own[row], peers[row]


def matched(ids, wrong, risk, groups):
    controls = ids[~wrong[ids]]
    left, right = [], []
    for case in ids[wrong[ids]]:
        distance = np.abs(risk[controls] - risk[case])
        valid = np.flatnonzero((groups[controls] != groups[case]) & (distance <= .025))
        chosen = controls[valid[np.argsort(distance[valid], kind='stable')[:4]]]
        left.extend([case]*len(chosen))
        right.extend(chosen)
    return np.array(left, int), np.array(right, int)


def fitting(fields, intercept_scores, wrong, ids, query, pairs):
    left, right = pairs
    # Each matched case gets one unit of weight regardless of control count.
    mass = 1 / np.bincount(left, minlength=len(wrong))[left]
    scale = np.maximum(fields[ids].std(0), .25)
    diff = (fields[left]-fields[right]) / scale
    gap = intercept_scores[left]-intercept_scores[right]
    def objective(beta):
        logits = gap + diff @ beta
        return (mass @ np.logaddexp(0, -logits) + 5*np.dot(beta, beta),
                diff.T @ (mass * (expit(logits)-1)) + 10*beta)
    fit = minimize(objective, np.zeros(fields.shape[1]), jac=True, method='L-BFGS-B')
    assert fit.success, fit.message
    learned = ((fields - fields[ids].mean(0)) / scale) @ fit.x
    return calibrated(intercept_scores + learned, wrong, ids, query)


def calibrated(score, wrong, ids, query):
    shift = brentq(lambda b: expit(score[ids]+b).mean()-wrong[ids].mean(), -50, 50)
    return expit(score[query]+shift)


def tree_offset(fields, offset, wrong, train, query):
    model = CatBoostClassifier(depth=3, iterations=100, learning_rate=.04, verbose=False,
                               random_seed=981, thread_count=2, allow_writing_files=False)
    model.fit(Pool(fields[train], wrong[train].astype(int), baseline=offset[train]))
    residual = model.predict(fields[query], prediction_type='RawFormulaVal')
    updated = model.predict(Pool(fields[query], baseline=offset[query]), prediction_type='RawFormulaVal')
    np.testing.assert_allclose(updated, residual+offset[query], rtol=0, atol=1e-6)
    return expit(updated)


def predict(methods, seed, fold, groups, wrong, risk, offset, use, vote, tree=False):
    estimates = {k: np.full(len(wrong), np.nan) for k in methods}
    if tree:
        estimates['tree_and_peers'] = np.full(len(wrong), np.nan)
    evaluated = []
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        for a, b in GroupKFold(3, shuffle=True, random_state=seed).split(ids, groups=groups[ids]):
            t, q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            for direction in [False, True]:
                selected = use & (vote == direction)
                train, query = t[selected[t]], q[selected[q]]
                pairs = matched(train, wrong, risk, groups)
                assert len(pairs[0]) and not np.intersect1d(np.unique(pairs[0]), query).size
                evaluated.append(matched(query, wrong, risk, groups))
                for k, fields in methods.items():
                    estimates[k][query] = (calibrated(offset, wrong, train, query) if fields is None else
                                          fitting(fields, offset, wrong, train, query, pairs))
                if tree:
                    estimates['tree_and_peers'][query] = tree_offset(methods['personal_and_peers'], offset, wrong, train, query)
    return estimates, evaluated


def run():
    row, y, frame, groups, _ = load()
    _, _, hard, old_right = hard_cohorts(row, y)
    fold, _, ref = eight_fold(row, y)
    scores = np.column_stack([p for _, p in source_scores(row, y, fold)])
    vote = scores[:, 0] >= .5
    unanimous = ((scores >= .5) == vote[:,None]).all(1)
    paid = frame[BILLS].gt(0).any(axis=1).to_numpy()
    use = unanimous & (paid != vote)
    wrong = vote != y
    risk = np.where(vote, 1-scores.mean(1), scores.mean(1))
    offset = logit(risk.clip(.001, .999))
    own, peers = roster_fields(row, groups)
    methods = dict(risk=None, personal=own, personal_and_peers=np.column_stack([own, peers]))
    estimates, evaluated = predict(methods, 981, fold, groups, wrong, risk, offset, use, vote, tree=True)
    left = np.concatenate([a for a, _ in evaluated])
    right = np.concatenate([b for _, b in evaluated])
    reference = np.where(use, vote, ref)
    assert use.sum() == 5258 and wrong[use].sum() == 732
    result = {}
    for direction in [False, True]:
        selected = use & (vote == direction)
        ids = np.flatnonzero(vote[left] == direction)
        cases, inv, n = np.unique(left[ids], return_inverse=True, return_counts=True)
        controls = right[ids]
        details = {}
        for k, p in estimates.items():
            np.testing.assert_array_equal(np.isfinite(p), use)
            a, b = p[left[ids]], p[controls]
            ranks = np.bincount(inv, weights=(a>b)+.5*(a==b)) / n
            flip = selected & (p>=.5)
            details[k] = dict(rank_fraction=float(ranks.mean()), fixed=int((flip & wrong).sum()),
                              harmed=int((flip & ~wrong).sum()), hard_fixed=int((flip & wrong & hard).sum()),
                              rank_by_fold=[float(ranks[fold[cases] == f].mean()) for f in np.unique(fold)])
        result[str(direction)] = dict(size=int(selected.sum()), errors=int(wrong[selected].sum()),
                                      matched_cases=len(cases), unique_controls=len(np.unique(controls)),
                                      pairs=len(ids), methods=details)
    full = {k: paired((reference ^ (use & (p>=.5))) == y, ref == y) for k,p in estimates.items()}
    _, _, _, _, previous, _, _ = panel(row, y, frame)
    seen = np.logical_or.reduce(list(previous.values()))
    flips = {k: use & (p >= .5) for k, p in estimates.items()}
    flips['both_peer_models'] = flips['personal_and_peers'] & flips['tree_and_peers']
    overlap = {}
    for k, flip in flips.items():
        fixed, harmed = flip & wrong, flip & ~wrong
        overlap[k] = dict(fixed=int(fixed.sum()), harmed=int(harmed.sum()), hard_fixed=int((fixed & hard).sum()),
                          hard_fixed_without_previous_stable_fix=int((fixed & hard & ~seen).sum()),
                          historically_correct_harmed=int((harmed & old_right).sum()))
    repeated = [flips['personal_and_peers']]
    for seed in [410, 213]:
        rep, _ = predict({'personal_and_peers': methods['personal_and_peers']}, seed, fold, groups,
                         wrong, risk, offset, use, vote)
        repeated.append(use & (rep['personal_and_peers'] >= .5))
    matched_every_time = np.all(repeated, axis=0)
    nominees = dict(previously_uncovered=hard & wrong & ~seen, shared_wrong=hard & wrong,
                    historically_correct=old_right & ~wrong, current_correct=~wrong, current_wrong=wrong)
    reproduction = {k: dict(first=int((v & repeated[0]).sum()), all_three=int((v & matched_every_time).sum()))
                    for k, v in nominees.items()}
    write(ROOT / 'reports/nov23_matched_learning_curve/within_consensus.json',
          dict(scopes=result, vs_reference=full, overlap=overlap, reproduction=reproduction))
    print(result, full)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
