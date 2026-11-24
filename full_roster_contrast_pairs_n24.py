"""Learn a shared sparse passenger score from global and local cross-order comparisons."""
import json

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.spatial import cKDTree
from scipy.special import expit
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_kernel_repair_overlap_n24 import from_cache


def basis(raw_x, train):
    """Use the same label-free features for ordinary and conditional logistic loss."""
    x = (raw_x - raw_x[train].mean(0)) / raw_x[train].std(0).clip(.3)
    rng = np.random.default_rng(34127)
    directions = rng.normal(size=(x.shape[1], 128))
    directions /= np.linalg.norm(directions, axis=0)
    projection = x @ directions
    # One knot per direction from a fixed rotating set of training quantiles.
    cuts = np.quantile(projection[train], [.2, .4, .6, .8], axis=0)
    hinges = np.maximum(projection - cuts[np.arange(128) % 4, np.arange(128)], 0)
    hinges = (hinges - hinges[train].mean(0)) / hinges[train].std(0).clip(.3)
    return x, np.column_stack([x, hinges])


def comparisons(x, y, group):
    """Draw pairs independently of labels, then condition on exactly one positive."""
    rng = np.random.default_rng(51931)
    n = len(x)
    # Break ties at the neighbor search boundary independently of labels and row order.
    geometry = x + rng.normal(scale=1e-8, size=x.shape)
    nearest = cKDTree(geometry).query(geometry, k=49)[1]
    owners = np.repeat(np.arange(n), 16)
    local = nearest[owners, rng.integers(1, 49, len(owners))]
    global_ = rng.integers(0, n, len(owners))
    samples = []
    weights = []
    counts = {}
    for name, partners in [('global', global_), ('local', local)]:
        eligible = group[owners] != group[partners]
        observed = eligible & (y[owners] != y[partners])
        left, right = owners[observed], partners[observed]
        pos = np.where(y[left], left, right)
        neg = np.where(y[left], right, left)
        assert y[pos].all() and not y[neg].any() and (group[pos] != group[neg]).all()
        samples.append(np.column_stack([pos, neg]))
        weights.append(np.full(len(pos), .5 / len(pos)))
        counts[name] = dict(eligible=int(eligible.sum()), discordant=len(pos))
    pairs = np.concatenate(samples)
    return pairs[:, 0], pairs[:, 1], np.concatenate(weights), counts


def fit(x, y, pair_data=None):
    """Sparse logistic scores; differences eliminate the intercept in the pair fit."""
    dim = x.shape[1]
    # Split each coefficient into positive/negative parts to solve the L1 problem.
    def objective(parameters):
        coef = parameters[:dim] - parameters[dim:2*dim]
        score = x @ coef + parameters[-1]
        if pair_data is None:
            loss = np.mean(np.logaddexp(0, score) - y * score)
            dscore = (expit(score) - y) / len(y)
        else:
            pos, neg, w = pair_data
            difference = score[pos] - score[neg]
            loss = w @ np.logaddexp(0, -difference)
            partial = -w * expit(-difference)
            dscore = np.bincount(pos, weights=partial, minlength=len(y))
            dscore -= np.bincount(neg, weights=partial, minlength=len(y))
        penalty = .0015
        gradient = x.T @ dscore + .0005 * coef
        value = loss + penalty * parameters[:-1].sum() + .00025 * (coef @ coef)
        return value, np.r_[gradient + penalty, -gradient + penalty, dscore.sum()]
    bounds = [(0, None)] * (2*dim) + ([(None, None)] if pair_data is None else [(0, 0)])
    estimate = minimize(objective, np.zeros(2*dim+1), jac=True, bounds=bounds,
                        method='L-BFGS-B', options={'maxiter': 700, 'ftol': 1e-10, 'gtol': 1e-6})
    assert estimate.success, estimate.message
    return estimate.x[:dim] - estimate.x[dim:2*dim]


def predict(fields, y, groups, rows, fold, tag):
    out = np.full((len(rows), 2), np.nan)
    audit = []
    for f in np.unique(fold):
        held = fold == f
        q = rows[held]
        t = rows[~held & ~np.isin(groups[rows], groups[q])]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=889).split(t, groups=groups[t]))
        train, calibration = t[a], t[b]
        for left, right in [(train, calibration), (train, q), (calibration, q)]:
            assert not np.isin(groups[left], groups[right]).any()
        geometry, x = basis(fields, train)
        pos, neg, w, counts = comparisons(geometry[train], y[train], groups[train])
        fits = {}
        for j, rule in enumerate(('point', 'pair')):
            coef = fit(x[train], y[train], (pos, neg, w) if rule == 'pair' else None)
            unscaled = x @ coef
            # Re-estimate only the absolute intercept, identically for both objectives.
            intercept = brentq(lambda v: expit(unscaled[calibration] + v).mean() - y[calibration].mean(),
                               -100., 100.)
            out[held, j] = expit(unscaled[q] + intercept)
            fits[rule] = dict(nonzero=int((abs(coef) > 1e-7).sum()), intercept=float(intercept))
        audit.append(dict(fold=int(f), train=len(train), calibration=len(calibration), query=len(q),
                          pairs=counts, fits=fits))
        print(tag, int(f), counts, flush=True)
    assert np.isfinite(out).all()
    return out, audit


def run():
    _, oy, og, rows, old_fold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, groups, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, groups, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, groups, fold)
    np.testing.assert_array_equal(gate >= .5, ref)
    features = passenger_vectors(raw)
    old_scores, old_audit = predict(features, y, groups, rows, old_fold, 'historical')
    new_scores, full_audit = predict(features, y, groups, np.arange(len(y)), fold, 'full')
    _, prev = from_cache('full', np.arange(len(y)), fold, y, groups)
    shared = ((full >= .5) != y[:, None]).all(1)
    unrescued = shared & ((prev >= .5) != y[:, None]).all(1)
    assert (shared.sum(), unrescued.sum()) == (257, 243)
    persistent = shared[rows] & (champion != oy)
    unseen = ~np.isin(groups, og)
    result = {}
    for j, rule in enumerate(('point', 'pair')):
        a, b = old_scores[:, j], new_scores[:, j]
        historical_quarter = np.where(champion == (old[:, 0] >= .5),
                                      .75 * old[:, 0] + .25 * a >= .5, champion)
        mitra = full[:, names.index('mitra96')]
        protected_quarter = np.where(ref == (mitra >= .5), .75 * mitra + .25 * b >= .5, ref)
        for mode, h, n in [('direct', a >= .5, b >= .5),
                           ('quarter_gate', historical_quarter, .75 * gate + .25 * b >= .5),
                           ('quarter_protected_mitra', historical_quarter, protected_quarter)]:
            right = n == y
            result[f'{rule}_{mode}'] = dict(
                historical=stats(h, champion, oy, og, old_fold), full=stats(n, ref, y, groups, fold),
                new_orders=stats(n[unseen], ref[unseen], y[unseen], groups[unseen], fold[unseen]),
                common_correct=int((shared & right).sum()), new_on_243=int((unrescued & right).sum()),
                new_on_243_by_true_class=[int((unrescued & right & (y == v)).sum()) for v in (False, True)],
                same_method_twice_on_201=int((persistent & (h == oy) & right[rows]).sum()))
    direct = (new_scores >= .5) == y[:, None]
    old_position = {int(row): i for i, row in enumerate(rows)}
    cases = []
    for row in np.flatnonzero(unrescued & direct.any(1)):
        previous = old_position.get(int(row))
        cases.append(dict(row=int(row), passenger=raw.PassengerId.iat[row], transported=bool(y[row]),
                          full_scores=new_scores[row].tolist(),
                          historical_scores=old_scores[previous].tolist() if previous is not None else None))
    report = dict(inputs='row-local compact fields; no learned expert scores or Name', audit=dict(
        historical=old_audit, full=full_audit), results=result,
        cases_new_to_the_four_models=cases, score_column_order=['point', 'pair'],
        direct_243_overlap=dict(point_only=int((unrescued & direct[:,0] & ~direct[:,1]).sum()),
                                pair_only=int((unrescued & ~direct[:,0] & direct[:,1]).sum()),
                                both=int((unrescued & direct.all(1)).sum())))
    path = ROOT / 'reports/nov23_matched_learning_curve/contrast_pairs.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    for k, v in result.items():
        print(k, [(v[p]['corrected'], v[p]['harmed']) for p in ('historical', 'full', 'new_orders')],
              v['new_on_243'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
