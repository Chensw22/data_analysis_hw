"""Predict transport from two competing banks of affine passenger profiles."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import panel
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import coordinates
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

LABEL = 'competing_affine_profiles'
REPORT = ROOT / 'reports' / ('nov23_' + LABEL)
PROFILES = 6


def design(frame, train):
    encoded, _ = coordinates(frame, train)
    bills = np.nan_to_num(frame[BILLS].to_numpy(float))
    total = bills.sum(1, keepdims=True)
    x = np.column_stack([encoded, np.log1p(bills), np.log1p(total), bills / (total + 1)])
    scale = np.maximum(x[train].std(0), .25)
    return np.column_stack([np.ones(len(x)), (x - x[train].mean(0)) / scale])


def response(x, params):
    planes = (x @ params.T).reshape(len(x), 2, PROFILES)
    pooled = logsumexp(planes, axis=2)
    return pooled[:, 1] - pooled[:, 0], planes


def objective(flat, x, target):
    params = flat.reshape(2 * PROFILES, -1)
    logits, planes = response(x, params)
    residual = expit(logits) - target
    weights = softmax(planes, axis=2)
    weights[:, 0] *= -1
    gradient = (weights.reshape(len(x), -1) * residual[:, None]).T @ x
    loss = np.sum(np.logaddexp(0., logits) - target * logits)
    return loss + .5 * np.sum(params**2), (gradient + params).ravel()


def fit_profiles(x, y, train, query, seed):
    start = np.random.default_rng(seed).normal(0, .10, (2 * PROFILES, x.shape[1]))
    fit = minimize(objective, start.ravel(), args=(x[train], y[train]),
                   method='L-BFGS-B', jac=True, options={'maxiter': 1200, 'ftol': 1e-6})
    assert fit.success, fit.message
    logits, planes = response(x[query], fit.x.reshape(start.shape))
    winners = planes.argmax(axis=2)
    return expit(logits), dict(
        iterations=int(fit.nit),
        winners=[np.bincount(winners[:, label], minlength=PROFILES).tolist() for label in range(2)])


def predict(frame, groups, y, train, query):
    assert not np.intersect1d(groups[train], groups[query]).size
    x = design(frame, train)
    linear = LogisticRegression(C=1, fit_intercept=False, max_iter=400).fit(x[train], y[train])
    estimates, diagnostics = zip(*(fit_profiles(x, y, train, query, s) for s in [611, 977, 1811]))
    return dict(linear=linear.predict_proba(x[query])[:, 1], competing=estimates[0],
                restart2=estimates[1], restart3=estimates[2], averaged=np.mean(estimates, axis=0)), dict(
                    fits=list(diagnostics))


def complementarity():
    row, y, frame, _, _ = load()
    agreed, direction, _, _, prior, _, model_count = panel(row, y, frame)
    seen = np.logical_or.reduce(list(prior.values()))
    missed = agreed & (direction != y)
    correct = agreed & (direction == y)
    observations, averaged = [], []
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART / ('nov23_' + LABEL) / f'{seed}.npz') as bank:
            for key, expected in [('row', row), ('y', y)]:
                np.testing.assert_array_equal(bank[key], expected)
            votes = np.stack([bank[k] >= .5 for k in ['competing', 'restart2', 'restart3']])
            observations.append(votes != direction)
            averaged.append((bank['averaged'] >= .5) != direction)
    observations = np.stack(observations)
    repeat = observations[:3, 0].all(0)
    mean_repeat = np.logical_and.reduce(averaged[:3])
    cohorts = {'all_shared_errors': missed, 'previously_correct': correct,
               'errors_without_prior_stable_dissent': missed & ~seen,
               'correct_without_prior_stable_dissent': correct & ~seen}
    result = {}
    for key, mask in cohorts.items():
        first_ever = mask & observations[:3, 0].any(0)
        # For the same passenger and split, did both additional initializations
        # reproduce the first model's change? Count distinct passengers once.
        all_initializations = observations[:3].all(1)
        result[key] = dict(size=int(mask.sum()), stable=int((mask & repeat).sum()),
                           mean_stable=int((mask & mean_repeat).sum()),
                           mean_stable_and_replayed=int((mask & mean_repeat & averaged[3]).sum()),
                           corroborated_in_replay=int((mask & repeat & observations[3, 0]).sum()),
                           first_ever=int(first_ever.sum()),
                           all_starts_on_one_split=int((mask & all_initializations.any(0)).sum()),
                           all_starts_on_every_split=int((mask & all_initializations.all(0)).sum()),
                           at_least_one_start_every_split=int((mask & observations[:3].any(1).all(0)).sum()))
    result['prior_recipes'] = int(model_count)
    write(REPORT / 'complementarity.json', result)


if __name__ == '__main__':
    with threadpool_limits(1):
        evaluate(LABEL, predict, REPORT)
        complementarity()
