"""Estimate feature effects shared by four booking/spending domains plus a rank-one deviation."""
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import coordinates
from broad_opposing_passengers_n23 import evaluate
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts

LABEL = 'booking_spend_multitask'
CONTRAST = helmert(4).T


def feature_map(frame, train):
    raw, _ = coordinates(frame, train)
    # Approximate a nonlinear kernel alongside the original fields; use the same
    # representation in all three models, fitted without validation labels.
    rng = np.random.default_rng(907)
    left = rng.choice(train, 2048)
    right = rng.choice(train, 2048)
    scale = np.sqrt(np.median(np.square(raw[left]-raw[right]).sum(1)))
    freq = rng.normal(size=(raw.shape[1], 256))/max(scale, .01)*2
    phases = rng.uniform(0, 2*np.pi, len(freq.T))
    kernel = np.sqrt(2/len(phases))*np.cos(raw @ freq + phases)
    return np.column_stack([raw, kernel])


def rank_loss(params, x, labels, domain, penalize_matrix=False):
    width = x.shape[1]
    shared, axis = params[:width], params[width:2*width]
    mixing, bias = params[2*width:-4], params[-4:]
    shift = CONTRAST @ mixing
    projection = x @ axis
    odds = x @ shared + bias[domain] + projection*shift[domain]
    error = expit(odds)-labels
    cost = (np.logaddexp(0., odds)-labels*odds).sum() + .5*np.dot(params[:-4], params[:-4])
    grad = np.r_[x.T@error + shared, x.T@(error*shift[domain]) + axis,
                 CONTRAST.T@np.bincount(domain, weights=error*projection, minlength=4) + mixing,
                 np.bincount(domain, weights=error, minlength=4)]
    if penalize_matrix:
        # Apply the ridge directly to the coefficient deviations too. The factor
        # penalty by itself grows only linearly with their singular value.
        axis_norm, mixing_norm = axis@axis, mixing@mixing
        cost += .5*axis_norm*mixing_norm
        grad[width:2*width] += mixing_norm*axis
        grad[2*width:-4] += axis_norm*mixing
    return cost, grad


def domains(frame):
    solo = frame.BookingSize.to_numpy() == 1
    paid = frame[BILLS].gt(0).any(axis=1).to_numpy()
    return 2*solo.astype(int) + paid.astype(int)


def fit_fold(frame, groups, y, train, query):
    domain = domains(frame)
    x = feature_map(frame, train)
    # A global linear part plus unshared intercepts permits different prevalence in all four domains.
    augmented = np.column_stack([x, np.eye(4)[domain]])
    model = LogisticRegression(C=1., max_iter=400, tol=1e-7).fit(augmented[train], y[train])
    pooled = model.decision_function(augmented[query])
    independent_weights, intercepts = [], []
    for d in range(4):
        ids = train[domain[train] == d]
        separate = LogisticRegression(C=1., max_iter=400, tol=1e-7).fit(x[ids], y[ids])
        independent_weights.append(separate.coef_[0])
        intercepts.append(separate.intercept_[0])
    by_domain = np.column_stack(independent_weights)
    bias = np.array(intercepts)
    ind = np.einsum('ij,ji->i', x[query], by_domain[:, domain[query]]) + bias[domain[query]]
    u, s, vh = np.linalg.svd(by_domain @ CONTRAST, full_matrices=False)
    width = x.shape[1]
    start = np.r_[by_domain.mean(1), u[:, 0]*np.sqrt(s[0]), vh[0]*np.sqrt(s[0]), bias]
    probabilities = dict(pooled=expit(pooled), independent=expit(ind))
    notes = dict(train_per_domain=np.bincount(domain[train], minlength=4).tolist())
    for key, matrix_penalty in [('rank_one', False), ('rank_one_ridge', True)]:
        fit = minimize(rank_loss, start, args=(x[train], y[train], domain[train], matrix_penalty), jac=True,
                       method='L-BFGS-B', options=dict(maxiter=2000, ftol=1e-7, maxcor=20))
        if not fit.success:
            raise RuntimeError(fit.message)
        w = fit.x
        adjustments = CONTRAST @ w[2*width:-4]
        odds = x[query]@w[:width] + w[-4:][domain[query]] + (x[query]@w[width:2*width])*adjustments[domain[query]]
        probabilities[key] = expit(odds)
        notes[key] = dict(iterations=int(fit.nit),
                          deviation_singular_value=float(np.linalg.norm(w[width:2*width])*np.linalg.norm(adjustments)))
    return probabilities, notes


def summarize(report):
    rows, y, frame, _, _ = load()
    _, _, hard, right = hard_cohorts(rows, y)
    domain = domains(frame)
    names = ['multi_no_positive', 'multi_positive', 'solo_no_positive', 'solo_positive']
    plans, history = {}, {'rank_one': [], 'rank_one_ridge': []}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_{LABEL}/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'], rows)
            np.testing.assert_array_equal(z['y'], y)
            reference = z['ref'] == y
            plans[seed] = {}
            for model, parts in history.items():
                hit = (z[model] >= .5) == y
                plans[seed][model] = {n: paired(hit[domain == d], reference[domain == d]) for d, n in enumerate(names)}
                if seed != 'v131':
                    parts.append(hit)
    stable = {}
    for model, parts in history.items():
        hit = np.stack(parts)
        stable[model] = {}
        for d, n in enumerate(names):
            scope = domain == d
            stable[model][n] = dict(population=int(scope.sum()), hard=int((hard & scope).sum()),
                                    hard_ever=int((hard & scope & hit.any(0)).sum()),
                                    hard_stable=int((hard & scope & hit.all(0)).sum()),
                                    right_stably_harmed=int((right & scope & ~hit.any(0)).sum()))
    write(report/'domains.json', dict(plans=plans, stable=stable))


if __name__ == '__main__':
    with threadpool_limits(1):
        report = ROOT/'reports'/('nov23_'+LABEL)
        evaluate(LABEL, fit_fold, report)
        summarize(report)
