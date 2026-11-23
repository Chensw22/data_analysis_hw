"""Fit a conditional latent-subclass classifier starting from passenger profile clusters."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from broad_local_profile_fit_n23 import coordinates
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT

LABEL = 'passenger_subclass_odds'
PROFILES = 12


def profile_scores(design, weights):
    scores = (design @ weights).reshape(len(design), 2, PROFILES)
    classes = logsumexp(scores, axis=2)
    return scores, classes[:, 1] - classes[:, 0]


def profile_loss(params, design, labels, initial, slopes):
    weights = initial.copy()
    delta = params.reshape(initial[slopes].shape)
    weights[slopes] += delta
    scores, odds = profile_scores(design, weights)
    error = expit(odds) - labels
    # Each class shares its gradient among its twelve current profile assignments.
    responsibilities = softmax(scores, axis=2) * error[:, None, None]
    responsibilities[:, 0, :] *= -1
    loss = (np.logaddexp(0., odds) - labels*odds).sum() + (delta**2).sum()
    gradient = design.T @ responsibilities.reshape(len(labels), -1)
    return loss, (gradient[slopes] + 2*delta).ravel()


def fit_fold(frame, groups, labels, train, query):
    x, _ = coordinates(frame, train)
    design = np.column_stack([x, np.ones(len(x))])
    initial_parts = []
    for label in [False, True]:
        support = x[train[labels[train] == label]]
        clusters = KMeans(n_clusters=PROFILES, n_init=3, random_state=591).fit(support)
        centers = clusters.cluster_centers_
        counts = np.bincount(clusters.labels_, minlength=PROFILES)
        # For squared distance, the common term in the query norm cancels in the odds.
        bias = -2*(centers**2).sum(1) + np.log(counts / len(train))
        initial_parts.append(np.vstack([4*centers.T, bias]))
    initial = np.column_stack(initial_parts)
    results, notes = {}, {}
    for name, slopes in [('fixed_profiles', slice(-1, None)), ('adaptive_profiles', slice(None))]:
        fit = minimize(profile_loss, np.zeros(initial[slopes].size),
                       args=(design[train], labels[train], initial, slopes),
                       jac=True, method='L-BFGS-B', options=dict(maxiter=250, ftol=1e-8))
        if not fit.success:
            raise RuntimeError(f'{name}: {fit.message}')
        weights = initial.copy()
        weights[slopes] += fit.x.reshape(initial[slopes].shape)
        results[name] = expit(profile_scores(design[query], weights)[1])
        notes[name] = dict(iterations=int(fit.nit), objective=float(fit.fun))
    return results, notes


if __name__ == '__main__':
    with threadpool_limits(1):
        evaluate(LABEL, fit_fold, ROOT/'reports'/('nov23_'+LABEL))
