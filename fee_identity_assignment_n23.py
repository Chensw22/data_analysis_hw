"""Use the likelihood of assigning five observed amounts to their actual services."""
from math import factorial

import numpy as np
from scipy.special import expit, gammaln
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT
from registration_bill_links_n23 import fields, fitted_tree, level_map

LABEL = 'fee_identity_assignment'
REPORT = ROOT/'reports'/('nov23_'+LABEL)


def tables(frame, levels):
    register, fees, categories = fields(frame, levels)
    amount = np.where(np.isnan(fees), -1., np.log1p(np.nan_to_num(fees)))
    anonymous = np.column_stack([register, np.sort(amount, axis=1)])
    category_mask = np.r_[categories, np.zeros(amount.shape[1], bool)]
    return anonymous, amount, category_mask


def tokens(anonymous, amount, outcome):
    n, width = amount.shape
    smaller = (amount[:, :, None] > amount[:, None, :]).sum(2)
    outcome = np.broadcast_to(outcome, (n,))
    return np.column_stack([np.repeat(anonymous, width, axis=0), amount.ravel(), smaller.ravel(),
                            np.repeat(outcome, width)])


def observed_assignment(log_weights, amount):
    # Sum the weights of every bijection using subset DP. Column j is the service
    # assigned to one token, and each service must be used exactly once.
    n, width, columns = log_weights.shape
    assert width == columns
    mass = np.full((n, 1 << width), -np.inf)
    mass[:, 0] = 0
    for mask in range(1, 1 << width):
        token = mask.bit_count()-1
        terms = [mass[:, mask ^ (1 << j)] + log_weights[:, token, j] for j in range(width) if mask & (1 << j)]
        mass[:, mask] = np.logaddexp.reduce(terms)
    observed = np.trace(log_weights, axis1=1, axis2=2)-mass[:, -1]
    # Swapping identical amounts gives the same observation. Include each
    # tied block's factorial just once in its observable probability.
    equal = amount[:, :, None] == amount[:, None, :]
    copies = equal.sum(2)
    observed += (gammaln(copies+1)/copies).sum(1)
    assert np.all(observed <= 1e-8) and np.all(observed >= -1000)
    np.testing.assert_allclose(observed[(copies == width).all(1)], 0, atol=1e-8)
    return observed


def fit(anonymous, amount, categories, labels):
    width = amount.shape[1]
    prior = fitted_tree(anonymous, labels, categories)
    identities = HistGradientBoostingClassifier(
        max_iter=70, max_leaf_nodes=15, learning_rate=.08, l2_regularization=4.,
        categorical_features=np.r_[categories, False, False, False], early_stopping=False, random_state=681,
    ).fit(tokens(anonymous, amount, labels), np.tile(np.arange(width), len(labels)))
    return prior, identities


def predict(prior, identities, anonymous, amount):
    anchor = prior.decision_function(anonymous)
    width = amount.shape[1]
    likelihoods = []
    for outcome in [0, 1]:
        scores = identities.decision_function(tokens(anonymous, amount, outcome)).reshape(-1, width, width)
        likelihoods.append(observed_assignment(scores, amount))
    delta = likelihoods[1]-likelihoods[0]
    return dict(anonymous=expit(anchor), assigned=expit(anchor+delta)), delta


def fit_fold(frame, groups, y, train, query):
    anonymous, amount, categories = tables(frame, level_map(frame))
    models = fit(anonymous[train], amount[train], categories, y[train])
    results, delta = predict(*models, anonymous[query], amount[query])
    return results, dict(permutations=factorial(amount.shape[1]), mean_absolute_log_odds_shift=float(np.abs(delta).mean()))


if __name__ == '__main__':
    with threadpool_limits(2):
        evaluate(LABEL, fit_fold, REPORT)
