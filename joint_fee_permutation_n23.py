"""Learn outcome-specific joint service allocations against uniformly permuted whole baskets."""
from itertools import permutations

import numpy as np
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

from broad_opposing_passengers_n23 import evaluate
from fee_identity_assignment_n23 import REPORT, tables
from registration_bill_links_n23 import fitted_tree, level_map

LABEL = 'joint_fee_permutation'
ORDERS = np.array(list(permutations(range(5))))
assert (ORDERS[0] == np.arange(5)).all()


def fit_density(anonymous, amount, categories, rng):
    # Constant baskets have just one observable allocation and no contrast.
    informative = np.ptp(amount, axis=1) > 0
    context, fees = anonymous[informative], amount[informative]
    samples = [fees]
    for _ in range(4):
        order = ORDERS[rng.integers(len(ORDERS), size=len(fees))]
        samples.append(np.take_along_axis(fees, order, axis=1))
    x = np.vstack([np.column_stack([context, sample]) for sample in samples])
    real = np.repeat([True, False], [len(fees), 4*len(fees)])
    return fitted_tree(x, real, np.r_[categories, np.zeros(5, bool)])


def density(tree, anonymous, amount):
    score = np.zeros(len(amount))
    informative = np.flatnonzero(np.ptp(amount, axis=1) > 0)
    for start in range(0, len(informative), 64):
        ids = informative[start:start+64]
        orderings = amount[ids][:, ORDERS].reshape(-1, 5)
        x = np.column_stack([np.repeat(anonymous[ids], len(ORDERS), axis=0), orderings])
        utilities = tree.decision_function(x).reshape(len(ids), len(ORDERS))
        # Ratio to a uniform assignment conditional on the same anonymous basket.
        # The count of duplicate orderings cancels in this density ratio.
        score[ids] = np.log(len(ORDERS)) + utilities[:, 0] - logsumexp(utilities, axis=1)
    return score


def fit_fold(frame, groups, y, train, query):
    anonymous, amount, categories = tables(frame, level_map(frame))
    prior = fitted_tree(anonymous[train], y[train], categories)
    anchor = prior.decision_function(anonymous[query])
    ratios = []
    rng = np.random.default_rng(291)
    for label in [False, True]:
        t = train[y[train] == label]
        tree = fit_density(anonymous[t], amount[t], categories, rng)
        ratios.append(density(tree, anonymous[query], amount[query]))
    delta = ratios[1]-ratios[0]
    return dict(anonymous=expit(anchor), joint=expit(anchor+delta)), dict(
        mean_absolute_shift=float(np.abs(delta).mean()), permuted_training_baskets_per_real=4,
        permutations_at_prediction=len(ORDERS))


if __name__ == '__main__':
    with threadpool_limits(2):
        evaluate(LABEL, fit_fold, REPORT)
