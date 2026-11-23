"""Estimate how much actual within-passenger registration/bill pairings inform outcomes."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT
from exaone_validation_data import CATEGORICAL


def fitted_tree(train, target, categories):
    return HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, learning_rate=.08, l2_regularization=3.,
        categorical_features=categories, early_stopping=False, random_state=501,
    ).fit(train, target)


def score(train, target, query, categories):
    return fitted_tree(train, target, categories).decision_function(query)


def pair_features(context, bills):
    # Exact class-preserving shuffles leave each original column's marginal
    # unchanged. Expose interactions so a greedy tree can make its first split.
    amounts = np.column_stack([np.log1p(np.nan_to_num(bills)), np.isnan(bills)])
    return (context[:, :, None] * amounts[:, None, :]).reshape(len(bills), -1)


def fields(frame, levels):
    registration = frame.drop(columns=BILLS).copy()
    categories = registration.columns.isin(CATEGORICAL)
    for col in registration.columns[categories]:
        registration[col] = pd.Categorical(registration[col], categories=levels[col]).codes
    register = registration.to_numpy(dtype=float)
    bills = frame[BILLS].to_numpy(dtype=float)
    return register, bills, categories


def level_map(frame):
    return {col: sorted(frame[col].unique().tolist()) for col in CATEGORICAL}


def fit_fold(frame, groups, y, train, query):
    register, bills, categories = fields(frame, level_map(frame))
    joint = np.column_stack([register, bills])
    joint_categories = np.r_[categories, np.zeros(len(BILLS), dtype=bool)]
    context = pd.get_dummies(frame[CATEGORICAL], dtype=float).to_numpy()
    age = frame.Age.to_numpy()
    context = np.column_stack([context, age < 13, age >= 40, frame.BookingSize > 1])
    pairs = pair_features(context, bills)
    reg_score = score(register[train], y[train], register[query], categories)
    bill_score = score(bills[train], y[train], bills[query], None)
    independent = reg_score + bill_score - logit(y[train].mean())

    # Within each class preserve both block marginals exactly. Circular shifts
    # after permutation keep each real bill in each synthetic sample once.
    # Filtering cross-booking pairs would alter those marginals.
    rng = np.random.default_rng(813)
    ratios = []
    for outcome in [False, True]:
        ids = rng.permutation(train[y[train] == outcome])
        shifts = rng.choice(np.arange(1, len(ids)), 3, replace=False)
        synthetic = [pair_features(context[ids], bills[np.roll(ids, shift)])
                     for shift in shifts]
        labels = np.repeat([True, False], [len(ids), len(ids)*len(shifts)])
        # Equal-prior density log ratio, correcting the 1:3 sample ratio.
        ratios.append(score(np.vstack([pairs[ids], *synthetic]), labels,
                            pairs[query], None) + np.log(len(shifts)))
    linked = independent + ratios[1] - ratios[0]
    delta_sd = float(np.std(linked-independent))
    assert delta_sd > .001
    direct = score(joint[train], y[train], joint[query], joint_categories)
    return dict(independent=expit(independent), linked=expit(linked), direct=expit(direct)), {
        'training': len(train), 'query': len(query), 'synthetic_per_observed': 3,
        'pair_log_odds_shift_sd': delta_sd,
    }


if __name__ == '__main__':
    with threadpool_limits(2):
        evaluate('registration_bill_links', fit_fold, ROOT/'reports/nov23_registration_bill_links')
