"""Predict outcome and billing state jointly for each withheld service on every passenger."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT
from exaone_validation_data import CATEGORICAL

LABEL = 'bill_completion_states'
REPORT = ROOT / 'reports/nov23_bill_completion_states'


def masked(frame, ids):
    parts = []
    for name in BILLS:
        view = frame.iloc[ids].copy()
        view[name] = np.nan
        view['WithheldService'] = name
        parts.append(view)
    return pd.concat(parts, ignore_index=True)


def fit_fold(frame, groups, y, train, query):
    assert not np.intersect1d(groups[train], groups[query]).size
    amounts = frame[BILLS].to_numpy(float)
    cuts = np.nanmedian(np.where(amounts[train]>0, amounts[train], np.nan), axis=0)
    state = (amounts>0).astype(int) + (amounts>cuts).astype(int)
    training = masked(frame, train)
    known = np.isfinite(amounts[train]).T.ravel()
    target = (3*y[train,None].astype(int) + state[train]).T.ravel()
    # Each passenger with observed services contributes one total unit of weight.
    observed = np.isfinite(amounts[train]).sum(1)
    weight = np.tile(1/np.maximum(observed,1),len(BILLS))[known]
    model = CatBoostClassifier(iterations=250, depth=6, learning_rate=.1, l2_leaf_reg=4,
                               random_seed=273, thread_count=2, verbose=False, allow_writing_files=False)
    model.fit(training.loc[known], target[known], sample_weight=weight*len(BILLS),
              cat_features=[*CATEGORICAL, 'WithheldService'])
    np.testing.assert_array_equal(model.classes_, np.arange(6))
    n = len(query)
    pred = model.predict_proba(masked(frame, query)).reshape(len(BILLS), n, 2, 3).transpose(1,0,2,3)
    state_prob = pred.sum(2)
    marginal = pred[:,:,1].sum(-1)
    conditional = pred[:,:,1] / state_prob
    known_outcome = np.take_along_axis(conditional, state[query,:,None], axis=2)[:,:,0]
    missing = np.isnan(amounts[query])
    mixed = np.where(missing, marginal, known_outcome)
    zero = conditional[:,:,0]
    paid = pred[:,:,1,1:].sum(-1) / state_prob[:,:,1:].sum(-1)
    count = missing.sum(1)
    def missing_mean(values):
        return np.where(count,(values*missing).sum(1)/np.maximum(count,1),mixed.mean(1))
    return dict(masked=marginal.mean(1), conditioned=mixed.mean(1),
                missing_marginal=missing_mean(marginal), missing_zero=missing_mean(zero),
                missing_paid=missing_mean(paid)), dict(train=len(train),query=n,missing=int((count>0).sum()))


if __name__ == '__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(2):
        evaluate(LABEL, fit_fold, REPORT)
