"""Learn joint outcomes for pairs from separate bookings and condition on labeled reference passengers."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from broad_local_profile_fit_n23 import coordinates
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT

REPORT = ROOT / 'reports/nov23_reference_passenger_outcomes'


def partners(x, groups, targets, pool, rng, count):
    nearest = pool[cKDTree(x[pool]).query(x[targets], k=40)[1]]
    valid = groups[nearest] != groups[targets, None]
    # Sample within nearby passengers; half the partners come from arbitrary other bookings.
    keys = np.where(valid, rng.random(nearest.shape), 2.)
    local = np.take_along_axis(nearest, np.argsort(keys, axis=1)[:, :count], axis=1)
    other = rng.choice(pool, (len(targets), count))
    invalid = groups[other] == groups[targets, None]
    while invalid.any():
        other[invalid] = rng.choice(pool, invalid.sum())
        invalid = groups[other] == groups[targets, None]
    result = np.column_stack([local, other])
    assert np.all(groups[result] != groups[targets, None])
    return result


def compare(x, targets, partners):
    target = np.repeat(x[targets], partners.shape[1], axis=0)
    donor = x[partners.ravel()]
    return np.column_stack((target, donor, np.abs(target - donor))).astype(np.float32)


def run_fold(frame, groups, y, train, query):
    fit_ids, ref_ids = next(GroupShuffleSplit(n_splits=1, test_size=.22, random_state=291).split(train, groups=groups[train]))
    fit, ref = train[fit_ids], train[ref_ids]
    assert not np.intersect1d(groups[fit], groups[ref]).size
    assert not np.intersect1d(groups[train], groups[query]).size
    x, _ = coordinates(frame, fit)
    rng = np.random.default_rng(318)
    train_peers = partners(x, groups, fit, fit, rng, 3)
    joint = 2 * y[fit, None].astype(int) + y[train_peers]
    model = CatBoostClassifier(iterations=150, depth=5, learning_rate=.12, l2_leaf_reg=5,
                               random_seed=381, thread_count=2, verbose=False, allow_writing_files=False)
    model.fit(compare(x, fit, train_peers), joint.ravel())
    np.testing.assert_array_equal(model.classes_, [0, 1, 2, 3])
    refs = partners(x, groups, query, ref, rng, 8)
    pred = model.predict_proba(compare(x, query, refs)).reshape(len(query), refs.shape[1], 2, 2)
    marginal = pred[:, :, 1].sum(-1)
    conditional = pred[:, :, 1] / pred.sum(2)
    def select(labels):
        return np.take_along_axis(conditional, labels[:, :, None].astype(int), axis=2)[:, :, 0]
    known = select(y[refs])
    shuffled = y.copy()
    shuffled[ref] = rng.permutation(y[ref])
    control = select(shuffled[refs])
    n_near = refs.shape[1] // 2
    return dict(marginal=marginal.mean(1), conditional=known.mean(1), permuted=control.mean(1),
                near=known[:, :n_near].mean(1), arbitrary=known[:, n_near:].mean(1)), dict(
                    paired_training_people=len(fit), separate_reference_people=len(ref), predicted_people=len(query))


if __name__ == '__main__':
    REPORT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(2):
        evaluate('reference_passenger_outcomes', run_fold, REPORT)
