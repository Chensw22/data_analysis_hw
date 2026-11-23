"""Use score changes with training size to identify errors on an unseen booking."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, historical, write
from matched_booking_learning_curve_n23 import portions
from near_full_information_attribution_n23 import paired_intervals

FAMILIES = ("v44", "tabicl", "native", "target_mix")
REPORT = ROOT / "reports/nov23_learning_size_gate"


def fit_offset(features, offset, y, train, query):
    # The reference decision is the prior. Standardized coefficients shrink to zero.
    mean = features[train].mean(0)
    scale = np.maximum(features[train].std(0), 0.05)
    x = np.c_[np.ones(len(features)), (features - mean) / scale]
    def objective(b):
        p = offset[train] + x[train] @ b
        loss = (np.logaddexp(0, p) - y[train] * p).sum()
        return loss + 16 * b @ b, x[train].T @ (expit(p) - y[train]) + 32 * b

    m = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not m.success:
        raise RuntimeError(m.message)
    return expit(offset[query] + x[query] @ m.x)


def run():
    rows, y, _, groups, _ = load()
    with np.load(ART / "nov23_near_full_distillation/predictions.npz") as z:
        np.testing.assert_array_equal(rows, z["row"])
        np.testing.assert_array_equal(y, z["y"])
        fold = z["fold"]
        sizes = z["sizes"]
        reference = z["reference"]
        small, large = np.stack([z[k] for k in FAMILIES], axis=2)
    np.testing.assert_allclose(reference, np.stack([small[:, :2].mean(1), large[:, :2].mean(1)]))
    base = reference[1]
    # Predicting a calibration correction in log-odds keeps scores inside [0, 1].
    odds = logit(base.clip(.001, .999))
    current = logit(large.clip(.001, .999))
    movement = current - logit(small.clip(.001, .999))
    designs = dict(endpoint=current, trajectory=np.c_[current, movement])
    probs = {name: np.full(len(y), np.nan) for name in designs}
    probs["reference"] = base
    probs["midpoint"] = reference.mean(0)
    probs["extrapolate"] = expit(odds + (odds - logit(reference[0].clip(.001, .999))))
    for f, s, t, ids in portions(groups):
        np.testing.assert_array_equal(ids, np.flatnonzero(fold == f))
        np.testing.assert_array_equal(sizes[f], [len(s), len(t), len(ids)])
        split = GroupKFold(3, shuffle=True, random_state=127)
        for a, b in split.split(ids, groups=groups[ids]):
            train, query = ids[a], ids[b]
            assert not np.intersect1d(groups[train], groups[query]).size
            for name, features in designs.items():
                probs[name][query] = fit_offset(features, odds, y, train, query)

    # These groups depend on historical outcomes and enter evaluation only.
    old_rows, old_y, history = historical()
    np.testing.assert_array_equal(old_rows, rows)
    np.testing.assert_array_equal(old_y, y)
    masks = dict(all=np.ones(len(y), bool), historically_always_correct=~history["ever"],
                 historically_mixed=history["ever"] & ~history["repeat"], historically_wrong=history["repeat"])
    np.testing.assert_array_equal(sum(m.astype(int) for m in list(masks.values())[1:]), np.ones(len(y)))
    correct = {k: (v >= .5) == y for k, v in probs.items()}
    results = {}
    for k, h in correct.items():
        assert np.isfinite(probs[k]).all()
        before = correct["reference"]
        results[k] = dict(accuracy=float(h.mean()), brier=float(np.square(probs[k] - y).mean()),
                          coverage={name: dict(n=int(m.sum()), corrected=int((m & h & ~before).sum()),
                                               harmed=int((m & ~h & before).sum()), right=int((m & h).sum()))
                                    for name, m in masks.items()})
    diff = correct["trajectory"].astype(float) - correct["endpoint"]
    effect = paired_intervals(dict(history_over_endpoint=diff), groups, masks)
    trajectory = correct["trajectory"]
    endpoint = correct["endpoint"]
    details = {name:dict(fixed=int((m & trajectory & ~endpoint).sum()), harmed=int((m & ~trajectory & endpoint).sum()))
               for name,m in masks.items()}
    write(REPORT / "result.json", dict(models=results, trajectory_effect=effect, trajectory_changes=details,
                                      fold_nets=[int(diff[fold == f].sum()) for f in np.unique(fold)]))
    print({k: (v["accuracy"], v["coverage"]["all"]) for k,v in results.items()}, flush=True)
    print(details, effect["history_over_endpoint"]["all"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
