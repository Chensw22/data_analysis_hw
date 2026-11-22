"""Cross-fit probability corrections on disjoint bookings never used to fit the predictor."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from grouped_five_way_coverage_n23 import verify
from nested_manifest_expressions_n23 import BILLS
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def attributes(raw):
    deck = raw.Cabin.str.split("/").str[0].fillna("missing")
    bills = raw[BILLS]
    spending = np.select([bills.gt(0).any(axis=1), bills.isna().any(axis=1)],
                         ["positive", "incomplete"], default="zero")
    groups = pd.DataFrame(dict(deck=deck, spend=spending, joint=deck + "/" + spending))
    return pd.get_dummies(groups, dtype=float).to_numpy()


def offsets(x, z, y, t, q, penalty):
    def objective(w):
        score = z[t] + x[t] @ w
        value = np.logaddexp(0, score).sum() - y[t] @ score + .5 * (penalty*w) @ w
        grad = x[t].T @ (expit(score)-y[t]) + penalty*w
        return value, grad

    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                      options=dict(maxiter=250, ftol=1e-12))
    if not result.success:
        raise RuntimeError(result.message)
    return expit(z[q] + x[q] @ result.x)


def corrected(base, fold, groups, context, y):
    z = logit(base.clip(1e-6, 1-1e-6))
    # One logit unit is shared by every slope, avoiding extreme passenger
    # probabilities amplifying a correction fitted on a small calibration set.
    basis = np.column_stack([np.ones(len(y)), np.clip(z, -4, 4)])
    p = {key:np.full(len(y), np.nan) for key in ["score_only","context","placebo"]}
    rng = np.random.default_rng(113)
    for k in np.unique(fold):
        pool = np.flatnonzero(fold == k)
        assert not np.intersect1d(groups[pool], groups[fold != k]).size
        # Shuffle covariates together within the base fold, preserving context counts.
        placebo = context.copy()
        placebo[pool] = context[rng.permutation(pool)]
        for a, b in GroupKFold(3, shuffle=True, random_state=117).split(pool, groups=groups[pool]):
            t, q = pool[a], pool[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            for key, c in [("score_only", None), ("context", context), ("placebo", placebo)]:
                x = basis if c is None else np.column_stack([basis, c, c*basis[:, 1:2]])
                penalties = np.r_[np.repeat(2., 2), np.repeat(15., x.shape[1]-2)]
                p[key][q] = offsets(x, z, y, t, q, penalties)
    return p


def run():
    frame = pd.read_csv(ART / "nov23_global_error_inventory.csv")
    rows = frame.row.to_numpy()
    y, anchor_hits, challengers = read_panel()
    hard = ~anchor_hits.any(0) & ~challengers.any(axis=(0, 1))
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId","Cabin",*BILLS]).iloc[rows]
    groups = raw.PassengerId.str.split("_").str[0].to_numpy()
    context = attributes(raw)
    result = {}
    out = ART / "nov23_conditional_calibration"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as cache:
            fold, base = cache["fold"], cache["base"]
            verify(cache, rows, y, fold)
        p = corrected(base, fold, groups, context, y)
        before = (base >= .5) == y
        metrics = {}
        for name, score in {"base":base, **p}.items():
            assert np.isfinite(score).all()
            ok = (score >= .5) == y
            metrics[name] = dict(accuracy=float(ok.mean()), logloss=float(log_loss(y,score)),
                                 brier=float(np.mean((score-y)**2)), fixed=int((ok & ~before).sum()),
                                 harmed=int((~ok & before).sum()), missed_by_all466_correct=int(ok[hard].sum()),
                                 always_wrong1070_correct=int(ok[~anchor_hits.any(0)].sum()))
        result[seed] = metrics
        np.savez_compressed(out/f"{seed}.npz", row=rows, y=y, fold=fold, base=base, **p)
        print(seed, metrics, flush=True)
    write(ROOT / "reports/nov23_conditional_calibration/result.json", dict(results=result,
          validation="For each base fold, calibrate using two thirds of its held-out bookings, evaluate on the other third; rotate. All corrections share a probability offset and fixed ridge penalties."))


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
