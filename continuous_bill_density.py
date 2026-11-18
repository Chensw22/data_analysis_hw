"""Cross-fit class-conditional continuous spend densities within exact service-use patterns."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

import relational_error_corrector as ref
from observation_collision_audit import BILLS, metrics

ROOT = Path(__file__).resolve().parent
NAMES = ["joint", "independent", "dependence_only"]


def likelihood(bills, pattern, y, train, query):
    result = np.zeros((len(query), len(NAMES)))
    eligible = np.zeros(len(query), bool)
    for key in np.unique(pattern[query]):
        if key <= 0:
            continue
        support = train[pattern[train] == key]
        if min((~y[support]).sum(), y[support].sum()) < 8:
            continue
        target = np.flatnonzero(pattern[query] == key)
        cols = np.flatnonzero(key & (1 << np.arange(len(BILLS))))
        x = bills[np.ix_(support, cols)]
        z = bills[np.ix_(query[target], cols)]
        # Scott's multivariate bandwidth on the pooled standardized support.
        width = np.maximum(x.std(axis=0), .05) * len(x) ** (-1 / (4 + len(cols)))
        kernel = -.5 * ((z[:, None, :] - x[None, :, :]) / width) ** 2
        scores = []
        for label in [False, True]:
            k = kernel[:, y[support] == label, :]
            n = np.log(k.shape[1])
            scores.append(np.column_stack([logsumexp(k.sum(axis=2), axis=1) - n,
                                           (logsumexp(k, axis=1) - n).sum(axis=1)]))
        ratios = scores[1] - scores[0]
        result[target] = np.clip(np.column_stack([ratios,ratios[:,0]-ratios[:,1]]), -8, 8)
        eligible[target] = True
    return result, eligible


def calibrate(signal, labels, teacher):
    # With no density signal, preserve the original teacher exactly.
    offset = ref.logit(teacher)
    def objective(beta):
        eta = offset + beta * signal
        return np.mean(np.logaddexp(0,eta) - labels*eta) + .01*beta**2
    est = minimize_scalar(objective, bounds=(-2, 2), method="bounded")
    if not est.success:
        raise RuntimeError(est.message)
    return float(est.x)


def run():
    inp, splits, folds, _, evidence = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    with np.load(ROOT / "artifacts/v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(dev,z["dev_ids"])
        np.testing.assert_array_equal(folds,z["fold_id"])
        p = z["v108"]
    with np.load(ROOT / "artifacts/consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(dev,z["dev"])
        routed = z["combined_by_historical_agreement"]
    base = (p >= .5) ^ routed
    catalog = pd.read_csv(ROOT / "artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,catalog.row_id)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    bills = inp["features"].iloc[dev][BILLS].to_numpy(float)
    pattern = (bills > 0) @ (1 << np.arange(len(BILLS)))
    pattern[~np.isfinite(bills).all(axis=1)] = -1
    bills = np.log1p(bills)
    decision = np.tile(base[:,None],(1,len(NAMES)))
    support_stats = []
    for f, (a, b) in enumerate(splits):
        teacher = ref.load_inner(f,a,b,dev,y,evidence)
        cross = np.zeros((len(a),len(NAMES)))
        inner = StratifiedKFold(3,shuffle=True,random_state=20261117+f)
        for train,query in inner.split(a,y[a]):
            cross[query], _ = likelihood(bills,pattern,y,a[train],a[query])
        query_signal, supported = likelihood(bills,pattern,y,a,b)
        betas = [calibrate(cross[:,j],y[a],teacher) for j in range(len(NAMES))]
        eligible = supported & ~routed[b]
        changed = (ref.logit(p[b])[:,None] + query_signal * betas) >= 0
        decision[b[eligible]] = changed[eligible]
        support_stats.append(dict(eligible=int(eligible.sum()), protected_v115=int(routed[b].sum()),
                                  beta=dict(zip(NAMES,betas))))
    result = dict(fold_support=support_stats, baseline_correct=int((base==y).sum()), rows=len(y),
                  result={m:metrics(decision[:,j],base,y,folds,hard) for j,m in enumerate(NAMES)})
    out = ROOT / "reports/continuous_bill_density/result.json"
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result)


if __name__ == "__main__":
    run()
