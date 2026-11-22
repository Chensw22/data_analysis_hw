"""Whole-manifest logistic ANOVA with piecewise linear effects and interactions."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from error_overlap_diagnostic import load_authenticated
from exact_fold_error_selector import counts
from peer_bill_correction import BILLS
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ROOT / "reports/global_error_models_nov23"


def hat(values, knots):
    """Nonnegative adjacent-knot interpolation; missing values get their own column."""
    values = np.asarray(values, float)
    grid = np.asarray(knots, float)
    missing = ~np.isfinite(values)
    point = np.clip(np.where(missing, grid[0], values), grid[0], grid[-1])
    left = np.searchsorted(grid, point, side="right").clip(1, len(grid)-1) - 1
    right_weight = (point-grid[left]) / (grid[left+1]-grid[left])
    basis = np.zeros((len(point), len(grid)+1), dtype=np.float64)
    idx = np.arange(len(point))
    basis[idx, left] = 1-right_weight
    basis[idx, left+1] = right_weight
    basis[missing] = 0
    basis[missing,-1] = 1
    return basis


def design(roster):
    cabin = roster.Cabin.str.split("/", expand=True)
    group = roster.PassengerId.str.split("_").str[0]
    bills = roster[BILLS]
    favorable = bills[["ShoppingMall", "FoodCourt"]].sum(axis=1, min_count=2)
    other = bills[["Spa", "VRDeck", "RoomService"]].sum(axis=1, min_count=3)
    total = favorable + other
    spend_knots = np.log1p([0, 1, 100, 500, 850, 1400, 3000, 10000, 40000])
    category = {
        "planet":roster.HomePlanet, "destination":roster.Destination,
        "sleep":roster.CryoSleep, "vip":roster.VIP, "deck":cabin[0], "side":cabin[2]}
    blocks = {name:pd.get_dummies(s.fillna("missing").astype(str)).to_numpy(float)
              for name,s in category.items()}
    blocks["age"] = hat(roster.Age, [0,5,12,17,25,40,60,80])
    blocks["position"] = hat(pd.to_numeric(cabin[1]), [0,250,500,750,1000,1250,1500,2000])
    blocks["group_size"] = hat(group.map(group.value_counts()), [1,2,3,4,6,8])
    for name, s in [*bills.items(), ("total",total), ("favorable",favorable), ("other",other)]:
        blocks[name] = hat(np.log1p(s), spend_knots)
    blocks["composition"] = hat((favorable+10)/(total+20), [0,.1,.3,.5,.7,.9,1])
    singles = sparse.hstack([sparse.csr_matrix(v) for v in blocks.values()], format="csr")
    # A tensor basis permits a different slope or outcome surface for each feature pair.
    # Fixed scaling shrinks interaction terms more strongly than marginal effects.
    products = []
    for left, right in combinations(blocks, 2):
        cross = blocks[left][:,:,None] * blocks[right][:,None,:]
        products.append(sparse.csr_matrix(cross.reshape(len(roster), -1)))
    pairs = sparse.hstack([singles, .35*sparse.hstack(products)], format="csr")
    return {"marginal":singles, "pairwise":pairs}


def run():
    inp, _, folds, _, _ = load_authenticated()
    dev, y, group = inp["dev"], inp["y"].astype(bool), np.asarray(inp["groups"])
    cols = lambda c:c not in ["Transported","Name"]
    raw = pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=cols) for s in ["train","test"]],
                    ignore_index=True)
    anchor = pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for stored, expected in [(anchor.row,dev),(anchor.true,y),(anchor.fold,folds)]:
        np.testing.assert_array_equal(stored,expected)
    matrices = design(raw)
    outputs, results = {}, {}
    for name, matrix in matrices.items():
        x = matrix[dev]
        p = np.full(len(dev),np.nan)
        for k in np.unique(folds):
            q = folds==k
            t = ~q & ~np.isin(group,group[q])
            fit = LogisticRegression(C=1., solver="lbfgs", max_iter=400, tol=1e-5)
            fit.fit(x[t],y[t])
            p[q] = fit.predict_proba(x[q])[:,1]
        assert np.all(np.isfinite(p))
        outputs[name] = p
        results[name] = counts(p>=.5, anchor.v131.to_numpy(bool), y,folds,
                               anchor.hard.to_numpy(bool), anchor.persistent.to_numpy(bool))
        results[name]["columns"] = x.shape[1]
        print(name,results[name],flush=True)
    save = ART / "global_tensor_nov23.npz"
    np.savez_compressed(save,row=dev,fold=folds,y=y,p=outputs["pairwise"],**outputs)
    write(OUT/"tensor.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
