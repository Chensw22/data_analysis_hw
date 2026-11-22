"""Estimate whole-roster class probabilities using independently labelled forest neighbors."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from awake_negative_recovery import matrices, BILLS
from error_overlap_diagnostic import load_authenticated
from exact_fold_error_selector import counts
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/nov23_honest_neighbors"


def predictors(x, groups):
    bills = x[BILLS].fillna(0)
    positive = bills.FoodCourt + bills.ShoppingMall
    negative = bills.RoomService + bills.Spa + bills.VRDeck
    total = positive + negative
    berth = x.CabinNumber
    party = pd.Series(groups)
    sizes = party.map(party.value_counts()).to_numpy()
    return np.column_stack([matrices(x)[0], np.log1p(positive), np.log1p(negative),
                            np.log1p(total), (positive-negative)/(1+total), berth.isna(),
                            berth.fillna(-1), sizes, bills.to_numpy()/(1+total.to_numpy()[:,None])])


def forest(seed):
    return RandomForestClassifier(n_estimators=160, max_features=.65, min_samples_leaf=5,
                                  bootstrap=True, n_jobs=2, random_state=seed)


def donor_posterior(model, x, y, donor, query):
    # Distinct markers for each tree. The inner product counts shared leaves.
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    donor_codes = encoder.fit_transform(model.apply(x[donor])).T.tocsr()
    query_codes = encoder.transform(model.apply(x[query]))
    probs = []
    for start in range(0, len(query), 128):
        affinity = (query_codes[start:start+128] @ donor_codes).toarray() / len(model.estimators_)
        # Cubic decay retains close supervised neighbors; a two-person prior stabilizes sparse queries.
        weights = affinity**3
        probs.extend((weights @ y[donor] + 2*y[donor].mean()) / (weights.sum(axis=1)+2))
    return np.asarray(probs)


def run():
    inp, _, fold, _, _ = load_authenticated()
    y, dev, groups = inp["y"].astype(bool), inp["dev"], inp["groups"]
    inc = pd.read_csv(ROOT / "artifacts/nov21_case_checks/v136_final_oof.csv")
    for truth, local in [(dev,inc.row),(fold,inc.fold),(y,inc.true)]:
        np.testing.assert_array_equal(truth, local)
    x = predictors(inp["features"].iloc[dev], groups)
    baseline, honest = np.full(len(y), np.nan), np.full(len(y), np.nan)
    diagnostics = []
    for k in np.unique(fold):
        query = np.flatnonzero(fold == k)
        eligible = np.flatnonzero((fold != k) & ~np.isin(groups, groups[query]))
        baseline[query] = forest(2311).fit(x[eligible], y[eligible]).predict_proba(x[query])[:,1]
        predictions = []
        for t, d in GroupKFold(3, shuffle=True, random_state=2311).split(eligible, groups=groups[eligible]):
            train, donor = eligible[t], eligible[d]
            assert not np.intersect1d(groups[train], groups[donor]).size
            assert not np.intersect1d(groups[eligible], groups[query]).size
            model = forest(2311).fit(x[train], y[train])
            predictions.append(donor_posterior(model, x, y, donor, query))
        honest[query] = np.mean(predictions, axis=0)
        diagnostics.append(dict(query=len(query),reference=len(eligible)))
    assert np.isfinite(honest).all() and np.isfinite(baseline).all()
    incumbent = inc.v131.to_numpy(bool)
    results = {}
    for name,p in [("ordinary_forest",baseline),("honest_forest_metric",honest)]:
        results[name] = counts(p>=.5,incumbent,y,fold,inc.hard.to_numpy(bool),inc.persistent.to_numpy(bool))
        results[name]["fold_accuracy"] = [float(((p[fold==k]>=.5)==y[fold==k]).mean()) for k in range(3)]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/"oof.npz",row=dev,fold=fold,y=y,p=honest,p_forest=baseline)
    write(ROOT/"reports/nov23_honest_neighbors/results.json",dict(assessed=len(y),folds=diagnostics,
        results=results,exclusive_vs_ordinary=int(((incumbent!=y) & ((honest>=.5)==y) & ((baseline>=.5)!=y)).sum()),
        notes="Original evaluation folds; whole evaluation bookings excluded from both reference partitions. "
              "Group size is computed from the unlabeled development subset. All decisions use threshold 0.5."))
    print(results, flush=True)


if __name__ == "__main__":
    run()
