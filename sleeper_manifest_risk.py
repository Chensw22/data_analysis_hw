"""Predict the outcome of zero-bill Earth sleepers from berth and manifest attributes alone."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref
from exact_fold_error_selector import counts
from route_consensus_errors import FAMILIES
from sleeper_consensus_scope import cohort
from stable_consensus_selector import persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def design(frame, expanded):
    # Fixed units and knots; the development and test designs share one scale.
    port = frame.Side.eq("P").to_numpy(float)
    side_missing = frame.Side.eq("__MISSING__").to_numpy(float)
    age = frame.Age.fillna(30).to_numpy(float)
    dest = np.column_stack([frame.Destination.eq(d).to_numpy(float)
                           for d in ["55 Cancri e", "PSO J318.5-22", "__MISSING__"]])
    basic = np.column_stack([port, side_missing, dest, age/40, age < 13, frame.Age.isna()])
    if not expanded:
        return basic
    berth = frame.CabinNumber.fillna(900).to_numpy(float)/1000
    pos = np.column_stack([berth, np.maximum(berth-.6, 0), np.maximum(berth-1.2, 0)])
    return np.column_stack([basic, pos, pos*port[:, None], (age / 40)*port, dest*port[:, None]])


def run():
    inp, _, folds, models, _ = ref.overlap.load_authenticated()
    dev, groups, y = inp["dev"], np.asarray(inp["groups"]), inp["y"].astype(bool)
    x = inp["features"].iloc[dev]
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], folds)
        base = z["v108"] >= .5
    domain = cohort(x)
    eligible = domain & base & np.column_stack([models[k] >= .5 for k in FAMILIES]).all(axis=1)
    hard = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(hard.row_id, dev)
    hard = hard.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, models, hard)
    ids = np.flatnonzero(domain)
    order, g = np.unique(groups[ids].astype(int), return_inverse=True)
    section = np.minimum(8, 9*g//len(order))
    directions = {"party_a": list(GroupKFold(3, shuffle=True, random_state=1121).split(ids, groups=groups[ids])),
                  "party_b": list(GroupKFold(3, shuffle=True, random_state=2121).split(ids, groups=groups[ids])),
                  "boarding_ranges": [(np.flatnonzero(((section-start)%9)>=3), np.flatnonzero(((section-start)%9)<3))
                                      for start in [0,3,6]]}
    result, predictions = {}, {}
    for name, large in [("manifest", False), ("manifest_berth", True)]:
        matrix = design(x, large)
        for part, splits in directions.items():
            pred = base.copy()
            seen = np.zeros(len(y), int)
            for t, q in splits:
                t, q = ids[t], ids[q]
                assert not np.intersect1d(groups[t], groups[q]).size
                # Outcome training uses every passenger in the observable cohort,
                # regardless of how the frozen base predicted them. Reference scores
                # and the historical error catalog never enter this fit.
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
                model.fit(matrix[t], y[t])
                q = q[eligible[q]]
                pred[q] = model.predict_proba(matrix[q])[:,1] >= .5
                seen[q] += 1
            assert (seen[eligible] == 1).all() and not seen[~eligible].any()
            key = name + "_" + part
            result[key] = counts(pred, base, y, folds, hard, persistent)
            predictions[key] = pred
    ref.write(ROOT / "reports/sleeper_manifest_risk/result.json",
              dict(domain=int(domain.sum()), eligible=int(eligible.sum()), results=result,
                   scope="Alternative trained from outcome labels and passenger attributes only on development rows. "
                         "Fit on group-disjoint parties, limit proposed changes to the selected consensus-positive "
                         "cohort. Fold net figures use the original reference folds across all validation designs."))
    np.savez_compressed(ART / "sleeper_manifest_risk.npz", dev=dev, fold=folds, base=base, **predictions)
    print({k:(v["fixed"],v["harmed"],v["persistent_fixed"]) for k,v in result.items()})


if __name__ == "__main__":
    run()
