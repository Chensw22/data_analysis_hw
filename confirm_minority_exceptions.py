"""Require independent support to overrule a minority anchor on disputed passengers."""
from pathlib import Path

import numpy as np
import pandas as pd

import relational_error_corrector as ref
from route_consensus_errors import FAMILIES, cluster_interval

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def run():
    inp, splits, folds, models, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"]
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(dev, z["dev_ids"])
        np.testing.assert_array_equal(folds, z["fold_id"])
        anchor = z["v108"] >= .5
    with np.load(ART / "composed_local_error_routes.npz") as z:
        np.testing.assert_array_equal(dev, z["dev"])
        np.testing.assert_array_equal(folds, z["fold"])
        base = z["base"]
        proposed = z["five_family_majority_with_specialist"]
    with np.load(ART / "sequential_rescue_local.npz") as z:
        np.testing.assert_array_equal(dev, z["dev"])
        np.testing.assert_array_equal(folds, z["folds"])
        first, _, second = (z["member"].mean(axis=1) >= .5).T
    alternate = np.all(np.column_stack([models[k] >= .5 for k in FAMILIES]) != anchor[:, None], axis=1)
    assert alternate.sum() == 43 and np.all(proposed[alternate] != base[alternate])
    cases = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cases.row_id, dev)
    np.testing.assert_array_equal(cases.y, y)
    hard = cases.all_recipes_still_wrong.to_numpy(bool)
    methods = {"follow_vote": np.ones(len(y), bool), "first_confirms": first == proposed,
               "second_confirms": second == proposed, "both_confirm": (first == proposed) & (second == proposed),
               "retain_exception": np.zeros(len(y), bool)}
    results, decisions = {}, {}
    for name, approve in methods.items():
        pred = np.where(alternate & ~approve, base, proposed)
        decisions[name] = pred
        fixed = (pred == y) & (base != y)
        harmed = (pred != y) & (base == y)
        results[name] = dict(accuracy=float(np.mean(pred == y)), correct=int((pred == y).sum()),
            fixed=int(fixed.sum()), harmed=int(harmed.sum()), hard_fixed=int((fixed & hard).sum()),
            hard_harmed=int((harmed & hard).sum()), subgroup_correct=int((pred[alternate] == y[alternate]).sum()),
            subgroup_rows=int(alternate.sum()), fold_net=[int(fixed[b].sum()) - int(harmed[b].sum()) for _, b in splits],
            party_bootstrap_pp=cluster_interval(pred != base, base, y, inp["groups"]))
    ref.write(ROOT / "reports/confirm_minority_exceptions/result.json", results)
    np.savez_compressed(ART / "confirm_minority_exceptions.npz", dev=dev, fold=folds, base=base, **decisions)
    print(results)


if __name__ == "__main__":
    run()
