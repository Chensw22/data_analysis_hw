"""Check if held-out error proposals survive recalibration and independent corroboration."""
from pathlib import Path

import numpy as np
import pandas as pd

import relational_error_corrector as ref
from exact_fold_error_selector import cached, counts
from route_consensus_errors import cluster_interval
from stable_consensus_selector import persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def run():
    inp, _, folds, bases, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    with np.load(ART / "exact_fold_error_selector.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], folds)
        base = z["anchor"]
        votes = {view: np.stack([z[f"v108_{view}_{i}"] != base for i in [119, 220, 321]]).sum(axis=0)
                 for view in ["expert_and_manifest", "expert_evidence"]}
    geo = cached("prototype_error_specialist.npz", "uncalibrated_four", dev, folds) != base
    second = (cached("sequential_rescue_local.npz", "member", dev, folds)[:, :, 2].mean(axis=1) >= .5) != base
    manifest, models = votes["expert_and_manifest"], votes["expert_evidence"]
    # All proposals below use decisions formed without this passenger's label.
    rules = dict(manifest_majority=manifest >= 2, manifest_unanimous=manifest == 3,
                 model_unanimous=models == 3,
                 both_stable=(manifest == 3) & (models == 3),
                 manifest_and_prototype=(manifest == 3) & geo,
                 manifest_and_sequential=(manifest == 3) & second,
                 manifest_and_both_specialists=(manifest == 3) & geo & second)
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    out = {}
    for key, change in rules.items():
        pred = base ^ change
        row = counts(pred, base, y, folds, hard, persistent)
        row["party_bootstrap_pp"] = cluster_interval(change, base, y, inp["groups"])
        out[key] = row
    ref.write(ROOT / "reports/corroborate_exact_errors/result.json", dict(methods=out,
        design="Repeated group-held-out calibration predictions on the same adaptive development set; repeated "
               "calibrators share training data. Intersections are exploratory and their bootstrap intervals do not "
               "correct for choosing among rules. Fresh-party calibration for a full-data v108 remains untested."))
    np.savez_compressed(ART / "corroborate_exact_errors.npz", dev=dev, fold=folds, anchor=base,
                        **{key: base ^ value for key, value in rules.items()})
    print(out)


if __name__ == "__main__":
    run()
