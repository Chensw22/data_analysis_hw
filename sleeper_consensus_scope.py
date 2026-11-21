"""Evaluate the same previously scored correction inside and outside the zero-bill sleeper cohort."""
from pathlib import Path

import numpy as np
import pandas as pd

import export_consensus_routes as routes
import relational_error_corrector as ref
from exact_fold_error_selector import counts
from peer_bill_correction import BILLS
from route_consensus_errors import FAMILIES, cluster_interval
from stable_consensus_selector import persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
NAME = "proposal_v127_sleeper_scope"
CHILD = "proposal_v118_error_union"


def cohort(x):
    # All five zeroes must be observed. A missing bill gives no evidence of zero spending.
    return (x.HomePlanet.eq("Earth") & x.Deck.eq("G") & x.CryoSleep.eq("True") &
            x[BILLS].eq(0).all(axis=1)).to_numpy()


def apply(p, votes, child, x):
    eligible = p & votes.all(axis=1) & cohort(x)
    return np.where(eligible, child, p), eligible


def evaluate():
    inp, _, fold, models, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    x = inp["features"].iloc[dev]
    votes = np.column_stack([models[k] >= .5 for k in FAMILIES])
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(dev, z["dev_ids"])
        np.testing.assert_array_equal(fold, z["fold_id"])
        base = z["v108"] >= .5
    rows = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id, dev)
    np.testing.assert_array_equal(rows.y, y)
    hard = rows.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, models, hard)
    scores, decisions = {}, {}
    with np.load(ART / "replicate_error_committee.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], fold)
        np.testing.assert_array_equal(z["base"], base)
        for mode in ["original", "new_party_partitions", "boarding_blocks"]:
            child = z[mode + "_either"]
            pred, eligible = apply(base, votes, child, x)
            outside = np.where(eligible, base, child)
            score = counts(pred, base, y, fold, hard, persistent)
            score["party_bootstrap_pp"] = cluster_interval(pred != base, base, y, inp["groups"])
            scores[mode] = dict(inside=score, outside=counts(outside, base, y, fold, hard, persistent))
            decisions[mode] = pred
    common = np.logical_and.reduce([pred != base for pred in decisions.values()])
    result = dict(rows=len(y), eligible=int(eligible.sum()), eligible_errors=int((eligible & (base != y)).sum()),
                  eligible_hard=int((eligible & hard).sum()), eligible_persistent=int((eligible & persistent).sum()),
                  by_calibration=scores, stable_changes=counts(base ^ common, base, y, fold, hard, persistent),
                  provenance="Cohort chosen after inspecting earlier development errors. All local results reuse "
                             "the same labels. Original partition corresponds to the full v118 export.")
    ref.write(ROOT / "reports/sleeper_consensus_scope/result.json", result)
    np.savez_compressed(ART / "sleeper_consensus_scope.npz", dev=dev, fold=fold, base=base, eligible=eligible, **decisions)
    print(result, flush=True)


def export():
    import json
    from exaone_validation_data import compact_features
    from kaggle_workflow import sha256, validate_submission
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    ids, x = test.PassengerId.to_numpy(str), compact_features(test)
    sources = {}
    def labels(name):
        val, sources[name] = routes.official_decisions(name, ids)
        return val
    base, child = labels(routes.BASE), labels(CHILD)
    votes = np.column_stack([labels(name) for name in routes.HISTORICAL])
    pred, eligible = apply(base, votes, child, x)
    out = ART / NAME
    out.mkdir(exist_ok=True)
    ref.write(out / "router.json", dict(base=routes.BASE, child=CHILD, families=routes.HISTORICAL, sources=sources,
              mask="Earth, deck G, CryoSleep True, five observed zero bills, base and five families positive."))
    spec = json.loads((out / "router.json").read_text())
    for n, h in spec["sources"].items():
        assert sha256(ART / n / "submission.csv") == h
    second, _ = apply(labels(spec["base"]), np.column_stack([labels(n) for n in spec["families"]]),
                      labels(spec["child"]), compact_features(pd.read_csv(ROOT / "data/raw/test.csv")))
    np.testing.assert_array_equal(pred, second)
    pd.DataFrame(dict(PassengerId=ids, Transported=pred)).to_csv(out / "submission.csv", index=False)
    pd.DataFrame(dict(PassengerId=ids, probability=pred.astype(float), reload_probability=second.astype(float))).to_csv(out / "test_probabilities.csv", index=False)
    manifest = dict(**validate_submission(out / "submission.csv", ROOT / "data/raw/test.csv"), version="v127_sleeper_scope",
        model_sha256=sha256(out / "router.json"), reload_difference=0, changed_vs_v108=int((pred != base).sum()),
        eligible=int(eligible.sum()), baseline=routes.BASE, source_candidate=CHILD)
    for folder in [ROOT / "reports" / NAME, out]:
        ref.write(folder / "manifest.json", manifest)
        ref.write(folder / "selection.json", dict(version=manifest["version"], threshold=.5))
    print(manifest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["evaluate", "export"])
    (evaluate if parser.parse_args().stage == "evaluate" else export)()
