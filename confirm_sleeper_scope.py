"""Require an independent manifest outcome model to confirm a v127 change."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exaone_validation_data import compact_features
from exact_fold_error_selector import counts
from kaggle_workflow import sha256, validate_submission
from sleeper_consensus_scope import ART, ROOT, NAME as PARENT, cohort
from sleeper_manifest_risk import design
from stable_consensus_selector import persistent_mask
import export_consensus_routes as routes
import relational_error_corrector as ref

NAME = "proposal_v128_manifest_confirmation"


def confirm(original, scoped, predicted):
    return np.where((original != scoped) & (predicted == scoped), scoped, original)


def evaluate():
    inp, _, fold, bases, _ = ref.overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)
    rows = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id, dev)
    hard = rows.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    results = {}
    with np.load(ART / "sleeper_consensus_scope.npz") as scoped, np.load(ART / "sleeper_manifest_risk.npz") as witness:
        for z in [scoped, witness]:
            np.testing.assert_array_equal(z["dev"], dev)
            np.testing.assert_array_equal(z["fold"], fold)
        np.testing.assert_array_equal(scoped["base"], witness["base"])
        base = scoped["base"]
        for scope_split, witness_split in [("original","party_a"), ("new_party_partitions","party_b"),
                                         ("boarding_blocks","boarding_ranges")]:
            results[scope_split] = {}
            for approach in ["manifest", "manifest_berth"]:
                decision = confirm(base, scoped[scope_split], witness[approach + "_" + witness_split])
                results[scope_split][approach] = counts(decision, base, y, fold, hard, persistent)
    ref.write(ROOT / "reports/confirm_sleeper_scope/result.json", dict(results=results,
        interpretation="Each member excludes the assessed passenger from its own supervised fits. "
        "The three calibration/specialist split pairings reuse the same development labels."))
    print(results)


def export():
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Name")
    test = pd.read_csv(ROOT / "data/raw/test.csv", usecols=lambda c: c != "Name")
    x, q = compact_features(train), compact_features(test)
    t = cohort(x)
    fit = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
    fit.fit(design(x, True)[t], train.Transported.to_numpy(bool)[t])
    ids = test.PassengerId.to_numpy(str)
    base, source = routes.official_decisions(routes.BASE, ids)
    parent, parent_source = routes.official_decisions(PARENT, ids)
    out = ART / NAME
    out.mkdir(exist_ok=True)
    model = out / "model.joblib"
    joblib.dump(fit, model)
    p = fit.predict_proba(design(q, True))[:,1]
    restored = joblib.load(model).predict_proba(design(q, True))[:,1]
    np.testing.assert_array_equal(p, restored)
    decision, replay = confirm(base, parent, p >= .5), confirm(base, parent, restored >= .5)
    np.testing.assert_array_equal(decision, replay)
    pd.DataFrame(dict(PassengerId=ids, Transported=decision)).to_csv(out / "submission.csv", index=False)
    pd.DataFrame(dict(PassengerId=ids, probability=decision.astype(float),
                      reload_probability=replay.astype(float))).to_csv(out / "test_probabilities.csv", index=False)
    manifest = dict(**validate_submission(out / "submission.csv", ROOT / "data/raw/test.csv"), version="v128_manifest_confirmation",
        model_sha256=sha256(model), reload_difference=0, training_cohort=int(t.sum()), baseline_sha256=source,
        v127_sha256=parent_source, changed_vs_v108=int((decision != base).sum()), v127_changes=int((parent != base).sum()))
    for folder in [out, ROOT / "reports" / NAME]:
        ref.write(folder / "manifest.json", manifest)
        ref.write(folder / "selection.json", dict(version=manifest["version"], threshold=.5))
    print(manifest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["evaluate", "export"])
    (evaluate if parser.parse_args().stage == "evaluate" else export)()
