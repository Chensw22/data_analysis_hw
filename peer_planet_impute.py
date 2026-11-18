"""Apply a leave-one-passenger-out HomePlanet consensus to missing test attributes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"
import numpy as np
import pandas as pd

import kaggle_workflow as workflow

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/proposal_online_first_counts15"
PROPOSAL = "proposal_nov18_peer_planet_impute"
VERSION = "v105_v96_peer_planet"
REPORT = ROOT / "reports" / PROPOSAL
OUT = ROOT / "artifacts" / PROPOSAL


def consensus():
    roster = pd.concat([pd.read_csv(ROOT/"data/raw"/file, usecols=["PassengerId", "HomePlanet"])
                        for file in ["train.csv", "test.csv"]], ignore_index=True)
    groups = roster.PassengerId.str.split("_").str[0]
    answers = pd.Series(index=roster.index, dtype="object")
    for indices in roster.groupby(groups).indices.values():
        if len(indices) < 2:
            continue
        known = roster.HomePlanet.iloc[indices]
        for idx in indices:
            peers = known.drop(index=idx).dropna().unique()
            if len(peers) == 1:
                answers.at[idx] = peers[0]
    known = roster.HomePlanet.notna() & answers.notna()
    missing = roster.HomePlanet.isna() & answers.notna()
    query = missing & (roster.index >= 8693)
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    hard = catalog.loc[catalog.all_recipes_still_wrong, "row_id"]
    stats = dict(masked_field_predictions=int(known.sum()),
                 masked_field_correct=int((roster.HomePlanet[known] == answers[known]).sum()),
                 test_missing=int(roster.HomePlanet.iloc[8693:].isna().sum()),
                 test_fillable=int(query.sum()), hard546_fillable=int(missing.iloc[hard].sum()))
    return roster, answers[query], stats


def export(roster, replacements, stats):
    import joblib
    import torch
    import mitra_counts_fixed15_online_first as counts

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    manifest = json.loads((SOURCE/"manifest.json").read_text())
    assert workflow.sha256(SOURCE/"model.joblib") == manifest["model_sha256"]
    for path, expected in manifest["source_hashes"].items():
        assert workflow.sha256(ROOT/path) == expected, path
    probs = pd.read_csv(SOURCE/"test_probabilities.csv", dtype={"probability":"float32","reload_probability":"float32"})
    np.testing.assert_array_equal(probs.PassengerId, roster.PassengerId.iloc[8693:])
    np.testing.assert_array_equal(probs.probability, probs.reload_probability)
    assert workflow.sha256(SOURCE/"submission.csv") == manifest["sha256"]
    np.testing.assert_array_equal(probs.probability >= .5, pd.read_csv(SOURCE/"submission.csv").Transported)
    idx = replacements.index
    base = probs.probability.to_numpy()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    if OUT.exists():
        raise FileExistsError(OUT)
    runs = []
    for _ in range(2):
        counts.base.seed_everything(counts.SEED, "mps")
        model = joblib.load(SOURCE/"model.joblib")
        model.device = "mps"
        lookup = model.preprocessor.roster
        np.testing.assert_array_equal(lookup.passenger_ids, roster.PassengerId)
        assert roster.HomePlanet.iloc[idx].isna().all()
        lookup.plain.loc[idx,"HomePlanet"] = replacements
        updated = base.copy()
        updated[idx-8693] = model.predict_proba(lookup.plain.loc[idx])[:,1]
        assert np.isfinite(updated).all()
        runs.append(updated)
        model.release()
        del model
        counts.base.release_memory()
    np.testing.assert_array_equal(*runs)
    changed = int(np.sum((base >= .5) != (runs[0] >= .5)))
    assert changed <= len(idx)
    OUT.mkdir(parents=True)
    probs.assign(probability=runs[0],reload_probability=runs[1]).to_csv(OUT/"test_probabilities.csv",index=False)
    pd.DataFrame(dict(PassengerId=probs.PassengerId,Transported=runs[0]>=.5)).to_csv(OUT/"submission.csv",index=False)
    check = workflow.validate_submission(OUT/"submission.csv", ROOT/"data/raw/test.csv")
    selection = dict(version=VERSION,components=[manifest["version"]],threshold=.5)
    result = dict(**stats, **check, changes_from_v96=changed, reload_max_difference=0,
                  version=VERSION, training_rows=8693, source_model_sha256=manifest["model_sha256"],
                  source_manifest_sha256=workflow.sha256(SOURCE/"manifest.json"),
                  recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    REPORT.mkdir(parents=True,exist_ok=True)
    for folder in [OUT,REPORT]:
        for name, obj in [("selection",selection),("manifest",result)]:
            (folder/(name+".json")).write_text(json.dumps(obj,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export",action="store_true")
    args = parser.parse_args()
    frame, updates, stats = consensus()
    print(json.dumps(stats),flush=True)
    if args.export:
        export(frame,updates,stats)
