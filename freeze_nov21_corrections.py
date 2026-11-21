"""Freeze and independently replay a submitted incumbent plus explicit correction IDs."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from confirm_sleeper_scope import NAME as BASE
from export_consensus_routes import official_decisions
from kaggle_workflow import ROOT, sha256, validate_submission
from relational_error_corrector import write


def freeze(version, folder, source, candidate=None):
    art = ROOT / "artifacts"
    name = "proposal_" + version
    path = art / folder / ((candidate or version.split("_")[0]) + ".csv")
    test = ROOT / "data/raw/test.csv"
    validate_submission(path, test)
    frame = pd.read_csv(path)
    ids = frame.PassengerId.to_numpy(str)
    anchor, anchor_hash = official_decisions(BASE, ids)
    label = frame.Transported.to_numpy(bool)
    changed = ids[label != anchor].tolist()
    if not changed:
        raise ValueError("This candidate reproduces the incumbent.")
    out = art / name
    out.mkdir(exist_ok=True)
    recipe = out / "corrections.json"
    write(recipe, dict(baseline=BASE, baseline_sha256=anchor_hash, change_passengers=changed))
    saved = json.loads(recipe.read_text())
    replay_ids = pd.read_csv(test, usecols=["PassengerId"]).PassengerId
    restored, digest = official_decisions(saved["baseline"], replay_ids.to_numpy(str))
    if digest != saved["baseline_sha256"] or len(saved["change_passengers"]) != len(set(changed)):
        raise ValueError("Correction ledger failed verification")
    replay = restored ^ replay_ids.isin(saved["change_passengers"]).to_numpy()
    np.testing.assert_array_equal(label, replay)
    pd.DataFrame(dict(PassengerId=ids,Transported=replay)).to_csv(out / "submission.csv", index=False)
    receipt = validate_submission(out / "submission.csv", test)
    for attempt in (ROOT / "reports").glob("*/online_attempts/*.json"):
        record = json.loads(attempt.read_text())
        if record.get("file_sha256") == receipt["sha256"] and record.get("run") != name:
            raise ValueError(f"Already submitted: {attempt}")
    pd.DataFrame(dict(PassengerId=ids,probability=label.astype(float),
                      reload_probability=replay.astype(float))).to_csv(out / "test_probabilities.csv",index=False)
    manifest = dict(**receipt, version=version, model_sha256=sha256(recipe), reload_difference=0,
                    source_csv=str(path.relative_to(ROOT)), source_file=source, source_sha256=sha256(ROOT / source), source_csv_sha256=sha256(path),
                    changed_vs_v128=len(changed), baseline_sha256=anchor_hash)
    for dest in [out, ROOT / "reports" / name]:
        write(dest / "manifest.json", manifest)
        write(dest / "selection.json", dict(version=version, threshold=.5))
    shutil.copy2(recipe, ROOT / "reports" / name / recipe.name)
    print(name, len(changed), receipt["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["version", "folder", "source"]:
        parser.add_argument(arg)
    parser.add_argument("--candidate", help="Original experiment name before assigning its release number")
    args = parser.parse_args()
    freeze(args.version, args.folder, args.source, args.candidate)
