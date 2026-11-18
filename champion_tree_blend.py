"""Equal-weight blend of the 30-column hard-label student and the champion backbone."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import kaggle_workflow as workflow
import teacher_distillation as student
from relational_error_corrector import pair, score

ROOT = Path(__file__).resolve().parent
NAME = "proposal_v109_champion_tree"
VERSION = "v109_v106_hard_student_half"
REPORT = ROOT / "reports" / NAME
OUT = ROOT / "artifacts" / NAME
SOURCE = ROOT / "artifacts/proposal_nov18_peer_cabin"


def saved_oof(name):
    result = json.loads((ROOT / "reports" / name / "result.json").read_text())
    path = ROOT / "artifacts" / name / "oof.npz"
    assert workflow.sha256(path) == result["oof_sha256"]
    return np.load(path, allow_pickle=False)


def evaluate():
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId", "Transported"])
    with saved_oof("teacher_distillation") as tree, saved_oof("v96_oof_completion") as backbone:
        for key in ["dev_ids", "passenger_ids", "fold_id", "raw"]:
            np.testing.assert_array_equal(tree[key], backbone[key])
        rows = train.iloc[tree["dev_ids"]]
        np.testing.assert_array_equal(rows.PassengerId, tree["passenger_ids"])
        y = rows.Transported.to_numpy()
        p = (tree["hard"] + backbone["prob"]) / 2
        hard = tree["all_recipes_common_wrong546"]
        base = backbone["prob"]
        result = dict(metrics=score(p, y), vs_v96=pair(p, base, y),
                      hard546=pair(p, base, y, hard),
                      folds=[pair(p, base, y, tree["fold_id"] == i) for i in range(3)],
                      source="Local v96 backbone; full export uses v106 with peer-cabin imputation.")
    student.write(REPORT / "development.json", result)
    print(json.dumps(result), flush=True)


def export():
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["Transported"])
    labels = train.Transported.to_numpy(dtype=int)
    frame = student.residual.features()
    model = CatBoostClassifier(**student.PARAMS, random_seed=20261112, cat_features=student.residual.CATS)
    model.fit(frame.iloc[:len(labels)], labels)
    test = frame.iloc[len(labels):]
    source = pd.read_csv(SOURCE / "test_probabilities.csv", float_precision="round_trip")
    source_record = json.loads((SOURCE / "manifest.json").read_text())
    assert workflow.sha256(SOURCE / "submission.csv") == source_record["sha256"]
    np.testing.assert_array_equal(source.probability, source.reload_probability)
    np.testing.assert_array_equal(source.probability >= .5, pd.read_csv(SOURCE / "submission.csv").Transported)
    np.testing.assert_array_equal(source.PassengerId, pd.read_csv(ROOT / "data/raw/test.csv", usecols=["PassengerId"]).PassengerId)
    p = (source.probability.to_numpy() + model.predict_proba(test)[:, 1]) / 2
    OUT.mkdir(parents=True, exist_ok=False)
    model.save_model(str(OUT / "student.cbm"))
    reloaded = CatBoostClassifier().load_model(str(OUT / "student.cbm"))
    q = (source.reload_probability.to_numpy() + reloaded.predict_proba(test)[:, 1]) / 2
    np.testing.assert_array_equal(p, q)
    pd.DataFrame(dict(PassengerId=source.PassengerId, probability=p, reload_probability=q)).to_csv(OUT / "test_probabilities.csv", index=False)
    pd.DataFrame(dict(PassengerId=source.PassengerId, Transported=p >= .5)).to_csv(OUT / "submission.csv", index=False)
    check = workflow.validate_submission(OUT / "submission.csv", ROOT / "data/raw/test.csv")
    manifest = dict(**check, version=VERSION, source_csv=source_record["sha256"],
                    decisions_changed=int(np.sum((p >= .5) != (source.probability >= .5))),
                    model_sha256=workflow.sha256(OUT / "student.cbm"), reload_difference=0,
                    recipe_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    for folder in [OUT, REPORT]:
        student.write(folder / "selection.json", dict(version=VERSION, threshold=.5))
        student.write(folder / "manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["evaluate", "export"])
    args = parser.parse_args()
    (evaluate if args.action == "evaluate" else export)()
