"""Export the pre-evaluated simple majority of five scored competition files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_workflow import sha256, validate_submission

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
ART = ROOT / "artifacts"
NAME = "proposal_n23_five_release_majority"
MEMBERS = ["proposal_v131_party_consensus", "nov23_robust_release_center",
           "proposal_n23_budget_quarter_three_family", "nov23_specialist_full_manifest",
           "proposal_n23_incumbent_quarter_tabicl_budget"]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_votes(recipe, ids):
    votes = []
    for name, digest in recipe["members"].items():
        path = ART / name / "submission.csv"
        assert sha256(path) == digest
        df = pd.read_csv(path)
        pd.testing.assert_series_equal(ids, df.PassengerId)
        assert df.Transported.dtype == bool
        votes.append(df.Transported.to_numpy())
    return np.column_stack(votes)


def main():
    out, report = ART / NAME, REPORTS / NAME
    for folder in (out, report):
        folder.mkdir(exist_ok=True)
    recipe = {"minimum_votes": 3, "members": {}}
    for name in MEMBERS:
        receipt, = (REPORTS / name / "online_attempts").glob("*.json")
        record = json.loads(receipt.read_text())
        assert record["evaluation"]["Status"] == "COMPLETE"
        recipe["members"][name] = record["file_sha256"]
    ids = pd.read_csv(ROOT / "data/raw/test.csv", usecols=["PassengerId"]).PassengerId
    model = out / "recipe.json"
    save(model, recipe)
    save(report / "recipe.json", recipe)
    matrix = load_votes(recipe, ids)
    with np.load(ART / "nov23_release_vote_information/panel.npz") as panel:
        np.testing.assert_array_equal(matrix, panel["official"])
    reload = json.loads(model.read_text())
    fractions = matrix.mean(axis=1)
    repeated = load_votes(reload, ids).mean(axis=1)
    np.testing.assert_array_equal(repeated, fractions)
    votes = matrix.sum(axis=1) >= recipe["minimum_votes"]
    np.testing.assert_array_equal(votes, repeated >= .5)
    csv = out / "submission.csv"
    pd.DataFrame({"PassengerId": ids, "Transported": votes}).to_csv(csv, index=False)
    pd.DataFrame({"PassengerId": ids, "probability": fractions, "reload_probability": repeated}).to_csv(
        out / "test_probabilities.csv", index=False)
    manifest = validate_submission(csv, ROOT / "data/raw/test.csv")
    for receipt in REPORTS.glob("*/online_attempts/*.json"):
        if receipt.parent.parent != report:
            assert json.loads(receipt.read_text()).get("file_sha256") != manifest["sha256"]
    manifest.update(version="five_release_majority", model_sha256=sha256(model), reload_difference=0,
                    changed_from_v131=int((votes != matrix[:, 0]).sum()))
    for folder in (report, out):
        save(folder / "manifest.json", manifest)
        save(folder / "selection.json", {"version": manifest["version"], "sha256": manifest["sha256"]})
    print(manifest)


if __name__ == "__main__":
    main()
