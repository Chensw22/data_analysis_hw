"""Fit the party-level count correction on original training groups and export v110."""
import json
import subprocess

import numpy as np
import pandas as pd
from scipy.special import expit
from catboost import CatBoostRegressor

from party_outcome_count import ROOT, learn, manifest
from export_party_bill import SOURCE, TEACHER
from relational_error_corrector import logit, write
from kaggle_workflow import sha256, validate_submission

NAME = "proposal_v110_party_counts"
VERSION = "v110_v106_party_count_residual"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME


def export():
    train = pd.read_csv(ROOT/"data/raw/train.csv", usecols=["PassengerId","Transported"])
    ids, groups, features = manifest()
    n = len(train)
    np.testing.assert_array_equal(ids.iloc[:n], train.PassengerId)
    source = pd.read_csv(SOURCE/"test_probabilities.csv", float_precision="round_trip")
    np.testing.assert_array_equal(ids.iloc[n:], source.PassengerId)
    source_record = json.loads((SOURCE/"manifest.json").read_text())
    assert sha256(SOURCE/"submission.csv") == source_record["sha256"]
    np.testing.assert_array_equal(source.probability >= .5, pd.read_csv(SOURCE/"submission.csv").Transported)
    raw_record = json.loads((TEACHER/"manifest.json").read_text())["raw_component"]["files"]
    assert sha256(TEACHER/"raw_probabilities.npz") == raw_record["raw_probabilities.npz"]
    with np.load(TEACHER/"raw_probabilities.npz") as z:
        np.testing.assert_array_equal(z["train_passenger_ids"], train.PassengerId)
        model = learn(groups[:n],features,train.Transported.to_numpy(int),z["oof_raw"],groups[n:])
    eligible = features.loc[groups[n:], "size"].to_numpy() > 1
    query = features.loc[groups[n:][eligible]]
    def predict(model, base):
        p = base.to_numpy().copy()
        p[eligible] = expit(logit(p[eligible]) + model.predict(query))
        return p
    p = predict(model,source.probability)
    OUT.mkdir(parents=True,exist_ok=False)
    model.save_model(str(OUT/"party.cbm"))
    loaded = CatBoostRegressor().load_model(str(OUT/"party.cbm"))
    q = predict(loaded,source.reload_probability)
    np.testing.assert_array_equal(p,q)
    pd.DataFrame(dict(PassengerId=source.PassengerId,probability=p,reload_probability=q)).to_csv(OUT/"test_probabilities.csv", index=False)
    pd.DataFrame(dict(PassengerId=source.PassengerId,Transported=p>=.5)).to_csv(OUT/"submission.csv", index=False)
    check = validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
    record = dict(**check, version=VERSION, source_csv=source_record["sha256"],
                  model_sha256=sha256(OUT/"party.cbm"), reload_difference=0,
                  changed_vs_v106=int(((p>=.5)!=(source.probability>=.5)).sum()),
                  recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip())
    for folder in [OUT,REPORT]:
        write(folder/"manifest.json",record)
        write(folder/"selection.json",dict(version=VERSION,threshold=.5))
    print(record)


if __name__ == "__main__":
    export()
