"""Fit the leave-self-out party correction and apply it to v106."""
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

import kaggle_workflow as workflow
import peer_bill_correction as peer
from relational_error_corrector import logit

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"artifacts/proposal_nov18_peer_cabin"
TEACHER=ROOT/"artifacts/proposal_ready_nonlinear_residual_v2"
NAME="proposal_v108_party_bills"
VERSION="v108_peer_bills_v106"
OUT=ROOT/"artifacts"/NAME
REPORT=ROOT/"reports"/NAME


def export():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_manifest=json.loads((SOURCE/"manifest.json").read_text())
    assert source_manifest["version"]=="v106_v96_peer_cabin" and source_manifest["reload_max_difference"]==0
    assert workflow.sha256(SOURCE/"submission.csv")==source_manifest["sha256"]
    teacher=json.loads((TEACHER/"manifest.json").read_text())["raw_component"]["files"]
    assert workflow.sha256(TEACHER/"raw_probabilities.npz")==teacher["raw_probabilities.npz"]
    labels=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","Transported"])
    ids,known,_,features=peer.context()
    n=len(labels)
    with np.load(TEACHER/"raw_probabilities.npz") as packet:
        np.testing.assert_array_equal(ids.iloc[:n],packet["train_passenger_ids"])
        np.testing.assert_array_equal(ids.iloc[:n],labels.PassengerId)
        training=known[:n]
        fitted=peer.fit_offset(features[:n][training],packet["oof_raw"][training],labels.Transported.to_numpy(int)[training])
    test=pd.read_csv(SOURCE/"test_probabilities.csv",float_precision="round_trip")
    np.testing.assert_array_equal(ids.iloc[n:],test.PassengerId)
    np.testing.assert_array_equal(test.probability,test.reload_probability)
    np.testing.assert_array_equal(test.probability>=.5,pd.read_csv(SOURCE/"submission.csv").Transported)
    def predict(pack,values):
        p=values.to_numpy().copy()
        mask=known[n:]
        p[mask]=expit(logit(p[mask])+peer.apply(pack,features[n:][mask]))
        return p
    p=predict(fitted,test.probability)
    assert len(p)==4277 and np.isfinite(p).all()
    OUT.mkdir(parents=True)
    joblib.dump(fitted,OUT/"peer_coefficients.joblib")
    q=predict(joblib.load(OUT/"peer_coefficients.joblib"),test.reload_probability)
    np.testing.assert_array_equal(p,q)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=q)).to_csv(OUT/"test_probabilities.csv",index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(OUT/"submission.csv",index=False)
    check=workflow.validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
    result=dict(**check,version=VERSION,eligible_test=int(known[n:].sum()),eligible_fit=int(known[:n].sum()),
                decisions_changed=int(((p>=.5)!=(test.probability>=.5)).sum()),reload_difference=0,
                source_csv=source_manifest["sha256"],teacher_oof_sha256=teacher["raw_probabilities.npz"],
                correction_sha256=workflow.sha256(OUT/"peer_coefficients.joblib"),
                recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    REPORT.mkdir(parents=True)
    for folder in [OUT,REPORT]:
        for k,v in [("selection",dict(version=VERSION,threshold=.5)),("manifest",result)]:
            (folder/(k+".json")).write_text(json.dumps(v,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    export()
