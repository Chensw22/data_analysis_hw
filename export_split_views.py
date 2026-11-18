"""Fit fresh split-view components, reuse the verified raw Mitra full-data component."""
import json
import subprocess
import sys
import fcntl
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import kaggle_workflow as workflow
from exaone_validation_data import compact_features
from split_view_correction import fit_views,predict_views,predictors,ROOT

PROPOSAL="proposal_top3_split_views"
VERSION="v107_spending_cabin_mitra_stack"
OUT=ROOT/"artifacts"/PROPOSAL
REPORT=ROOT/"reports"/PROPOSAL
RAW=ROOT/"artifacts/proposal_ready_nonlinear_residual_v2"


def raw_inputs():
    manifest = json.loads((RAW/"manifest.json").read_text())
    assert manifest["raw_component"]["raw_reload_max_abs_diff"]==0
    for file,digest in manifest["raw_component"]["files"].items():
        assert workflow.sha256(RAW/file)==digest
    return np.load(RAW/"raw_probabilities.npz",allow_pickle=False)


def export():
    if OUT.exists():
        raise FileExistsError(OUT)
    train,test=[pd.read_csv(ROOT/"data/raw"/name,usecols=lambda c:c!="Name") for name in ["train.csv","test.csv"]]
    roster=pd.concat([train.drop(columns="Transported"),test],ignore_index=True)
    x=compact_features(roster)
    groups=roster.PassengerId.str.split("_").str[0]
    x["GroupSize"]=groups.map(groups.value_counts())
    x["CabinOccupants"]=roster.Cabin.map(roster.Cabin.value_counts())
    y=train.Transported.to_numpy(int)
    with raw_inputs() as cache:
        np.testing.assert_array_equal(cache["train_passenger_ids"],train.PassengerId)
        np.testing.assert_array_equal(cache["test_ids"],test.PassengerId)
        oof,raw,fold_ids=cache["oof_raw"],cache["test_raw"],cache["fold_ids"]
    cross=np.full((len(train),2),np.nan)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261112).split(x.iloc[:len(y)],y)):
        assert np.all(fold_ids[va]==fold)
        cross[va]=predict_views(fit_views(x,y,tr,20261112),x.iloc[va])
    meta=make_pipeline(StandardScaler(),LogisticRegression(C=.1))
    meta.fit(predictors(oof,cross,x.iloc[:len(y)]),y)
    models=fit_views(x,y,np.arange(len(y)),20261112)
    query=x.iloc[len(y):]
    def predict(stack, views, raw_p):
        return stack.predict_proba(predictors(raw_p,predict_views(views,query),query))[:,1]
    p=predict(meta,models,raw)
    OUT.mkdir(parents=True)
    joblib.dump((meta,models),OUT/"model.joblib")
    del meta,models
    with np.load(RAW/"raw_reload_probabilities.npz") as reloaded:
        np.testing.assert_array_equal(test.PassengerId,reloaded["test_ids"])
        restored=predict(*joblib.load(OUT/"model.joblib"),reloaded["test_raw"])
    np.testing.assert_array_equal(p,restored)
    assert np.isfinite(p).all()
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(OUT/"submission.csv",index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=restored)).to_csv(OUT/"test_probabilities.csv",index=False)
    check=workflow.validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
    selection=dict(version=VERSION,threshold=.5)
    result=dict(**check,version=VERSION,raw_component_manifest_sha256=workflow.sha256(RAW/"manifest.json"),
                model_sha256=workflow.sha256(OUT/"model.joblib"),training_rows=len(y),reload_difference=0,
                recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    REPORT.mkdir(parents=True)
    for folder in [OUT,REPORT]:
        for name,record in [("manifest",result),("selection",selection)]:
            (folder/(name+".json")).write_text(json.dumps(record,indent=2)+"\n")
    print(result)


def submit():
    from kaggle.api.kaggle_api_extended import KaggleApi
    with (ROOT/"artifacts/online_iteration_20261117/submission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=json.loads((REPORT/"manifest.json").read_text())
        check=workflow.validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
        assert result["sha256"]==check["sha256"] and result["reload_difference"]==0
        assert workflow.sha256(OUT/"model.joblib")==result["model_sha256"]
        assert workflow.sha256(RAW/"manifest.json")==result["raw_component_manifest_sha256"]
        if any(json.loads(p.read_text()).get("file_sha256")==check["sha256"] for p in (ROOT/"reports").rglob("online_attempts/*.json")):
            raise ValueError("Duplicate upload")
        api=KaggleApi()
        api.authenticate()
        limits=json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        print(limits,flush=True)
        if limits["numAllowedNow"]<=0:
            raise RuntimeError("Daily quota exhausted")
        subprocess.Popen(["osascript",str(ROOT/"remind_online_first_submission.applescript"),"v107 消费、舱位与 Mitra 组合已核验，现提交一次线上评测。"],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        workflow.submit(PROPOSAL,120)


if __name__=="__main__":
    {"export":export,"submit":submit}[sys.argv[1]]()
