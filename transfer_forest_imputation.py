"""Transfer the paired imputation effect only to passengers with incomplete attributes."""
import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

import mixed_type_forest_imputation as forest
import relational_error_corrector as ref
from kaggle_workflow import sha256, validate_submission
from peer_bill_correction import load_v96

ROOT = Path(__file__).resolve().parent
NAME = "proposal_v112_forest_transfer"
VERSION = "v112_v108_paired_imputation"
OUT, REPORT = ROOT/"artifacts"/NAME, ROOT/"reports"/NAME
SOURCE = ROOT/"artifacts/proposal_v108_party_bills"


def transfer(base, before, after, mask):
    result = np.asarray(base,dtype=float).copy()
    result[mask] = expit(ref.logit(result[mask]) + .5*(ref.logit(after[mask])-ref.logit(before[mask])))
    return result


def evaluate():
    inp,splits,folds,bases,_=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"]
    rowmask=forest.semantic(inp["features"].iloc[dev]).isna().any(axis=1).to_numpy()
    base=load_v96(dev,folds)
    with np.load(ROOT/"artifacts/peer_bill_offsets.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        proxy=expit(ref.logit(base)+z["peer_bills"])
    path=ROOT/"artifacts/mixed_type_forest_imputation_oof.npz"
    assert sha256(path)==json.loads((forest.REPORT/"result.json").read_text())["oof_sha256"]
    with np.load(path) as z:
        np.testing.assert_array_equal(dev,z["dev_ids"])
        np.testing.assert_array_equal(folds,z["fold_id"])
        p=transfer(proxy,z["native_missing"],z["forest_chain"],rowmask)
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(metrics=ref.score(p,y),vs_proxy=ref.pair(p,proxy,y),hard546=ref.pair(p,proxy,y,hard),
                eligible=int(rowmask.sum()),fold_net=[ref.pair(p[b],proxy[b],y[b])["net_correct"] for _,b in splits])
    ref.write(REPORT/"development.json",result)
    print(result)


def export():
    roster=pd.concat([pd.read_csv(ROOT/"data/raw"/name,usecols=lambda c:c not in ("Name","Transported"))
                      for name in ["train.csv","test.csv"]],ignore_index=True)
    labels=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["Transported","PassengerId"])
    n=len(labels)
    np.testing.assert_array_equal(roster.PassengerId.iloc[:n],labels.PassengerId)
    x,_=forest.iterate.build_features(roster,False)
    models,_=forest.fit_pair(x.iloc[:n],labels.Transported.to_numpy(int),20261112)
    test=x.iloc[n:]
    eligible=forest.semantic(test).isna().any(axis=1).to_numpy()
    source=pd.read_csv(SOURCE/"test_probabilities.csv",float_precision="round_trip")
    np.testing.assert_array_equal(roster.PassengerId.iloc[n:],source.PassengerId)
    source_record=json.loads((SOURCE/"manifest.json").read_text())
    assert sha256(SOURCE/"submission.csv")==source_record["sha256"]
    np.testing.assert_array_equal(pd.read_csv(SOURCE/"submission.csv").Transported,source.probability>=.5)
    def predict(pack, base):
        return transfer(base,pack["native_missing"].predict_proba(test)[:,1],pack["forest_chain"].predict_proba(test)[:,1],eligible)
    p=predict(models,source.probability)
    OUT.mkdir(parents=True,exist_ok=False)
    joblib.dump(models,OUT/"models.joblib")
    q=predict(joblib.load(OUT/"models.joblib"),source.reload_probability)
    # Forest regression parallel summation may differ by floating-point roundoff.
    np.testing.assert_allclose(p,q,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(p>=.5,q>=.5)
    pd.DataFrame(dict(PassengerId=source.PassengerId,probability=p,reload_probability=q)).to_csv(OUT/"test_probabilities.csv",index=False)
    pd.DataFrame(dict(PassengerId=source.PassengerId,Transported=p>=.5)).to_csv(OUT/"submission.csv",index=False)
    check=validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
    manifest=dict(**check,version=VERSION,source_csv=source_record["sha256"],model_sha256=sha256(OUT/"models.joblib"),
                  eligible_test=int(eligible.sum()),changed_vs_v108=int(((p>=.5)!=(source.probability>=.5)).sum()),
                  reload_difference=float(np.abs(p-q).max()),recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    for folder in [OUT,REPORT]:
        ref.write(folder/"manifest.json",manifest)
        ref.write(folder/"selection.json",dict(version=VERSION,threshold=.5))
    print(manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["evaluate","export"])
    args=parser.parse_args()
    (evaluate if args.action=="evaluate" else export)()
