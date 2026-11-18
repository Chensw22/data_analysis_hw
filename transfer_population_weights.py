"""Test if population weights outperform shuffled weights and transfer the paired change."""
import argparse
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from catboost import CatBoostClassifier

import population_shift as source
from kaggle_workflow import sha256,validate_submission
from peer_bill_correction import load_v96

ref=source.ref
ROOT=Path(__file__).resolve().parent
NAME="proposal_v113_population_transfer"
VERSION="v113_v108_population_weight_effect"
OUT,REPORT=ROOT/"artifacts"/NAME,ROOT/"reports"/NAME
PARENT=ROOT/"artifacts/proposal_v108_party_bills"


def build():
    roster=pd.concat([pd.read_csv(ROOT/"data/raw"/f,usecols=lambda c:c not in ("Transported","Name"))
                      for f in ["train.csv","test.csv"]],ignore_index=True)
    features,_=source.forest.iterate.build_features(roster,False)
    return roster,features


def model(seed):
    return CatBoostClassifier(**source.forest.CATPARAMS,cat_features=source.forest.CAT,random_seed=seed)


def adjust(base, weighted, native):
    return expit(ref.logit(base)+.5*(ref.logit(weighted)-ref.logit(native)))


def evaluate():
    inp,splits,folds,bases,_=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"]
    roster,x=build()
    np.testing.assert_array_equal(roster.PassengerId.iloc[dev],inp["passengers"])
    with np.load(ROOT/"artifacts/population_shift.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        w=z["weight"];weighted=z["target_weighted"];native=z["native"]
    control=np.empty(len(y))
    for fold,(a,b) in enumerate(splits):
        train_w=np.random.default_rng(20261118+fold).permutation(w[dev[a]])
        fitted=model(20261112+fold)
        fitted.fit(x.iloc[dev[a]],y[a],sample_weight=train_w/train_w.mean())
        control[b]=fitted.predict_proba(x.iloc[dev[b]])[:,1]
    with np.load(ROOT/"artifacts/peer_bill_offsets.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        base=expit(ref.logit(load_v96(dev,folds))+z["peer_bills"])
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(shuffled_tree_metrics=ref.score(control,y),weighted_vs_shuffled=ref.pair(weighted,control,y))
    for name,other in [("weighted",weighted),("shuffled",control)]:
        p=adjust(base,other,native)
        result[name]=dict(metrics=ref.score(p,y),vs_proxy=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
            fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(REPORT/"development.json",result)
    print(result)


def export():
    roster,x=build()
    labels=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["Transported","PassengerId"])
    n=len(labels)
    np.testing.assert_array_equal(labels.PassengerId,roster.PassengerId.iloc[:n])
    w,_=source.weights(roster,x,n)
    native,weighted=model(20261112),model(20261112)
    native.fit(x.iloc[:n],labels.Transported.to_numpy(int))
    weighted.fit(x.iloc[:n],labels.Transported.to_numpy(int),sample_weight=w[:n]/w[:n].mean())
    base=pd.read_csv(PARENT/"test_probabilities.csv",float_precision="round_trip")
    m=json.loads((PARENT/"manifest.json").read_text())
    assert sha256(PARENT/"submission.csv")==m["sha256"]
    np.testing.assert_array_equal(base.PassengerId,roster.PassengerId.iloc[n:])
    np.testing.assert_array_equal(base.probability>=.5,pd.read_csv(PARENT/"submission.csv").Transported)
    def predict(models,probs):
        return adjust(probs,models[1].predict_proba(x.iloc[n:])[:,1],models[0].predict_proba(x.iloc[n:])[:,1])
    p=predict((native,weighted),base.probability)
    OUT.mkdir(parents=True,exist_ok=False)
    joblib.dump((native,weighted),OUT/"models.joblib")
    q=predict(joblib.load(OUT/"models.joblib"),base.reload_probability)
    np.testing.assert_array_equal(p,q)
    pd.DataFrame(dict(PassengerId=base.PassengerId,probability=p,reload_probability=q)).to_csv(OUT/"test_probabilities.csv",index=False)
    pd.DataFrame(dict(PassengerId=base.PassengerId,Transported=p>=.5)).to_csv(OUT/"submission.csv",index=False)
    check=validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
    manifest=dict(**check,version=VERSION,source_csv=m["sha256"],model_sha256=sha256(OUT/"models.joblib"),
                  reload_difference=0,changed_vs_v108=int(((p>=.5)!=(base.probability>=.5)).sum()),
                  recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    for folder in [OUT,REPORT]:
        ref.write(folder/"manifest.json",manifest);ref.write(folder/"selection.json",dict(version=VERSION,threshold=.5))
    print(manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["evaluate","export"])
    args=parser.parse_args()
    (evaluate if args.action=="evaluate" else export)()
