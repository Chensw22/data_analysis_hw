"""Test selective downweighting of confident teacher disagreements against a matched shuffle."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import relational_error_corrector as ref
import teacher_distillation as teachers
import mixed_type_forest_imputation as forest
from peer_bill_correction import load_v96

ROOT=Path(__file__).resolve().parent


def normalized(flag,labels):
    w=np.where(flag,.4,1.)
    for label in [0,1]:
        mask=labels==label
        w[mask]/=w[mask].mean()
    return w


def run():
    inp,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"]
    x=teachers.residual.features().iloc[dev]
    with np.load(ROOT/"artifacts/mixed_type_forest_imputation_oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev);native=z["native_missing"]
    probs={k:np.empty(len(y)) for k in ["consensus","shuffled"]}
    flagged=[]
    for fold,(a,b) in enumerate(splits):
        raw=ref.load_inner(fold,a,b,dev,y,evidence)
        kumo=teachers.kumo_inner(fold,a,b,dev,y,evidence)
        truth=y[a]==1
        # Teachers give at most 15% probability to the observed label, independently of validation outcomes.
        flag=(np.where(truth,raw,1-raw)<.15)&(np.where(truth,kumo,1-kumo)<.15)
        shuffled=flag.copy()
        bins=pd.DataFrame(dict(label=y[a],planet=x.HomePlanet.iloc[a].to_numpy()))
        rng=np.random.default_rng(20261118+fold)
        for ids in bins.groupby(["label","planet"]).indices.values():
            shuffled[ids]=rng.permutation(flag[ids])
        flagged.append(int(flag.sum()))
        for name,mask in [("consensus",flag),("shuffled",shuffled)]:
            model=CatBoostClassifier(**forest.CATPARAMS,cat_features=forest.CAT,random_seed=20261112+fold)
            model.fit(x.iloc[a],y[a],sample_weight=normalized(mask,y[a]))
            probs[name][b]=model.predict_proba(x.iloc[b])[:,1]
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(flagged_training_rows=flagged)
    v96=load_v96(dev,folds)
    for name,p in probs.items():
        result[name]=dict(metrics=ref.score(p,y),vs_native=ref.pair(p,native,y),vs_shuffled=ref.pair(p,probs["shuffled"],y),
                          hard546=ref.pair(p,native,y,hard),half_v96=ref.score((p+v96)/2,y),
                          fold_net=[ref.pair(p[b],native[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/consensus_training_weights/result.json",result)
    print(result)


if __name__=="__main__":
    run()
