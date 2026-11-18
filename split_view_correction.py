"""Nested stacking of separate on-board spending and passenger/berth views."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import relational_error_corrector as ref

ROOT = Path(__file__).resolve().parent
BILLS = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]


def fit_views(x, y, train, seed):
    models = []
    for cols in [BILLS, [c for c in x if c not in BILLS]]:
        categorical = [c for c in cols if x[c].dtype == object or pd.api.types.is_string_dtype(x[c])]
        clf = CatBoostClassifier(iterations=360,depth=5,learning_rate=.05,random_seed=seed,thread_count=4,
                                 verbose=False,allow_writing_files=False,cat_features=categorical)
        clf.fit(x.iloc[train][cols],y[train])
        models.append((cols, clf))
    return models


def predict_views(models, rows):
    return np.column_stack([clf.predict_proba(rows[cols])[:,1] for cols, clf in models])


def learn_views(x, y, train, query, seed):
    return predict_views(fit_views(x,y,train,seed),x.iloc[query])


def predictors(raw,views,x):
    r = logit(np.clip(raw,1e-6,1-1e-6))[:,None]
    v = logit(np.clip(views,1e-6,1-1e-6))
    # Let each view's contribution depend on whether the person's bills are complete or all zero.
    missing = x[BILLS].isna().any(axis=1).to_numpy()[:,None]
    zero = (x[BILLS].sum(axis=1,min_count=5)==0).to_numpy()[:,None]
    return np.column_stack([r,v,missing,zero,v*missing,v*zero])


def run():
    inputs,splits,_,bases,evidence = ref.overlap.load_authenticated()
    dev,y = inputs["dev"],inputs["y"]
    roster = pd.concat([pd.read_csv(ROOT/"data/raw"/name,usecols=["PassengerId","Cabin"])
                          for name in ["train.csv","test.csv"]],ignore_index=True)
    x = inputs["features"].copy()
    groups = roster.PassengerId.str.split("_").str[0]
    x["GroupSize"] = groups.map(groups.value_counts())
    x["CabinOccupants"] = roster.Cabin.map(roster.Cabin.value_counts())
    x=x.iloc[dev].copy()
    raw=bases["rawMitra"]
    views = np.full((len(y),2),np.nan)
    combo = np.empty(len(y))
    for fold,(a,b) in enumerate(splits):
        inner_raw = ref.load_inner(fold,a,b,dev,y,evidence)
        with np.load(ROOT/f"artifacts/planet_mixture_20261117/outer{fold}_inner_oof.npz") as z:
            ids=z["inner_fold_id"]
        inner=np.full((len(a),2),np.nan)
        for j in range(3):
            inner[ids==j]=learn_views(x,y,a[ids!=j],a[ids==j],20261112+fold)
        views[b]=learn_views(x,y,a,b,20261112+fold)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1))
        model.fit(predictors(inner_raw,inner,x.iloc[a]),y[a])
        combo[b]=model.predict_proba(predictors(raw[b],views[b],x.iloc[b]))[:,1]
    cat=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cat.row_id,dev)
    hard=cat.all_recipes_still_wrong.to_numpy(bool)
    remaining=hard.copy()
    for file,key,id_field in [("hard_weighted_combo_oof.npz","weighted_cat","dev_ids"),
                              ("mechanism_combo_empty_query_fix/oof.npz","mechanism","dev_ids"),
                              ("top3_stage/density_oof.npz","density","dev")]:
        with np.load(ROOT/"artifacts"/file) as z:
            np.testing.assert_array_equal(z[id_field],dev)
            remaining &= (z[key]>=.5)!=y
    assert remaining.sum()==475
    result={}
    for k,p in [("spending",views[:,0]),("passenger_and_cabin",views[:,1]),("stack",combo)]:
        result[k]=dict(metrics=ref.score(p,y),versus_raw=ref.pair(p,raw,y),hard546=ref.pair(p,raw,y,hard),
            remaining475=ref.pair(p,raw,y,remaining),fold_net=[ref.pair(p[b],raw[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/top3_stage/split_views.json",result)
    np.savez_compressed(ROOT/"artifacts/top3_stage/split_views.npz",dev=dev,views=views,stack=combo)
    print(json.dumps(result))


if __name__=="__main__":
    run()
