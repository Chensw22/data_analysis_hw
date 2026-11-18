"""Use group-held-out predictions of manifest fields as label-free passenger features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

import relational_error_corrector as reference
from exaone_validation_data import CATEGORICAL

ROOT = Path(__file__).resolve().parent
REPORT = ROOT/"reports/manifest_surprise"
LOCAL = ROOT/"artifacts/manifest_surprise"
BILLS = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
FIELDS = [*BILLS, "Age", "CryoSleep", "HomePlanet", "Destination", "Deck", "Side"]


def field_target(values, name):
    if name in CATEGORICAL:
        return values.astype(str).where(values.ne("__MISSING__"))
    cuts = [0,100,500,1500] if name in BILLS else [12,17,29,49]
    return pd.Series(np.searchsorted(cuts, values),index=values.index,dtype=float).where(values.notna())


def auxiliary_features(x, groups):
    # Cross-fit on attributes of the fixed public roster. A travel party belongs to exactly one fold.
    unique = np.unique(groups)
    rng = np.random.default_rng(20261119)
    partition = dict(zip(rng.permutation(unique),np.arange(len(unique))%3))
    folds = groups.map(partition).to_numpy()
    extras = pd.DataFrame(index=x.index)
    checks = {}
    for name in FIELDS:
        target = field_target(x[name],name)
        known = target.notna().to_numpy()
        features = x.drop(columns=name)
        categorical = [key for key in CATEGORICAL if key in features]
        surprise = np.full(len(x),np.nan)
        uncertainty = np.zeros(len(x))
        hits = 0
        for fold in range(3):
            fit = (folds!=fold)&known
            query = folds==fold
            model = CatBoostClassifier(iterations=130,depth=5,learning_rate=.09,thread_count=4,
                                       random_seed=20261119,verbose=False,allow_writing_files=False,
                                       cat_features=categorical)
            model.fit(features.loc[fit], target.loc[fit])
            probs = model.predict_proba(features.loc[query])
            lookup = pd.Index(model.classes_).get_indexer(target.loc[query])
            measured = lookup>=0
            assert np.array_equal(measured,known[query])
            ids = np.flatnonzero(query)
            surprise[ids[measured]] = -np.log(np.maximum(probs[measured,lookup[measured]],1e-9))
            uncertainty[ids] = -np.sum(probs*np.log(np.maximum(probs,1e-9)),axis=1)
            hits += int(np.sum(probs[measured].argmax(axis=1)==lookup[measured]))
        extras[name+"_surprise"] = surprise
        extras[name+"_entropy"] = uncertainty
        checks[name] = {"scored":int(known.sum()),"field_accuracy":hits/int(known.sum())}
        print(name,checks[name],flush=True)
    return extras,checks


def fit_model(x,y):
    model = CatBoostClassifier(iterations=650,depth=6,learning_rate=.045,thread_count=4,
                               random_seed=20261119,verbose=False,allow_writing_files=False,
                               cat_features=CATEGORICAL)
    return model.fit(x,y)


def run():
    inputs,splits,_,bases,evidence = reference.overlap.load_authenticated()
    x,dev,y = inputs["features"],inputs["dev"],inputs["y"]
    roster = pd.concat([pd.read_csv(ROOT/"data/raw"/name,usecols=["PassengerId","Cabin"])
                        for name in ["train.csv","test.csv"]],ignore_index=True)
    groups = roster.PassengerId.str.split("_").str[0]
    extras,aux_scores = auxiliary_features(x,groups)
    control = x.assign(GroupSize=groups.map(groups.value_counts()),
                       CabinOccupants=roster.Cabin.map(roster.Cabin.value_counts()))
    extended = pd.concat([control,extras],axis=1)
    raw = bases["rawMitra"]
    results = {}
    preds = {}
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    np.testing.assert_array_equal(catalog.y,y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    for name,frame in [("control",control),("augmented",extended)]:
        p = np.empty(len(dev))
        for train,query in splits:
            model = fit_model(frame.iloc[dev[train]],y[train])
            p[query]=model.predict_proba(frame.iloc[dev[query]])[:,1]
        preds[name]=p
        preds[name+"_with_raw"]=(p+raw)/2
    for name,p in preds.items():
        results[name]=dict(metrics=reference.score(p,y),versus_raw=reference.pair(p,raw,y),
                           hard546=reference.pair(p,raw,y,hard),
                           fold_net=[reference.pair(p[b],raw[b],y[b])["net_correct"] for _,b in splits])
    results["added_features_vs_control"]=reference.pair(preds["augmented"],preds["control"],y)
    LOCAL.mkdir(parents=True,exist_ok=True)
    extras.to_pickle(LOCAL/"attributes.pkl")
    np.savez_compressed(LOCAL/"oof.npz",dev=dev,**preds)
    reference.write(REPORT/"result.json",dict(auxiliary=aux_scores,models=results))
    print(json.dumps(results),flush=True)


if __name__=="__main__":
    with threadpool_limits(4):
        run()
