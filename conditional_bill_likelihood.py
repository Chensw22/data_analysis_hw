"""Factor passenger outcome odds into a prior and five conditional bill likelihood ratios."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL

ROOT = Path(__file__).resolve().parent
REPORT = ROOT/"reports/bill_likelihood"
BILLS = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
PARAMS = dict(iterations=240,depth=5,learning_rate=.08,thread_count=4,
              random_seed=20261119,verbose=False,allow_writing_files=False)


def classify(x,y,categorical):
    model = CatBoostClassifier(cat_features=categorical,**PARAMS)
    return model.fit(x,y)


def train_and_predict(train,y,query):
    """Score each observed query bill under both possible outcomes."""
    context = [name for name in train if name not in BILLS]
    prior_model = classify(train[context],y,CATEGORICAL)
    prior = prior_model.predict_proba(query[context])[:,1]
    evidence = []
    for name in BILLS:
        observed = train[name].notna().to_numpy()
        amounts = train.loc[train[name]>0,name]
        cuts = np.unique(amounts.quantile([.2,.4,.6,.8]).to_numpy())
        def encode(series):
            return np.where(series==0,0,1+np.searchsorted(cuts,series))
        features = train.loc[observed,context].assign(hypothesis=y[observed])
        response = encode(train.loc[observed,name])
        model = classify(features,response,CATEGORICAL)
        measured = query[name].notna().to_numpy()
        categories = pd.Index(model.classes_).get_indexer(encode(query.loc[measured,name]))
        assert (categories>=0).all()
        measured_features = query.loc[measured,context]
        loglik = []
        for hypothesis in [0,1]:
            probs = model.predict_proba(measured_features.assign(hypothesis=hypothesis))
            loglik.append(np.log(np.maximum(probs[np.arange(len(categories)),categories],1e-8)))
        increment = np.zeros(len(query))
        increment[measured] = loglik[1]-loglik[0]
        evidence.append(increment)
        context.append(name)
    posterior = expit(logit(np.clip(prior,1e-7,1-1e-7))+np.sum(evidence,axis=0))
    return prior,posterior


def run():
    inputs,splits,_,bases,_ = ref.overlap.load_authenticated()
    dev,y = inputs["dev"],inputs["y"]
    frame = inputs["features"].iloc[dev]
    raw = bases["rawMitra"]
    preds = {k:np.empty(len(y)) for k in ["passenger_prior","bill_posterior","direct_classifier"]}
    for fold,(a,b) in enumerate(splits):
        train,query = frame.iloc[a],frame.iloc[b]
        p,q = train_and_predict(train,y[a],query)
        preds["passenger_prior"][b] = p
        preds["bill_posterior"][b] = q
        preds["direct_classifier"][b] = classify(train,y[a],CATEGORICAL).predict_proba(query)[:,1]
        print("Completed fold",fold,flush=True)
    for name in ["bill_posterior","direct_classifier"]:
        preds[name+"_half_raw"] = (preds[name]+raw)/2
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    np.testing.assert_array_equal(catalog.y,y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    result = {key:dict(metrics=ref.score(p,y),versus_raw=ref.pair(p,raw,y),hard=ref.pair(p,raw,y,hard),
                       fold_net=[ref.pair(p[b],raw[b],y[b])["net_correct"] for _,b in splits])
              for key,p in preds.items()}
    result["posterior_vs_direct"] = ref.pair(preds["bill_posterior"],preds["direct_classifier"],y)
    ref.write(REPORT/"result.json",result)
    np.savez_compressed(ROOT/"artifacts/bill_likelihood_oof.npz",dev=dev,**preds)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    run()
