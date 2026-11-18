"""Measure train/test covariate shift on unseen parties and test propensity weighting."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import mixed_type_forest_imputation as forest
import relational_error_corrector as ref
from peer_bill_correction import load_v96

ROOT=Path(__file__).resolve().parent


def weights(roster, x, train_rows):
    domain=(np.arange(len(x))>=train_rows).astype(int)
    groups=roster.PassengerId.str.split("_").str[0]
    features=x.copy()
    features["GroupSize"]=groups.map(groups.value_counts())
    features["CabinOccupants"]=roster.Cabin.map(roster.Cabin.value_counts())
    propensity=np.zeros(len(x))
    folds=[]
    for fold,(a,b) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=20261118).split(features,domain,groups)):
        assert not set(groups.iloc[a]) & set(groups.iloc[b])
        model=CatBoostClassifier(iterations=240,depth=4,learning_rate=.035,random_seed=20261118+fold,
            l2_leaf_reg=8,thread_count=2,allow_writing_files=False,verbose=False,cat_features=forest.CAT)
        model.fit(features.iloc[a],domain[a])
        propensity[b]=model.predict_proba(features.iloc[b])[:,1]
        folds.append(float(roc_auc_score(domain[b],propensity[b])))
    odds=propensity/(1-propensity)*train_rows/(len(x)-train_rows)
    weight=np.clip(np.sqrt(odds),.5,2)
    return weight,dict(auc=float(roc_auc_score(domain,propensity)),fold_auc=folds,
        membership_prevalence=float(domain.mean()),weight_quantiles_train=np.quantile(weight[:train_rows],[0,.05,.5,.95,1]).tolist())


def run():
    inp,splits,folds,bases,_=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"]
    files=[pd.read_csv(ROOT/"data/raw"/name,usecols=lambda c:c not in ("Name","Transported")) for name in ["train.csv","test.csv"]]
    roster=pd.concat(files,ignore_index=True)
    x,_=forest.iterate.build_features(roster,False)
    np.testing.assert_array_equal(roster.PassengerId.iloc[dev],inp["passengers"])
    w,domain=weights(roster,x,len(files[0]))
    result=dict(domain=domain)
    preds={}
    for mode in ["native","target_weighted"]:
        p=np.empty(len(y))
        for fold,(a,b) in enumerate(splits):
            model=CatBoostClassifier(**forest.CATPARAMS,cat_features=forest.CAT,random_seed=20261112+fold)
            weights_train=w[dev[a]]
            model.fit(x.iloc[dev[a]],y[a],sample_weight=None if mode=="native" else weights_train/weights_train.mean())
            p[b]=model.predict_proba(x.iloc[dev[b]])[:,1]
        preds[mode]=p
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    v96=load_v96(dev,folds)
    for mode,p in preds.items():
        result[mode]=dict(metrics=ref.score(p,y),vs_native=ref.pair(p,preds["native"],y),
             hard546=ref.pair(p,preds["native"],y,hard),
             fold_net=[ref.pair(p[b],preds["native"][b],y[b])["net_correct"] for _,b in splits],
             half_v96=ref.score((v96+p)/2,y))
    np.savez_compressed(ROOT/"artifacts/population_shift.npz",dev=dev,weight=w,**preds)
    ref.write(ROOT/"reports/population_shift/result.json",result)
    print(result)


if __name__=="__main__":
    run()
