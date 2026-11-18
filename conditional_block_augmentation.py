"""Test within-class expense-block resampling as training augmentation."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import relational_error_corrector as ref
import mixed_type_forest_imputation as forest
from peer_bill_correction import BILLS,load_v96

ROOT=Path(__file__).resolve().parent


def resample(train,labels,seed):
    strata=pd.DataFrame(dict(planet=train.HomePlanet.fillna("missing").astype(str),
        cryo=train.CryoSleep.fillna("missing").astype(str),
        child=train.Age.lt(13),target=labels)).reset_index(drop=True)
    party=train.PassengerId.str.split("_").str[0].to_numpy()
    donors=np.arange(len(train))
    rng=np.random.default_rng(seed)
    for group in strata.groupby(list(strata.columns),sort=False).indices.values():
        if len(group)<2:continue
        ordering=rng.permutation(group)
        other=np.roll(ordering,1)
        valid=party[ordering]!=party[other]
        donors[ordering[valid]]=other[valid]
    np.testing.assert_array_equal(labels[donors],labels)
    result=train.copy()
    result[BILLS]=train.iloc[donors][BILLS].to_numpy()
    return result,int((donors!=np.arange(len(train))).sum())


def fit_predict(frame,y,weights,query,seed):
    x,cats=forest.iterate.build_features(frame,False)
    model=CatBoostClassifier(**forest.CATPARAMS,cat_features=cats,random_seed=seed)
    model.fit(x,y,sample_weight=weights)
    return model.predict_proba(query)[:,1]


def run():
    inp,splits,folds,bases,_=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"]
    roster=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported"))
    np.testing.assert_array_equal(roster.PassengerId.iloc[dev],inp["passengers"])
    x,_=forest.iterate.build_features(roster,False)
    with np.load(ROOT/"artifacts/mixed_type_forest_imputation_oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        native=z["native_missing"]
    p={k:np.empty(len(y)) for k in ["duplicate","expense_block"]}
    donors=[]
    for fold,(a,b) in enumerate(splits):
        train=roster.iloc[dev[a]].reset_index(drop=True)
        synthetic,n=resample(train,y[a],20261118+fold);donors.append(n)
        for name,extra in [("duplicate",train),("expense_block",synthetic)]:
            frame=pd.concat([train,extra],ignore_index=True)
            weights=np.r_[np.ones(len(a)),np.full(len(a),.35)]
            p[name][b]=fit_predict(frame,np.tile(y[a],2),weights,x.iloc[dev[b]],20261112+fold)
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(donor_rows_by_fold=donors)
    v96=load_v96(dev,folds)
    for name,q in p.items():
        result[name]=dict(metrics=ref.score(q,y),vs_native=ref.pair(q,native,y),vs_duplicate=ref.pair(q,p["duplicate"],y),
                          hard546=ref.pair(q,native,y,hard),hard_correct=int(((q[hard]>=.5)==y[hard]).sum()),
                          half_v96=ref.score((q+v96)/2,y),fold_net_vs_duplicate=[ref.pair(q[b],p["duplicate"][b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/conditional_block_augmentation/result.json",result)
    print(result)


if __name__=="__main__":
    run()
