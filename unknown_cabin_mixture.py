"""Average conditional predictions over plausible cabins when companions cannot fill one."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit

import relational_error_corrector as ref
from peer_bill_correction import load_v96
from exaone_validation_data import CATEGORICAL, compact_features

ROOT = Path(__file__).resolve().parent
MISSING = "__MISSING__"


def roster():
    people = pd.concat([pd.read_csv(ROOT/"data/raw"/f,usecols=lambda c:c not in ("Name","Transported"))
                        for f in ("train.csv","test.csv")],ignore_index=True)
    x = compact_features(people)
    group = people.PassengerId.str.split("_").str[0].astype(int)
    location = x.Deck + "/" + x.Side
    known = people.Cabin.notna()
    places = location[known].groupby(group[known]).nunique()
    # Unique known companion deck and side already belong to the v106 repair.
    eligible = (~known & group.map(places).fillna(0).ne(1)).to_numpy()
    attributes = x.drop(columns=["Deck","Side","CabinNumber"])
    prior = CatBoostClassifier(iterations=180,depth=4,learning_rate=.08,random_seed=20261118,
        cat_features=[c for c in CATEGORICAL if c in attributes],thread_count=3,
        verbose=False,allow_writing_files=False)
    prior.fit(attributes.loc[known],location[known])
    columns = prior.classes_.tolist()
    weights = prior.predict_proba(attributes.loc[eligible])
    filled = []
    target_groups = group[eligible]
    for index, place in enumerate(columns):
        mask = known & location.eq(place)
        numbers = x.CabinNumber[mask].groupby(group[mask]).median().sort_index()
        part = x.loc[eligible].copy()
        deck, side = place.split("/")
        part["Deck"] = deck
        part["Side"] = side
        # All knots use observed attributes. No target outcomes enter the cabin model.
        part["CabinNumber"] = np.rint(np.interp(target_groups,numbers.index,numbers.to_numpy()))
        filled.append(part)
    return people.PassengerId, x, eligible, weights, filled


def run():
    inputs,splits,folds,bases,_ = ref.overlap.load_authenticated()
    dev,y=inputs["dev"],inputs["y"]
    ids,x,eligible,weights,filled = roster()
    np.testing.assert_array_equal(ids.iloc[dev],inputs["passengers"])
    mapping = pd.Series(np.arange(eligible.sum()),index=np.flatnonzero(eligible))
    delta = np.zeros(len(dev))
    original = np.empty(len(dev))
    marginal = np.empty(len(dev))
    for a,b in splits:
        model=CatBoostClassifier(iterations=500,depth=6,learning_rate=.05,verbose=False,thread_count=3,
                                 random_seed=20261112,allow_writing_files=False,cat_features=CATEGORICAL)
        model.fit(x.iloc[dev[a]],y[a])
        original[b]=model.predict_proba(x.iloc[dev[b]])[:,1]
        marginal[b]=original[b]
        q=b[eligible[dev[b]]]
        pos=mapping.loc[dev[q]].to_numpy()
        hypotheses=np.column_stack([model.predict_proba(part.iloc[pos])[:,1] for part in filled])
        marginal[q]=(hypotheses*weights[pos]).sum(axis=1)
        delta[q]=ref.logit(marginal[q])-ref.logit(original[q])
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(eligible_dev=int(eligible[dev].sum()),eligible_test=int(eligible[8693:].sum()),
                eligible_hard=int(eligible[dev][hard].sum()))
    for name,base in [("direct_tree",original),("raw",bases["rawMitra"]),("v96",load_v96(dev,folds))]:
        p=base.copy()
        q=eligible[dev]
        p[q]=expit(ref.logit(base[q])+delta[q])
        result[name]=dict(metrics=ref.score(p,y),paired=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
                          fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/unknown_cabin_mixture/result.json",result)
    print(result)


if __name__ == "__main__":
    run()
