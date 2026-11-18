"""Estimate spatial residual correlation across independent passenger parties."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

import relational_error_corrector as ref
from peer_bill_correction import load_v96

ROOT = Path(__file__).resolve().parent


def smooth(train, query, sigma=16., prior_information=3.):
    offset = pd.Series(0.,index=query.index)
    keys = ["deck","side","sleep"]
    # Collapse related occupants of one party to a single source observation per cabin.
    cells = train.dropna(subset=["number"]).groupby(keys+["number","party"],as_index=False)[["residual","variance"]].mean()
    for key, block in query.dropna(subset=["number"]).groupby(keys):
        neighbors = cells
        for name,value in zip(keys,key):
            neighbors = neighbors.loc[neighbors[name].eq(value)]
        if neighbors.empty:
            continue
        delta = np.abs(block.number.to_numpy()[:,None]-neighbors.number.to_numpy()[None,:])
        distinct = block.party.to_numpy()[:,None] != neighbors.party.to_numpy()[None,:]
        weights = np.exp(-delta/sigma) * distinct
        offset.loc[block.index] = (weights @ neighbors.residual.to_numpy()) / (prior_information+weights @ neighbors.variance.to_numpy())
    return offset.to_numpy()


def run():
    inputs,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    x=inputs["features"].iloc[inputs["dev"]]
    d=pd.DataFrame(dict(deck=x.Deck.to_numpy(),side=x.Side.to_numpy(),sleep=x.CryoSleep.to_numpy(),
                        number=x.CabinNumber.to_numpy(),party=inputs["groups"]))
    dev,y,raw=inputs["dev"],inputs["y"],bases["rawMitra"]
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result={}
    for mode in ["observed","placebo"]:
        shift=np.zeros(len(y))
        for fold,(a,b) in enumerate(splits):
            inner=ref.load_inner(fold,a,b,dev,y,evidence)
            train=d.iloc[a].assign(residual=y[a]-inner,variance=inner*(1-inner))
            if mode=="placebo":
                # Preserve local demography and co-location across peers, break only exact cabin order.
                keys=["deck","side"]
                unique=train.dropna(subset=["number"])[keys+["number"]].drop_duplicates().copy()
                unique["block"]=(unique.number//128).astype(int)
                rng=np.random.default_rng(20261118+fold)
                unique["reordered"]=unique.groupby(keys+["block"]).number.transform(rng.permutation)
                mapping=unique.set_index(keys+["number"]).reordered
                train["number"]=mapping.reindex(pd.MultiIndex.from_frame(train[keys+["number"]])).to_numpy()
            shift[b]=smooth(train,d.iloc[b])
        result[mode]={}
        for name,base in [("raw",raw),("v96",load_v96(dev,folds))]:
            p=expit(ref.logit(base)+shift)
            result[mode][name]=dict(metrics=ref.score(p,y),vs_base=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
                                    fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/nearby_party_residual/result.json",result)
    print(result)


if __name__=="__main__":
    run()
