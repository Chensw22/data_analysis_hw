"""Learn from training pairs when a similar passenger's known outcome transfers."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder,QuantileTransformer

import relational_error_corrector as ref
from passenger_graph_diffusion import CATS
from peer_bill_correction import BILLS, load_v96

ROOT=Path(__file__).resolve().parent
NUM=["Age","CabinNumber",*BILLS]


def geometry(frame,train):
    onehot=OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float32)
    category=onehot.fit(frame.iloc[train][CATS]).transform(frame[CATS])
    numeric=frame[NUM]
    filled=numeric.fillna(numeric.iloc[train].median())
    ranks=QuantileTransformer(n_quantiles=256,subsample=None,random_state=20261118).fit(filled.iloc[train]).transform(filled)
    paid=numeric[BILLS].gt(0).to_numpy().astype(float)
    missing=numeric.isna().to_numpy().astype(float)
    coords=np.column_stack([.7*category,ranks,paid,.5*missing]).astype(np.float32)
    return coords,frame[CATS].to_numpy().astype(str),ranks,paid,missing


def neighbors(search,coordinates,queries,training,groups):
    dist,rel=search.kneighbors(coordinates[queries])
    candidates=training[rel]
    distinct=groups[candidates]!=groups[queries,None]
    select=distinct&(np.cumsum(distinct,axis=1)<=16)
    if not np.all(select.sum(axis=1)==16):raise ValueError("Too few independent anchors")
    return candidates[select].reshape(len(queries),16),dist[select].reshape(len(queries),16)


def pair_features(q, anchors, distances, cats, ranks, paid, missing, y):
    i=np.repeat(q,anchors.shape[1])
    j=anchors.ravel()
    # Absolute distances are symmetric. Anchor label and numerical means put those distances in context.
    return np.column_stack([y[j],distances.ravel(),(cats[i]==cats[j]).astype(float),
        np.abs(ranks[i]-ranks[j]),(ranks[i]+ranks[j])*.5,
        np.abs(paid[i]-paid[j]),np.abs(missing[i]-missing[j])]).astype(np.float32)


def run():
    inp,splits,folds,bases,_=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"]
    groups=np.asarray(inp["groups"])
    frame=inp["features"].iloc[dev]
    learned=np.empty(len(y));voter=np.empty(len(y))
    for fold,(a,b) in enumerate(splits):
        coords,cats,ranks,paid,missing=geometry(frame,a)
        search=NearestNeighbors(n_neighbors=25,n_jobs=2).fit(coords[a])
        anchors,dist=neighbors(search,coords,a,a,groups)
        features=pair_features(a,anchors,dist,cats,ranks,paid,missing,y)
        target=(y[a,None]==y[anchors]).ravel()
        predictor=CatBoostClassifier(iterations=240,depth=5,learning_rate=.07,l2_leaf_reg=10,
            thread_count=2,random_seed=20261118+fold,verbose=False,allow_writing_files=False)
        predictor.fit(features,target)
        anchors,dist=neighbors(search,coords,b,a,groups)
        features=pair_features(b,anchors,dist,cats,ranks,paid,missing,y)
        agreement=predictor.predict_proba(features)[:,1].reshape(len(b),16)
        learned[b]=np.where(y[anchors]==1,agreement,1-agreement).mean(axis=1)
        voter[b]=(1+y[anchors].sum(axis=1))/18
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    base=load_v96(dev,folds)
    result={}
    for name,p in [("neighbor_vote",voter),("learned_agreement",learned),("half_v96",(base+learned)/2)]:
        result[name]=dict(metrics=ref.score(p,y),vs_v96=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
            hard_correct=int(((p[hard]>=.5)==y[hard]).sum()),fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/passenger_analogy/result.json",result)
    np.savez_compressed(ROOT/"artifacts/passenger_analogy.npz",dev=dev,learned=learned,voter=voter)
    print(result)


if __name__=="__main__":
    run()
