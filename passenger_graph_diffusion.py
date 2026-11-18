"""Diffuse cross-fitted residual evidence through a mutual-neighbor passenger graph."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

import relational_error_corrector as ref
from peer_bill_correction import BILLS,load_v96

ROOT=Path(__file__).resolve().parent
CATS=["HomePlanet","CryoSleep","Destination","VIP","Deck","Side"]


def topology(frame,train,groups):
    cats=OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float32).fit(frame.iloc[train][CATS])
    numeric=frame[["Age","CabinNumber",*BILLS]].copy()
    # A positive expenditure and its percentile carry different information.
    positive=numeric[BILLS].gt(0).to_numpy(np.float32)
    median=numeric.iloc[train].median()
    transform=QuantileTransformer(n_quantiles=256,subsample=None,random_state=20261118)
    transform.fit(numeric.iloc[train].fillna(median))
    coords=np.column_stack([cats.transform(frame[CATS])*.7,transform.transform(numeric.fillna(median)),
                            positive,numeric.isna().to_numpy()*.5]).astype(np.float32)
    # Fixed ordering is shared across folds. Exclude all party members before taking 24 nearest neighbors.
    neighbors=NearestNeighbors(n_neighbors=33,n_jobs=2).fit(coords)
    distances,indices=neighbors.kneighbors(coords)
    ok=groups[indices]!=groups[:,None]
    take=(ok.cumsum(axis=1)<=24)&ok
    if not np.all(take.sum(axis=1)==24):
        raise ValueError("Insufficient non-party neighbors")
    idx=indices[take].reshape(len(frame),24)
    dist=distances[take].reshape(len(frame),24)
    local=np.maximum(dist[:,-1],1e-6)
    weights=np.exp(-dist**2/(local[:,None]*local[idx]))
    rows=np.repeat(np.arange(len(frame)),24)
    matrix=csr_matrix((weights.ravel(),(rows,idx.ravel())),shape=(len(frame),len(frame)))
    matrix=matrix.minimum(matrix.T)
    degree=np.asarray(matrix.sum(axis=1)).ravel()
    return diags(1/np.where(degree>0,degree,1))@matrix,degree


def run():
    inputs,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=inputs["dev"],inputs["y"],bases["rawMitra"]
    frame=inputs["features"].iloc[dev].reset_index(drop=True)
    groups=np.asarray(inputs["groups"])
    shifts={mode:np.zeros(len(y)) for mode in ["one_hop","diffusion"]}
    graph_info=[]
    for fold,(a,b) in enumerate(splits):
        matrix,degrees=topology(frame,a,groups)
        inner=ref.load_inner(fold,a,b,dev,y,evidence)
        anchors=np.column_stack([y[a]-inner,inner*(1-inner)])
        direct=matrix[b][:,a]@anchors
        query_edges=matrix[b][:,b]
        propagated=direct.copy()
        # A longer path is discounted more strongly; anchor values stay clamped.
        for _ in range(50):
            update=direct+.65*(query_edges@propagated)
            if np.max(np.abs(update-propagated))<1e-8:
                break
            propagated=update
        for key,moments in [("one_hop",direct),("diffusion",propagated)]:
            shifts[key][b]=moments[:,0]/(.15+moments[:,1])
        graph_info.append(dict(fold=fold,isolates=int((degrees[b]==0).sum()),edges=int(matrix.nnz//2)))
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(graph=graph_info)
    for name,base in [("raw",raw),("v96",load_v96(dev,folds))]:
        result[name]={}
        for key,shift in shifts.items():
            p=expit(ref.logit(base)+shift)
            result[name][key]=dict(metrics=ref.score(p,y),vs_base=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
                                   fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    np.savez_compressed(ROOT/"artifacts/passenger_graph_diffusion.npz",dev=dev,**shifts)
    ref.write(ROOT/"reports/passenger_graph_diffusion/result.json",result)
    print(result)


if __name__=="__main__":
    run()
