"""Test if three disjoint sets of anchor parties independently support the same correction."""
import numpy as np
import pandas as pd
from scipy.special import expit

import passenger_graph_diffusion as graph

ref=graph.ref
ROOT=graph.ROOT


def run():
    inputs,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=inputs["dev"],inputs["y"],bases["rawMitra"]
    frame=inputs["features"].iloc[dev].reset_index(drop=True)
    groups=np.asarray(inputs["groups"])
    partitions=groups.astype(int)%3
    shifts=np.zeros((len(dev),3))
    for outer,(a,b) in enumerate(splits):
        matrix,_=graph.topology(frame,a,groups)
        inner=ref.load_inner(outer,a,b,dev,y,evidence)
        anchors=np.column_stack([y[a]-inner,inner*(1-inner)])
        edges=matrix[b][:,b]
        for k in range(3):
            subset=partitions[a]==k
            # Compensate for using one third of the anchors, with identical regularization.
            direct=3*(matrix[b][:,a[subset]]@anchors[subset])
            moments=direct.copy()
            for _ in range(50):
                update=direct+.65*(edges@moments)
                if np.max(np.abs(update-moments))<1e-8:
                    break
                moments=update
            shifts[b,k]=moments[:,0]/(.15+moments[:,1])
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result={}
    for name,base in [("raw",raw),("v96",graph.load_v96(dev,folds))]:
        guesses=expit(ref.logit(base)[:,None]+shifts)
        unanimous=np.all((guesses>=.5)!=(base[:,None]>=.5),axis=1)
        p=np.where(unanimous,guesses.mean(axis=1),base)
        result[name]=dict(metrics=ref.score(p,y),vs_base=ref.pair(p,base,y),hard546=ref.pair(p,base,y,hard),
                          challenged=int(unanimous.sum()),fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/passenger_graph_consensus/result.json",result)
    print(result)


if __name__=="__main__":
    run()
