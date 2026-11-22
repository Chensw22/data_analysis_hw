"""Fit Laplace-kernel outcome surfaces with a training-only gradient-learned metric.

Method reference: Radhakrishnan et al., https://arxiv.org/abs/2212.13881.
We use a fixed landmark approximation and retain an identity component.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from awake_negative_recovery import matrices
from error_overlap_diagnostic import load_authenticated
from exact_fold_error_selector import counts
from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"


def fit_views(x,y,query,seed):
    rng=np.random.default_rng(seed)
    centers=x[rng.choice(len(x),size=min(768,len(x)),replace=False)]
    metric=np.eye(x.shape[1])
    predictions={}
    for iteration in range(3):
        values,vectors=np.linalg.eigh(metric)
        transform=vectors*np.sqrt(np.maximum(values,0))
        center_coords=centers@transform
        distances=cdist(x@transform,center_coords)
        width=np.median(distances[:500])
        kernel=np.exp(-distances/width)
        gram=np.exp(-cdist(center_coords,center_coords)/width)
        system=kernel.T@kernel+.5*gram+1e-5*np.eye(len(gram))
        alpha=solve(system,kernel.T@(y-y.mean()),assume_a="pos")
        predictions[iteration]=np.clip(y.mean()+np.exp(-cdist(query@transform,center_coords)/width)@alpha,.001,.999)
        if iteration<2:
            # Differentiate the fitted kernel expansion; identical landmark coordinates
            # contribute zero. Only training inputs and their fitted coefficients enter.
            weight=np.divide(kernel*alpha,distances,out=np.zeros_like(kernel),where=distances>1e-7)
            grad=((weight@centers-weight.sum(1)[:,None]*x)@metric)/width
            cov=grad.T@grad/len(grad)
            eig,vec=np.linalg.eigh(cov)
            root=(vec*np.sqrt(np.maximum(eig,0)))@vec.T
            metric=.15*np.eye(len(root))+.85*root*(len(root)/max(root.trace(),1e-12))
    return predictions


def run():
    inp,_,_,_,_=load_authenticated()
    dev,y,group=inp["dev"],inp["y"].astype(bool),np.asarray(inp["groups"])
    rows=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    np.testing.assert_array_equal(rows.row,dev)
    np.testing.assert_array_equal(rows.true,y)
    x=inp["features"].iloc[dev]
    raw, _=matrices(x)
    # Group size is observable for the complete unlabeled manifest at inference.
    tickets=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId"]).PassengerId
                       for s in ["train","test"]],ignore_index=True).str.split("_").str[0]
    size=tickets.map(tickets.value_counts()).iloc[dev].to_numpy()
    raw=np.column_stack([raw,x.CabinNumber.isna(),x.CabinNumber.fillna(850)/1000,np.log(size)])
    results={}
    out=ART/"nov23_group_kernel"
    out.mkdir(exist_ok=True)
    for seed in [20261112,20261115,20261116]:
        refs={}
        for family in ["mitra","v44"]:
            with np.load(ART/f"top5_group_phase/refs/{family}/group_{seed}.npz") as z:
                np.testing.assert_array_equal(z["dev_ids"],dev)
                np.testing.assert_array_equal(z["group_ids"].astype(int),group.astype(int))
                if refs:
                    np.testing.assert_array_equal(fold,z["fold_id"])
                fold=z["fold_id"].copy()
                refs[family]=z["prob"].copy()
        base=(refs["mitra"]+refs["v44"])/2
        preds=np.full((len(y),3),np.nan)
        for k in np.unique(fold):
            q=fold==k
            assert not np.intersect1d(group[q],group[~q]).size
            for j,p in fit_views(raw[~q],y[~q].astype(float),raw[q],seed+int(k)).items():
                preds[q,j]=p
        assert np.isfinite(preds).all()
        variants={"base":base,"fixed":preds[:,0],"learned_once":preds[:,1],"learned_twice":preds[:,2],
                  "fixed_blend":.75*base+.25*preds[:,0],"learned_blend":.75*base+.25*preds[:,2]}
        results[str(seed)]={name:counts(p>=.5,base>=.5,y,fold,rows.hard.to_numpy(),rows.persistent.to_numpy())
                            for name,p in variants.items()}
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,**variants)
        print(seed,{n:round(v["accuracy"],5) for n,v in results[str(seed)].items()},flush=True)
    write(ROOT/"reports/nov23_group_kernel/result.json",results)


if __name__=="__main__":
    with threadpool_limits(3):
        run()
