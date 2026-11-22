"""Test conditional reconstruction residuals of individual and companion attributes on unseen bookings."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from error_overlap_diagnostic import load_authenticated
from exaone_validation_data import CATEGORICAL
from peer_bill_correction import BILLS
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEEDS = [20261112, 20261115, 20261116]


def encode(x):
    # Reconstruct all columns in a block from *other* blocks, so one-hot equivalents
    # of the target and its missingness do not enter its own predictors.
    def categorical(cols):
        return pd.get_dummies(x[cols], dtype=float).to_numpy()
    def numbers(values):
        values = np.asarray(values, float)
        return np.column_stack([np.nan_to_num(values), ~np.isfinite(values)])
    return [categorical(["HomePlanet","Destination"]), categorical(["CryoSleep","VIP"]),
            np.column_stack([categorical(["Deck","Side"]), numbers(x.CabinNumber.to_numpy()/1000)]),
            numbers(x.Age.to_numpy()/40), numbers(np.log1p(x[BILLS])/6)]


def condition_matrices(blocks):
    rng=np.random.default_rng(20261123)
    matrices=[]
    for i in range(len(blocks)):
        x=np.column_stack([v for j,v in enumerate(blocks) if j!=i])
        projection=rng.normal(0, .7, (x.shape[1],96))
        bias=rng.uniform(-np.pi,np.pi,96)
        matrices.append(np.column_stack([x, np.cos(x@projection+bias)]))
    return matrices


def reconstruction(matrices, blocks, groups, dev, train, query):
    # For each outer-training booking use a self-supervised fit excluding the whole
    # booking. The query bookings use the mean of the same two reconstruction fits.
    residual=np.zeros((len(groups),sum(b.shape[1] for b in blocks)))
    for fit,cal in GroupKFold(2).split(train, groups=groups[dev[train]]):
        rows=dev[train[fit]]
        cal_members=np.flatnonzero(np.isin(groups,groups[dev[train[cal]]]))
        query_members=np.flatnonzero(np.isin(groups,groups[dev[query]]))
        assert not np.intersect1d(groups[rows],groups[np.r_[cal_members,query_members]]).size
        indices=np.r_[cal_members,query_members]
        values=[]
        for matrix,block in zip(matrices,blocks):
            reg=Ridge(alpha=8).fit(matrix[rows],block[rows])
            values.append(block[indices]-reg.predict(matrix[indices]))
        diff=np.column_stack(values)
        diff[len(cal_members):]*=.5
        residual[indices]+=diff
    return residual


def peer_means(values,groups):
    frame=pd.DataFrame(values)
    num=pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    total=frame.groupby(groups).transform("sum").to_numpy()
    mean=(total-values)/np.maximum(1,num-1)[:,None]
    mean[num==1]=0
    return mean


def run():
    inp,_,_,_,_=load_authenticated()
    x=inp["features"].reset_index(drop=True)
    dev,y=inp["dev"],inp["y"].astype(bool)
    groups=pd.concat([pd.read_csv(ROOT/f"data/raw/{split}.csv",usecols=["PassengerId"]) for split in ["train","test"]],
                     ignore_index=True).PassengerId.str.split("_").str[0].to_numpy()
    np.testing.assert_array_equal(groups[dev],inp["groups"])
    inv=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    np.testing.assert_array_equal(inv.row,dev)
    np.testing.assert_array_equal(inv.target,y)
    unresolved=(inv.v131_wrong & inv.number_new_correct.eq(0)).to_numpy(copy=True)
    seeds={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],dev)
            np.testing.assert_array_equal(z["y"],y)
            for k in ["base","fixed","learned_twice"]:
                unresolved &= (z[k]>=.5)!=y
            seeds[seed]=(z["fold"],z["base"])
    assert unresolved.sum()==447
    blocks=encode(x)
    cond=condition_matrices(blocks)
    raw=np.column_stack(blocks)
    size=pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    control=pd.concat([x,pd.DataFrame(peer_means(raw,groups)).add_prefix("peer_obs_")],axis=1)
    control["booking_size"]=size
    out=ART/"nov23_peer_attr_surprise"
    out.mkdir(exist_ok=True)
    results={}
    for seed,(fold,base) in seeds.items():
        pred={k:np.full(len(y),np.nan) for k in ["raw_peers","self_peer_surprise"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(groups[dev[t]],groups[dev[q]]).size
            res=reconstruction(cond,blocks,groups,dev,t,q)
            extra=np.column_stack([res,peer_means(res,groups)])
            rich=pd.concat([control,pd.DataFrame(extra).add_prefix("surprise_")],axis=1)
            for name,matrix in [("raw_peers",control),("self_peer_surprise",rich)]:
                model=CatBoostClassifier(iterations=480,depth=5,learning_rate=.045,l2_leaf_reg=6,
                    random_seed=20261123,thread_count=3,verbose=False,allow_writing_files=False)
                model.fit(matrix.iloc[dev[t]],y[t],cat_features=CATEGORICAL)
                pred[name][q]=model.predict_proba(matrix.iloc[dev[q]])[:,1]
        pred.update({key+"_blend":.75*base+.25*p for key,p in list(pred.items())})
        def metrics(p,ref):
            assert np.isfinite(p).all()
            hit=(p>=.5)==y
            correct=(ref>=.5)==y
            return dict(accuracy=float(hit.mean()),repaired=int((hit&~correct).sum()),
                        harmed=int((~hit&correct).sum()),unresolved447_correct=int((hit&unresolved).sum()),
                        hard546_correct=int((hit&inv.historical_546).sum()))
        results[str(seed)]={name:metrics(p,base) for name,p in pred.items()}
        results[str(seed)]["surprise_vs_raw"]=metrics(pred["self_peer_surprise"],pred["raw_peers"])
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,p=pred["self_peer_surprise"],**pred)
        print(seed,results[str(seed)],flush=True)
    write(ROOT/"reports/nov23_peer_attr_surprise/result.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
