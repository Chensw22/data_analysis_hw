"""Carry within-row donor dependence into multiple imputations for whole-booking validation."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS,hard_rows
from cross_booking_twins_n23 import masks
from exaone_validation_data import compact_features,CATEGORICAL
from relational_error_corrector import write

NUM=["Age","RoomService","FoodCourt","Spa","VRDeck","ShoppingMall","CabinNumber"]
CATS=list(CATEGORICAL)
CORE=CATS+NUM


def metadata(x,party,slot):
    observed=x.replace("__MISSING__",np.nan)
    data={"party_size":party.map(party.value_counts()),"slot":slot}
    for col in ["HomePlanet","Deck","Side","CryoSleep"]:
        for value in sorted(observed[col].dropna().unique()):
            yes=(observed[col]==value).astype(float)
            count=observed[col].notna().astype(int)
            n=count.groupby(party).transform("sum")-count
            data[f"other_{col}_{value}"]=(yes.groupby(party).transform("sum")-yes)/n.replace(0,np.nan)
    for col in CORE:
        data[f"missing_{col}"]=observed[col].isna().astype(float)
    return pd.DataFrame(data,index=x.index)


def donor_candidates(frame,meta,g,t):
    complete=~meta.filter(like="missing_").astype(bool).any(axis=1).to_numpy()
    donors=t[complete[t]]
    needed=np.flatnonzero(~complete)
    cats=frame[CATS].replace("__MISSING__",np.nan)
    num=frame[NUM].copy()
    for name in NUM:
        if name not in ["Age","CabinNumber"]:
            num[name]=np.log1p(num[name])
    # Normalize each genuine field using outer-training donors only.
    spread=(num.iloc[donors].quantile(.95)-num.iloc[donors].quantile(.05)).clip(lower=1)
    num=num/spread
    side=meta[[c for c in meta if c.startswith("other_")]].fillna(-1).to_numpy(float)
    structure=np.column_stack([np.log(meta.party_size),np.log(meta.slot)])
    candidates=[]
    weights=[]
    for start in range(0,len(needed),128):
        ids=needed[start:start+128]
        distance=np.zeros((len(ids),len(donors)))
        for c in CATS:
            v=cats[c].to_numpy()
            seen=cats[c].notna().to_numpy()[ids,None]
            distance += seen*(v[ids,None]!=v[donors][None,:])
        for c in NUM:
            v=num[c].to_numpy()
            delta=np.abs(v[ids,None]-v[donors][None,:])
            distance+=np.nan_to_num(delta,nan=0).clip(0,1)
        for j in range(side.shape[1]):
            distance+=.12*np.abs(side[ids,j,None]-side[donors,j][None,:]).clip(0,1)
        for j in range(2):
            distance+=.5*np.abs(structure[ids,j,None]-structure[donors,j][None,:]).clip(0,1)
        distance[g[ids,None]==g[donors][None,:]]=np.inf
        # Break ties with donor position consistently without introducing target information.
        distance+=np.arange(len(donors))*1e-10
        neighbors=np.argpartition(distance,15,axis=1)[:,:16]
        ds=np.take_along_axis(distance,neighbors,axis=1)
        order=np.argsort(ds,axis=1)
        ds=np.take_along_axis(ds,order,axis=1)
        neighbors=np.take_along_axis(neighbors,order,axis=1)
        w=np.exp(-(ds-ds[:,:1])/.25)
        candidates.extend(donors[neighbors])
        weights.extend(w/w.sum(1,keepdims=True))
    return needed,np.asarray(candidates),np.asarray(weights)


def complete_frame(x,meta,needed,donor):
    # All missing fields in a passenger come from one joint donor draw.
    f=x.copy()
    for c in CORE:
        take=meta[f"missing_{c}"].to_numpy(bool)[needed]
        f.loc[needed[take],c]=x[c].to_numpy()[donor[take]]
    return f.join(meta)


def fit_predict(x,y,t,q):
    m=CatBoostClassifier(iterations=450,depth=6,learning_rate=.045,l2_leaf_reg=6,
                         verbose=False,random_seed=211,thread_count=2,allow_writing_files=False)
    m.fit(x.iloc[t],y[t],cat_features=CATS)
    return m.predict_proba(x.iloc[q])[:,1]


def score(p,anchor,y,hard,harder,missing):
    ok=(p>=.5)==y
    bad=(anchor>=.5)!=y
    def delta(mask):
        return dict(rows=int(mask.sum()),correct=int(ok[mask].sum()),
                    fixed=int((ok&bad&mask).sum()),harmed=int((~ok&~bad&mask).sum()))
    return dict(accuracy=float(ok.mean()),all=delta(np.ones(len(ok),bool)),missing=delta(missing),
                complete=delta(~missing),hard447_correct=int(ok[hard].sum()),hard356_correct=int(ok[harder].sum()))


def run():
    catalog=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=catalog.row.to_numpy(),catalog.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{p}.csv",usecols=lambda s:s not in ["Name","Transported"])
                   for p in ["train","test"]],ignore_index=True)
    tickets=raw.PassengerId.str.split("_",expand=True).astype(int)
    x=compact_features(raw)
    meta=metadata(x,tickets[0],tickets[1]).iloc[dev].reset_index(drop=True)
    x=x.iloc[dev].reset_index(drop=True)
    g=tickets[0].to_numpy()[dev]
    missing=meta.filter(like="missing_").astype(bool).any(axis=1).to_numpy()
    hard,harder=hard_rows(dev,y),masks(catalog)["hard356"]
    results={}
    out=ART/"nov23_party_hotdeck"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],dev)
            np.testing.assert_array_equal(z["y"],y)
            anchor,fold=z["base"],z["fold"]
        preds={k:np.full(len(y),np.nan) for k in ["native","nearest","single","multiple"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not set(g[t])&set(g[q])
            needed,candidates,w=donor_candidates(x,meta,g,t)
            assert np.isin(candidates,t).all() and not (g[needed,None]==g[candidates]).any()
            preds["native"][q]=fit_predict(x.join(meta),y,t,q)
            preds["nearest"][q]=fit_predict(complete_frame(x,meta,needed,candidates[:,0]),y,t,q)
            rng=np.random.default_rng(seed+int(k))
            draws=[]
            for _ in range(4):
                choice=np.minimum((rng.random((len(needed),1))>w.cumsum(axis=1)).sum(1),w.shape[1]-1)
                donor=candidates[np.arange(len(needed)),choice]
                draws.append(fit_predict(complete_frame(x,meta,needed,donor),y,t,q))
            preds["single"][q]=draws[0]
            preds["multiple"][q]=np.mean(draws,axis=0)
        metrics={}
        for name,p in preds.items():
            assert np.isfinite(p).all()
            metrics[name]=score(p,anchor,y,hard,harder,missing)
            metrics[name+"_blend"]=score(.25*p+.75*anchor,anchor,y,hard,harder,missing)
        metrics["multiple_vs_native"]=score(preds["multiple"],preds["native"],y,hard,harder,missing)
        results[str(seed)]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=dev,fold=fold,y=y,base=anchor,**preds)
        print(seed,{k:(round(v["accuracy"],5),v["hard356_correct"]) for k,v in metrics.items()},flush=True)
    write(ROOT/"reports/nov23_party_hotdeck/result.json",dict(results=results,draws=4,neighbors=16,
        note="Donor labels unused, donors always complete outer-training passengers in other bookings. Joint hot-deck imputation preserves dependence among simultaneously missing fields. Peer summaries and observed missing flags are identical across candidate and controls."))


def evaluate_missing_policy():
    catalog=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=catalog.row.to_numpy(),catalog.target.to_numpy(bool)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    x=compact_features(raw.iloc[dev]).replace("__MISSING__",np.nan)
    missing=x.isna().any(axis=1).to_numpy()
    hard,harder=hard_rows(dev,y),masks(catalog)["hard356"]
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_party_hotdeck/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],dev)
            np.testing.assert_array_equal(z["y"],y)
            native,multiple,anchor=z["native"],z["multiple"],z["base"]
        conditional=np.where(missing,multiple,native)
        result[str(seed)]={"policy_vs_native":score(conditional,native,y,hard,harder,missing),
                           "policy_blend_vs_anchor":score(.25*conditional+.75*anchor,anchor,y,hard,harder,missing)}
    write(ROOT/"reports/nov23_party_hotdeck/missing_only.json",result)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
        evaluate_missing_policy()
