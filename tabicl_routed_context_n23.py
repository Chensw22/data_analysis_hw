"""Evaluate local attribute-routed contexts for frozen TabICLv2 on held-out bookings."""
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
import torch
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHT = ART / "pretrained/tabicl-classifier-v2-20260212.ckpt"
DIGEST = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
SEEDS = [20261112,20261115,20261116]
CAT = ["HomePlanet","CryoSleep","Destination","VIP","Deck","Side"]
BILLS = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]


def route_features(frame,t):
    encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(frame.iloc[t][CAT])
    bill=frame[BILLS].to_numpy(float)
    return np.column_stack([encoder.transform(frame[CAT]), np.log1p(np.nan_to_num(bill))/6,
                            np.isnan(bill),frame.Age.fillna(28)/40,frame.CabinNumber.fillna(850)/1000])


def infer(x,y,s,q,seed):
    model=TabICLClassifier(n_estimators=2,model_path=WEIGHT,allow_auto_download=False,device="cpu",
                           batch_size=1,n_jobs=2,random_state=seed,kv_cache=False,use_amp=False,use_fa3=False)
    model.fit(x.iloc[s],y[s])
    return model.predict_proba(x.iloc[q])[:,1]


def hard_masks(row,y):
    panel=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    np.testing.assert_array_equal(panel.row,row)
    np.testing.assert_array_equal(panel.target,y)
    old=panel.v131_wrong.to_numpy(bool)&(panel.number_new_correct.to_numpy()==0)
    newer=np.ones(len(row),bool)
    for seed in (20261112,20261115,20261116):
        for folder,keys in {
            "group_kernel":["base","fixed","learned_twice"],
            "boarding_sequence":["sequence"],"projection_shapes":["projection_shapes"],
            "peer_attr_surprise":["self_peer_surprise"],"manifest_tree":["manifest_tree"],
            "group_disputes":["shallow_tree","two_level_table"]}.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                np.testing.assert_array_equal(row,z["row"])
                np.testing.assert_array_equal(y,z["y"])
                for key in keys:
                    wrong=(z[key]>=.5)!=y
                    if folder=="group_kernel":
                        old &= wrong
                    else:
                        newer &= wrong
    # A convex blend cannot disagree with both agreeing binary endpoints.
    newer &= old
    assert old.sum()==447 and newer.sum()==356
    return old,newer


def run(seed):
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        row,fold,y,base=(z[k] for k in ("row","fold","y","base"))
    roster=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[row].reset_index(drop=True)
    group=roster.PassengerId.str.split("_").str[0].to_numpy()
    x=roster.drop(columns=["Cabin","PassengerId"]).copy()
    x[["Deck","CabinNumber","Side"]]=roster.Cabin.str.split("/",expand=True)
    x.CabinNumber=pd.to_numeric(x.CabinNumber)
    x[CAT]=x[CAT].fillna("__MISSING__").astype(str)
    hard,unresolved=hard_masks(row,y)
    out=ART/"nov23_tabicl_context"
    out.mkdir(exist_ok=True)
    modes={name:np.full(len(row),np.nan) for name in ["random","local","global_context"]}
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not set(group[t])&set(group[q])
        modes["global_context"][q]=infer(x,y,t,q,43+int(k))
        g=route_features(x,t)
        partition=KMeans(n_clusters=4,random_state=43,n_init=8).fit(g[t])
        distance=partition.transform(g[t])
        chosen=partition.predict(g[q])
        rng=np.random.default_rng(43+int(k))
        for cluster in range(4):
            query=q[chosen==cluster]
            if not len(query):
                continue
            # Sample from a neighborhood to avoid collecting only central/typical rows.
            neighbors=t[np.argsort(distance[:,cluster],kind="stable")[:1440]]
            for name,frame in [("random",t),("local",neighbors)]:
                support=np.sort(rng.choice(frame,960,replace=False))
                modes[name][query]=infer(x,y,support,query,43+int(k))
            print(int(k),cluster,len(query),flush=True)
        np.savez_compressed(out/f"{seed}_fold{int(k)}.npz",row=row[q],y=y[q],**{n:p[q] for n,p in modes.items()})
    summary={}
    variants={**modes,"global_local":.5*(modes["local"]+modes["global_context"]),"global_plus_base":.5*(modes["global_context"]+base),"local_plus_base":.5*(modes["local"]+base),"random_plus_base":.5*(modes["random"]+base)}
    for name,p in variants.items():
        assert np.isfinite(p).all()
        ok=(p>=.5)==y
        old=(base>=.5)==y
        summary[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
            hard447_correct=int(ok[hard].sum()),hard356_correct=int(ok[unresolved].sum()))
    np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**variants)
    report=ROOT/f"reports/nov23_tabicl_context/{seed}.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(dict(seed=seed,checkpoint=DIGEST,results=summary),indent=2)+"\n")
    print(summary,flush=True)


def reconcile():
    panels={name:[] for name in ["random","local","global_context","global_local","global_plus_base","local_plus_base"]}
    for seed in SEEDS:
        with np.load(ART/f"nov23_tabicl_context/{seed}.npz") as z:
            if seed==SEEDS[0]:
                row,y=z["row"],z["y"]
                hard,unresolved=hard_masks(row,y)
            else:
                np.testing.assert_array_equal(z["row"],row)
                np.testing.assert_array_equal(z["y"],y)
            for name in panels:
                panels[name].append((z[name]>=.5)==y)
    summary={}
    for name,checks in panels.items():
        correct=np.asarray(checks).all(0)
        summary[name]=dict(hard447_stably_correct=int((hard&correct).sum()),
            hard356_stably_correct=int((unresolved&correct).sum()),
            hard356_ever_correct=int((unresolved&np.asarray(checks).any(0)).sum()))
    (ROOT/"reports/nov23_tabicl_context/result.json").write_text(json.dumps(
        dict(assignments=SEEDS,stable=summary,checkpoint=DIGEST,
             notes="Same riders across three partitions. Frozen weights on CPU. KMeans and support selection use only outer training features. Matched and random contexts have 960 rows each and two estimators. Global control uses all training rows."),
        indent=2)+"\n")
    print(summary,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        for seed in SEEDS:
            run(seed)
        reconcile()
