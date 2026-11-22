"""Bounded expression search using held-booking incremental log-loss, with matched controls."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["OMP_NUM_THREADS"]="2"

from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ROOT, ART, SEEDS, hard_rows
from cross_booking_twins_n23 import masks
from exaone_validation_data import compact_features, CATEGORICAL
from relational_error_corrector import write

BILLS = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]


def expressions(x,g):
    vals = {}
    columns = {c:x[c].to_numpy(float) for c in BILLS}
    total = x[BILLS].sum(axis=1,min_count=5)
    for left,right in combinations(BILLS,2):
        a,b = columns[left],columns[right]
        vals[f"{left}_share_{right}"]=(1+a)/(2+a+b)
        vals[f"{left}_both_{right}"]=np.log1p(a)*np.log1p(b)
        vals[f"{left}_diff_{right}"]=np.sign(a-b)*np.log1p(np.abs(a-b))
    for name,series in [*x[BILLS].items(),("total",total),("age",x.Age)]:
        for context in ["HomePlanet","Deck","Destination","CryoSleep"]:
            median=series.groupby(x[context]).transform("median")
            vals[f"{name}_vs_{context}"]=np.log1p(series)-np.log1p(median)
            vals[f"{name}_rank_{context}"]=series.groupby(x[context]).rank(pct=True)
        peers=series.groupby(g).transform("sum")-series.fillna(0)
        denom=series.groupby(g).transform("count")-series.notna().astype(int)
        vals[f"{name}_vs_other_party_members"]=np.log1p(series)-np.log1p(peers/denom.replace(0,np.nan))
        if name!="age":
            vals[f"{name}_per_year"]=np.log1p(series)/(1+np.log1p(x.Age))
            vals[f"{name}_cabin_cross"]=np.log1p(series)*x.CabinNumber/1000
            vals[f"{name}_share_total"]=(1+series)/(2+total)
    return pd.DataFrame(vals)


def fit(x,y,t,seed):
    model=CatBoostClassifier(iterations=420,depth=6,learning_rate=.045,l2_leaf_reg=6,
                             random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    return model.fit(x.iloc[t],y[t],cat_features=CATEGORICAL)


def select(x,v,y,g,train):
    assignment=np.full(len(y),-1)
    for k,(_,b) in enumerate(GroupKFold(3,shuffle=True,random_state=6123).split(train,groups=g[train])):
        assignment[train[b]]=k
    deltas=[]
    for k in range(3):
        a,b,c=[np.flatnonzero(assignment==(k+j)%3) for j in range(3)]
        assert not (set(g[a])&set(g[b]) or set(g[b])&set(g[c]) or set(g[a])&set(g[c]))
        model=fit(x,y,a,155)
        p=model.predict_proba(x.iloc[b])[:,1]
        z=logit(model.predict_proba(x.iloc[c])[:,1])
        def loss(scores):
            return np.mean(np.logaddexp(0.,scores)-y[c]*scores)
        baseline_loss=loss(z)
        improvements=[]
        for _,col in v.items():
            qb=col.to_numpy()[b]
            qc=col.to_numpy()[c]
            present=qb[np.isfinite(qb)]
            if not len(present):
                improvements.append(0.)
                continue
            edges=np.unique(np.quantile(present,[.2,.4,.6,.8]))
            # Reserve bin zero for missing, even when all observed values tie.
            ib=np.where(np.isfinite(qb),1+np.searchsorted(edges,qb),0)
            ic=np.where(np.isfinite(qc),1+np.searchsorted(edges,qc),0)
            num=np.bincount(ib,weights=y[b]-p,minlength=len(edges)+2)
            den=np.bincount(ib,weights=p*(1-p),minlength=len(edges)+2)
            offset=np.clip(num/(den+5),-.8,.8)
            improvements.append(baseline_loss-loss(z+offset[ic]))
        deltas.append(improvements)
    deltas=np.array(deltas)
    scores=deltas.mean(0)
    keep=(scores>0)&((deltas>0).sum(0)>=2)
    # Avoid near-identical candidates using only observed training attributes.
    ranked=v.iloc[train].rank(pct=True).fillna(-1).corr().abs().to_numpy()
    chosen=[]
    for j in np.argsort(-scores,kind="stable"):
        if keep[j] and not any(ranked[j,i]>.97 for i in chosen):
            chosen.append(j)
        if len(chosen)==8:
            break
    return list(v.columns[chosen]), {v.columns[i]:float(scores[i]) for i in chosen}


def run():
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{p}.csv", usecols=lambda c:c not in ["Transported","Name"])
                   for p in ["train","test"]],ignore_index=True)
    party=raw.PassengerId.str.split("_").str[0]
    frame=compact_features(raw)
    x=frame.assign(PartySize=party.map(party.value_counts())).iloc[dev].reset_index(drop=True)
    v=expressions(frame,party).iloc[dev].reset_index(drop=True)
    g=party.to_numpy()[dev]
    hard=hard_rows(dev,y)
    harder=masks(inventory)["hard356"]
    result={}
    out=ART/"nov23_expression_search"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],dev)
            np.testing.assert_array_equal(z["y"],y)
            base,fold=z["base"],z["fold"]
        choices=[]
        pred={n:np.full(len(dev),np.nan) for n in ["raw","random","selected"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not set(g[t]) & set(g[q])
            chosen,scores=select(x,v,y,g,t)
            random=np.random.default_rng(seed+int(k)).choice(v.columns.difference(chosen),len(chosen),replace=False).tolist()
            for n,cols in [("raw",[]),("selected",chosen),("random",random)]:
                m=x.join(v[cols])
                pred[n][q]=fit(m,y,t,155).predict_proba(m.iloc[q])[:,1]
            choices.append(dict(fold=int(k),selected=chosen,random=random,inner_gain=scores))
        summary={}
        for n,p in pred.items():
            assert np.isfinite(p).all()
            for label,score in [(n,p),(n+"_blend",.75*base+.25*p)]:
                ok=(score>=.5)==y
                old=(base>=.5)==y
                summary[label]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((old&~ok).sum()),
                                    hard447_correct=int(ok[hard].sum()),hard356_correct=int(ok[harder].sum()))
        result[str(seed)]=dict(metrics=summary,choices=choices)
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**pred)
        print(seed,summary,flush=True)
    write(ROOT/"reports/nov23_expression_search/result.json",dict(results=result,expression_count=len(v.columns),
          note="Outer rows/groups never participate in incremental-utility evaluation or selection. Label-free manifest aggregates use the complete roster."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
