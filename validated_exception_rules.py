"""Find observable error-prone cohorts and verify each rule on separate training parties."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betainc

import relational_error_corrector as ref
from peer_bill_correction import BILLS,load_v96

ROOT=Path(__file__).resolve().parent


def risk_groups(frame):
    x=frame[["HomePlanet","CryoSleep","Destination","Deck","Side","VIP"]].copy()
    x["age"]=pd.cut(frame.Age,[-np.inf,12,18,35,60,np.inf]).astype(str)
    bills=frame[BILLS]
    for name in BILLS:
        x[name]=pd.cut(bills[name],[-np.inf,0,250,1000,np.inf]).astype(str)
    x["services_bought"]=bills.gt(0).sum(axis=1).clip(upper=3).astype(str)
    x["dominant"]=bills.fillna(0).idxmax(axis=1).where(bills.gt(0).any(axis=1),"none")
    x["position"]=pd.cut(frame.CabinNumber/200.,[-np.inf,1,3,6,np.inf]).astype(str)
    return x.reset_index(drop=True).astype(str)


def run():
    inp,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=inp["dev"],inp["y"],bases["rawMitra"]
    x=risk_groups(inp["features"].iloc[dev])
    groups=np.asarray(inp["groups"]).astype(int)
    choices=list(itertools.combinations(x.columns,2))
    backbone=load_v96(dev,folds)
    outputs={k:p>=.5 for k,p in [("raw",raw),("v96",backbone)]}
    selected=[]
    for fold,(a,b) in enumerate(splits):
        inner=ref.load_inner(fold,a,b,dev,y,evidence)
        train=x.iloc[a].copy()
        train["direction"]=inner>=.5
        train["error"]=train.direction.ne(y[a])
        train["partition"]=groups[a]%2
        tests={k:x.iloc[b].assign(direction=p[b]>=.5) for k,p in [("raw",raw),("v96",backbone)]}
        active={k:np.zeros(len(b),bool) for k in tests}
        count=0;rules=[]
        for left,right in choices:
            keys=[left,right,"direction"]
            s=train.groupby(keys+["partition"]).error.agg(["size","sum"]).unstack("partition",fill_value=0)
            n=s["size"];e=s["sum"]
            prob=betainc(n-e+1,e+1,.5)
            keep=((n.to_numpy()>=8)&(prob>.8)).all(axis=1)
            idx=s.index[keep]
            count+=len(idx)
            for v in idx:
                rules.append(dict(fields=[left,right],values=[str(v[0]),str(v[1])],direction=bool(v[2])))
            if not len(idx):
                continue
            for name,test in tests.items():
                active[name] |= pd.MultiIndex.from_frame(test[keys]).isin(idx)
        for name in outputs:
            outputs[name][b]^=active[name]
        selected.append(dict(fold=fold,rules=count,query_flips={k:int(v.sum()) for k,v in active.items()},rule_definitions=rules))
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result=dict(folds=selected)
    for name,p in [("raw",raw),("v96",backbone)]:
        q=outputs[name]
        result[name]=dict(accuracy=float(np.mean(q==y)),vs_base=ref.pair(q,p,y),hard546=ref.pair(q,p,y,hard),
            fold_net=[ref.pair(q[b],p[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/validated_exception_rules/result.json",result)
    print({k:v for k,v in result.items() if k!="folds"})
    print("Rules per fold", [d["rules"] for d in selected])


if __name__=="__main__":
    run()
