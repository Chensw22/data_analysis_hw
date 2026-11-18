"""Transfer a residual learned from leave-one-passenger-out party spending to a base prediction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref

ROOT=Path(__file__).resolve().parent
BILLS=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
REPORT=ROOT/"reports/peer_bill_correction"


def context():
    frame=pd.concat([pd.read_csv(ROOT/"data/raw"/name,usecols=["PassengerId","CryoSleep",*BILLS])
                     for name in ["train.csv","test.csv"]],ignore_index=True)
    group=frame.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()
    bills=frame[BILLS]
    # Zero is a reported bill, missing values never count as zero-valued observations.
    observed=bills.notna().astype(int)
    peers=observed.groupby(group).transform("sum")-observed
    peer_sum=bills.fillna(0).groupby(group).transform("sum")-bills.fillna(0)
    positive=bills.gt(0).astype(int)
    peer_paid=(positive.groupby(group).transform("sum")-positive)/peers.replace(0,np.nan)
    peer_log=np.log1p(peer_sum/peers.replace(0,np.nan))
    sleeping=frame.CryoSleep.eq(True).to_numpy().astype(float)
    zero=bills.sum(axis=1,min_count=5).eq(0).to_numpy().astype(float)
    known=(peers>0).any(axis=1).to_numpy() & (size>1)
    base=np.column_stack([np.log1p(size-1),sleeping,zero])
    new=np.column_stack([peer_log,peer_paid])
    full=np.column_stack([base,new,new*sleeping[:,None],new*zero[:,None]])
    return frame.PassengerId,known,base,full


def fit_offset(features,teacher,y):
    med=np.nanmedian(features,axis=0)
    med=np.nan_to_num(med)
    x=np.where(np.isfinite(features),features,med)
    scale=StandardScaler().fit(x)
    weights,_=ref.fit_corrector(teacher,y,scale.transform(x))
    return med,scale,weights


def apply(fitted,features):
    med,scale,w=fitted
    return w[0]+scale.transform(np.where(np.isfinite(features),features,med))@w[1:]


def run():
    data,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=data["dev"],data["y"],bases["rawMitra"]
    passengers,known,base,full=context()
    np.testing.assert_array_equal(passengers.iloc[dev],data["passengers"])
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    result={}
    offsets={}
    for name,design in [("matched",base),("peer_bills",full)]:
        shift=np.zeros(len(y))
        for fold,(a,b) in enumerate(splits):
            teacher=ref.load_inner(fold,a,b,dev,y,evidence)
            eligible=known[dev[a]]
            pack=fit_offset(design[dev[a[eligible]]],teacher[eligible],y[a[eligible]])
            q=b[known[dev[b]]]
            shift[q]=apply(pack,design[dev[q]])
        offsets[name]=shift
    for base_name,probs in [("raw",raw),("v96",load_v96(dev,folds))]:
        if probs is None:
            continue
        adjusted={}
        for name,shift in offsets.items():
            p=probs.astype(float).copy()
            mask=known[dev]
            p[mask]=expit(ref.logit(p[mask])+shift[mask])
            adjusted[name]=p
            result[base_name+"_"+name]=dict(metrics=ref.score(p,y),vs_base=ref.pair(p,probs,y),
                hard546=ref.pair(p,probs,y,hard),fold_net=[ref.pair(p[b],probs[b],y[b])["net_correct"] for _,b in splits])
        result[base_name+"_incremental"]=ref.pair(adjusted["peer_bills"],adjusted["matched"],y)
    np.savez_compressed(ROOT/"artifacts/peer_bill_offsets.npz",dev=dev,**offsets)
    ref.write(REPORT/"result.json",result)
    print(json.dumps(result))


def load_v96(dev,folds):
    path=ROOT/"artifacts/v96_oof_completion/oof.npz"
    if not path.exists():
        return None
    record=json.loads((ROOT/"reports/v96_oof_completion/result.json").read_text())
    assert ref.sha(path)==record["oof_sha256"]
    with np.load(path) as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        np.testing.assert_array_equal(z["fold_id"],folds)
        return z["prob"].copy()


if __name__=="__main__":
    run()
