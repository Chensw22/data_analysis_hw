"""Batch Mondrian partitions with hierarchical empirical class prediction."""
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from broad_dispute_leaf_evidence_n23 import cohorts
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from fresh_family_rescue_n23 import NEW


def grow(t, q, x, y, parent, survival, total, rng, mode):
    if len(q)==0:
        return
    n=len(t)
    p=(y[t].sum()+6*parent)/(n+6)
    if n<=18 or y[t].min()==y[t].max():
        total[q]+=survival*p
        return
    sample=x[t]
    low,high=sample.min(0),sample.max(0)
    span=high-low
    width=span.sum()
    if width<1e-9:
        total[q]+=survival*p
        return
    # Each coordinate of the bounding box has an exponential split clock at
    # rate equal to its extent; the earliest determines time and dimension.
    elapsed=rng.exponential(1/width)
    if mode=="mondrian":
        col=rng.choice(len(span),p=span/width)
    else:
        col=rng.choice(np.flatnonzero(span>0))
    cut=rng.uniform(low[col],high[col])
    in_left=sample[:,col]<=cut
    left=x[q,col]<=cut
    # Competing extensions outside the box use the parent-level estimate.
    distance=(np.maximum(0,x[q]-high)+np.maximum(0,low-x[q])).sum(1)
    remain=np.exp(-elapsed*distance)
    total[q]+=survival*(1-remain)*parent
    weight=survival*remain
    left_seed,right_seed=rng.integers(2**31,size=2)
    grow(t[in_left],q[left],x,y,p,weight[left],total,np.random.default_rng(left_seed),mode)
    grow(t[~in_left],q[~left],x,y,p,weight[~left],total,np.random.default_rng(right_seed),mode)


def forest(x,y,t,q,seed,mode):
    count=64
    total=np.zeros(len(y))
    rng=np.random.default_rng(seed)
    for _ in range(count):
        tree_rng=np.random.default_rng(rng.integers(2**31))
        grow(t,q,x,y,y[t].mean(),np.ones(len(q)),total,tree_rng,mode)
    return total[q]/count


def run():
    row,y,base,old,_,_,_=panel()
    archived=[predictions(row,y,s,dict(**NEW,native_carte=("carte","p"))) for s in SEEDS]
    all_old=(np.concatenate([old,np.stack([a[1] for a in archived])],axis=2)>=.5)==y[None,:,None]
    hard=((base>=.5)!=y).all(0)
    no_stable=hard & ~all_old.all(0).any(1)
    never=hard & ~all_old.any(axis=(0,2))
    previous=cohorts(row,y)["new50"]
    rotation=[]
    for seed,(_,_,fold) in zip(SEEDS,archived):
        with np.load(ART/f"nov23_broad_rotation_forest/{seed}.npz") as z:
            for k,v in [("row",row),("fold",fold),("y",y)]:
                np.testing.assert_array_equal(z[k],v)
            rotation.append((z["rotation"]>=.5)==y)
    previous=previous | (no_stable & np.logical_and.reduce(rotation))
    assert previous.sum()==57
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[row]
    frame=compact_features(raw)
    g=raw.PassengerId.str.split("_").str[0].to_numpy()
    out=ART/"nov23_broad_mondrian_partition"
    out.mkdir(exist_ok=True)
    records,hits={},{}
    for i,(seed,(_,_,fold)) in enumerate(zip(SEEDS,archived)):
        scores={k:np.full(len(y),np.nan) for k in ("mondrian","uniform_axis")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            x,_=design(frame,t)
            x=MinMaxScaler().fit(x[t]).transform(x)
            for mode in list(scores):
                scores[mode][q]=forest(x,y,t,q,3113+int(f),mode)
        for mode in list(scores):
            scores[mode+"_quarter"]=(scores[mode]+3*base[i])/4
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base[i],**scores)
        records[seed]={}
        ref=(base[i]>=.5)==y
        for name,p in scores.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            ok=(p>=.5)==y
            hits.setdefault(name,[]).append(ok)
            records[seed][name]=dict(accuracy=ok.mean(),fixed=int((ok&~ref).sum()),harmed=int((~ok&ref).sum()))
        print(seed,records[seed],flush=True)
    coverage={}
    for name,arr in hits.items():
        ever=np.any(arr,0)
        stable=np.all(arr,0)
        coverage[name]=dict(persistent_stable=int((hard&stable).sum()),new_to_46=int((no_stable&stable).sum()),
                            overlap_with_57=int((previous&stable).sum()),fresh=int((no_stable&~previous&stable).sum()),
                            never_any=int((never&ever).sum()),never_stable=int((never&stable).sum()))
    folder=ROOT/"reports/nov23_broad_mondrian_partition"
    folder.mkdir(exist_ok=True)
    (folder/"result.json").write_text(json.dumps(dict(records=records,coverage=coverage),indent=2)+"\n")


if __name__=="__main__":
    run()
