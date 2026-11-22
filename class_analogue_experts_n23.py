"""Fit local discriminative boundaries around class-specific manifest prototypes."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

BILLS=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]


def design(frame,t):
    categories=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(frame.iloc[t][CATEGORICAL])
    b=frame[BILLS].to_numpy(float)
    numeric=np.nan_to_num(b)
    log=np.log1p(numeric)
    total=numeric.sum(axis=1)
    x=np.column_stack([categories.transform(frame[CATEGORICAL]),np.isnan(b),numeric>0,log/6,
                       frame.Age.fillna(28)/35,frame.Age.isna(),frame.CabinNumber.fillna(850)/1100,
                       frame.CabinNumber.isna(),np.log1p(total)/7,
                       (numeric[:,1]+numeric[:,2]+10)/(total+20)])
    # Local slopes can represent a change of spending response or age trend inside a neighborhood.
    phi=np.column_stack([x,np.maximum(log-6.9,0)/2,frame.Age.fillna(28).lt(13)])
    return x,phi


def soft_experts(x,phi,y,t,q,conditional):
    if conditional:
        sub=[t[y[t]==cls] for cls in [False,True]]
    else:
        sub=[t]
    centers=np.vstack([KMeans(n_clusters=20//len(sub),n_init=2,random_state=384).fit(x[part]).cluster_centers_
                       for part in sub])
    train_dist=cdist(centers,x[t],metric="sqeuclidean")
    query_dist=cdist(x[q],centers,metric="sqeuclidean")
    values,means=[],[]
    for d in train_dist:
        pos=np.argpartition(d,479)[:480]
        near=t[pos]
        radius=max(d[pos].max(),1.e-6)
        weight=np.exp(-d[pos]/radius)
        mean=(weight@y[near]+1)/(weight.sum()+2)
        means.append(float(mean))
        if y[near].min()==y[near].max():
            values.append(np.full(len(q),mean))
        else:
            fit=LogisticRegression(C=.3,solver="lbfgs",max_iter=250).fit(phi[near],y[near],sample_weight=weight)
            values.append(fit.predict_proba(phi[q])[:,1])
    # Use the same three closest centers and weights for the constant and sloped models.
    use=np.argpartition(query_dist,2,axis=1)[:,:3]
    ix=np.arange(len(q))[:,None]
    dist=query_dist[ix,use]
    w=np.exp(-(dist-dist.min(1)[:,None])/max(np.median(np.min(train_dist,axis=0)),.1))
    w/=w.sum(1)[:,None]
    return (w*np.asarray(means)[use]).sum(1),(w*np.asarray(values).T[ix,use]).sum(1)


def predict(frame,group,y,t,q):
    assert not np.intersect1d(group[t],group[q]).size
    x,phi=design(frame,t)
    global_prob=LogisticRegression(C=.3,max_iter=300).fit(phi[t],y[t]).predict_proba(phi[q])[:,1]
    out={"global":global_prob}
    for name,flag in [("class",True),("plain",False)]:
        out[name+"_mean"],out[name+"_linear"]=soft_experts(x,phi,y,t,q,flag)
    distances=cdist(x[q],x[t],"sqeuclidean")
    ix=np.argpartition(distances,30,axis=1)[:,:31]
    out["knn"]=y[t[ix]].mean(1)
    return out


def weight(logical,alt,y):
    def loss(w):
        p=np.clip((1-w)*logical+w*alt,1e-6,1-1e-6)
        return -np.sum(y*np.log(p)+(1-y)*np.log1p(-p))+2*w*w
    return minimize_scalar(loss,bounds=(0,1),method="bounded").x


def run():
    catalog=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    y,old,history=read_panel()
    np.testing.assert_array_equal(y,catalog.target)
    rows=catalog.row.to_numpy()
    stubborn=~old.any(0)
    unresolved=stubborn & ~history.any(axis=(0,1))
    assert stubborn.sum()==1070 and unresolved.sum()==466
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[rows]
    frame=compact_features(raw).reset_index(drop=True)
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    results={}
    dest=ART/"nov23_class_analogues"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,rows,y,fold)
        outer={}
        weights=[]
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            inner_g=np.full(len(t),np.nan)
            inner_l=inner_g.copy()
            for a,b in GroupKFold(3,shuffle=True,random_state=234).split(t,groups=group[t]):
                o=predict(frame,group,y,t[a],t[b])
                inner_g[b],inner_l[b]=o["global"],o["class_linear"]
            assert np.isfinite(inner_g).all() and np.isfinite(inner_l).all()
            w=weight(inner_g,inner_l,y[t])
            weights.append(float(w))
            o=predict(frame,group,y,t,q)
            o["nested"]=(1-w)*o["global"]+w*o["class_linear"]
            for n,v in o.items():
                outer.setdefault(n,np.full(len(y),np.nan))[q]=v
        predictions=dict(outer,nested_plus_base=.75*base+.25*outer["nested"])
        result={}
        for n,p in predictions.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            bad=(base>=.5)!=y
            result[n]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),harmed=int((~bad&~ok).sum()),
                           previous1070_correct=int(ok[stubborn].sum()),previous466_correct=int(ok[unresolved].sum()))
        np.savez_compressed(dest/f"{seed}.npz",row=rows,y=y,fold=fold,p=outer["nested"],base=base,**predictions)
        results[str(seed)]=dict(results=result,local_weights=weights)
        print(seed,{k:(round(v["accuracy"],5),v["previous466_correct"]) for k,v in result.items()},flush=True)
    write(ROOT/"reports/nov23_class_analogues/result.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
