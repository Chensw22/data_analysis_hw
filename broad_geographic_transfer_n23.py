"""Test whole-population correction rules under held-out booking intervals and cabin regions."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS
from broad_dispute_leaf_evidence_n23 import cohorts
from exaone_validation_data import compact_features,CATEGORICAL
from relational_error_corrector import write

REPORT=ROOT/"reports/nov23_broad_geographic_transfer"
OUT=ART/"nov23_broad_geographic_transfer"
BILLS=["Spa","FoodCourt","VRDeck","RoomService","ShoppingMall"]


def load():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    g=raw.PassengerId.str.split("_").str[0].astype(int)
    berth=raw.Cabin.str.split("/",expand=True)
    # Select the most frequent observed deck-side for the entire booking;
    # sort ties deterministically. All members are assigned together.
    cabin=pd.DataFrame({"group":g,"region":berth[0]+"-"+berth[2]}).dropna()
    counts=cabin.value_counts(["group","region"]).rename("n").reset_index().sort_values(
        ["group","n","region"],ascending=[True,False,True])
    region=g.map(counts.drop_duplicates("group").set_index("group").region).fillna("missing")
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        rows,y=z["row"],z["y"]
    frame=compact_features(raw.iloc[rows]).reset_index(drop=True)
    frame["BookingSize"]=g.map(g.value_counts()).to_numpy()[rows]
    return rows,y,frame,g.to_numpy()[rows],region.to_numpy()[rows]


def assign(ids,g,region,kind,n,seed):
    if kind=="random":
        out=np.full(len(ids),-1)
        for k,(_,b) in enumerate(GroupKFold(n,shuffle=True,random_state=seed).split(ids,groups=g[ids])):
            out[b]=k
    elif kind=="order":
        keys,freq=np.unique(g[ids],return_counts=True)
        block=np.minimum((n*(np.cumsum(freq)-freq/2)/len(ids)).astype(int),n-1)
        out=block[np.searchsorted(keys,g[ids])]
    else:
        labels,counts=np.unique(region[ids],return_counts=True)
        assert len(labels)>=n
        order=np.lexsort((labels,-counts))
        total=np.zeros(n,int)
        mapping={}
        for j in order:
            k=int(np.argmin(total))
            mapping[labels[j]]=k
            total[k]+=counts[j]
        out=np.array([mapping[v] for v in region[ids]])
    assert len(np.unique(out))==n
    for k in range(n):
        a,b=ids[out!=k],ids[out==k]
        assert not np.intersect1d(g[a],g[b]).size
        if kind=="cabin":
            assert not np.intersect1d(region[a],region[b]).size
    return out


def opinions(frame,y,t,q):
    views=[list(frame.columns),
           [n for n in frame.columns if n not in ("Deck","CabinNumber","Side")],
           [n for n in frame.columns if n not in BILLS]]
    result=[]
    for cols in views:
        model=CatBoostClassifier(iterations=245,learning_rate=.08,depth=6,thread_count=2,
                                 verbose=False,random_seed=542,allow_writing_files=False)
        model.fit(frame.iloc[t][cols],y[t].astype(int),cat_features=[n for n in CATEGORICAL if n in cols])
        result.append(model.predict_proba(frame.iloc[q][cols])[:,1])
    return logit(np.stack(result,axis=1).clip(.002,.998))


def design(logits):
    first=logits[:,0]
    diff=logits[:,1:]-first[:,None]
    return np.column_stack([np.ones(len(first)),first,diff,diff*np.tanh(first[:,None]/2)])


def fit_weight(environments,mode):
    if mode=="random_readout":
        environments=[v for v in environments if v[0]=="random"]
    penalty=np.array([2.,8.,12.,12.,20.,20.])
    cache=[(design(p),p[:,0],y,np.logaddexp(0,p[:,0])-y*p[:,0]) for _,p,y in environments]
    def objective(coef):
        loss=[]
        grads=[]
        for x,ref,y,previous_loss in cache:
            z=ref+x@coef
            loss.append(np.mean(np.logaddexp(0,z)-y*z-previous_loss))
            grads.append((x.T@(expit(z)-y))/len(y))
        loss=np.asarray(loss)
        if mode=="robust_readout":
            # Optimize the smooth maximum of *extra* log loss in each domain.
            # Comparing against the reference removes intrinsic domain difficulty.
            weights=softmax(loss/.01)
            risk=.01*(logsumexp(loss/.01)-np.log(len(loss)))
        else:
            weights=np.full(len(loss),1/len(loss))
            risk=weights@loss
        scale=sum(len(c[2]) for c in cache)/len(cache)
        return scale*risk+.5*penalty@(coef**2),scale*(weights@np.stack(grads))+penalty*coef
    m=minimize(objective,np.zeros(6),method="L-BFGS-B",jac=True)
    if not m.success:
        raise RuntimeError(m.message)
    return m.x


def fold_fit(frame,y,g,region,t,q,seed):
    assert not np.intersect1d(g[t],g[q]).size
    env=[]
    for kind in ("random","order","cabin"):
        folds=assign(t,g,region,kind,2,seed)
        for h in range(2):
            a,b=t[folds!=h],t[folds==h]
            env.append((kind,opinions(frame,y,a,b),y[b]))
    p=opinions(frame,y,t,q)
    pred={"full":expit(p[:,0]),"no_cabin":expit(p[:,1]),"no_bills":expit(p[:,2])}
    for mode in ("random_readout","mixed_readout","robust_readout"):
        pred[mode]=expit(p[:,0]+design(p)@fit_weight(env,mode))
    return pred


def summary(y,bank,reference,fold,masks):
    before=(reference>=.5)==y
    out={}
    for name,score in bank.items():
        assert np.isfinite(score).all()
        good=(score>=.5)==y
        fixed=good&~before
        harmed=~good&before
        out[name]=dict(accuracy=float(good.mean()),fixed=int(fixed.sum()),harmed=int(harmed.sum()),
            net_by_fold=[int(fixed[fold==f].sum()-harmed[fold==f].sum()) for f in np.unique(fold)],
            right_in_historical_cohorts={k:int((good&m).sum()) for k,m in masks.items()})
    return out


def run():
    rows,y,frame,g,region=load()
    ids=np.arange(len(rows))
    masks=cohorts(rows,y)
    OUT.mkdir(exist_ok=True)
    report={}
    regular_hits={}
    plans=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(rows,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            plans.append((str(seed),z["fold"],z["base"]))
    for kind in ("order","cabin"):
        plans.append((kind,assign(ids,g,region,kind,3,1829),None))
    for name,fold,strong in plans:
        bank={}
        for k in np.unique(fold):
            t,q=ids[fold!=k],ids[fold==k]
            for label,p in fold_fit(frame,y,g,region,t,q,1829+int(k)).items():
                bank.setdefault(label,np.full(len(y),np.nan))[q]=p
            print(name,"fold",int(k),"complete",flush=True)
        results={"vs_full":summary(y,bank,bank["full"],fold,masks),
                 "held_sizes":[int((fold==k).sum()) for k in np.unique(fold)],
                 "held_booking_ranges":[[int(g[fold==k].min()),int(g[fold==k].max())] for k in np.unique(fold)]}
        if strong is not None:
            for key in ("random_readout","mixed_readout","robust_readout"):
                delta=logit(bank[key].clip(.002,.998))-logit(bank["full"].clip(.002,.998))
                bank[key+"_transfer"]=expit(logit(strong.clip(.002,.998))+delta)
            results["vs_strong"]=summary(y,bank,strong,fold,masks)
            for k,p in bank.items():
                regular_hits.setdefault(k,[]).append((p>=.5)==y)
        report[name]=results
        np.savez_compressed(OUT/f"{name}.npz",row=rows,y=y,fold=fold,**bank)
        write(REPORT/"result.json",dict(plans=report))
    stable={n:{k:int((np.logical_and.reduce(h)&m).sum()) for k,m in masks.items()} for n,h in regular_hits.items()}
    write(REPORT/"result.json",dict(plans=report,stable_regular=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
