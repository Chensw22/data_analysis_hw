"""Map whole passenger profiles to a shared reference distribution within registration contexts."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logit
from scipy.stats import rankdata, qmc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS,load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_refit_dissent_n23 import features
from broad_release_miss_coverage_n23 import summarize
from broad_training_membership_n23 import model
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import compact_features

REPORT=ROOT/"reports/nov23_manifest_vector_ranks"
OUT=ART/"nov23_manifest_vector_ranks"


def context_groups(frame):
    # Rare combinations share a parent. All assignments use registration only.
    keys=frame.HomePlanet+"|"+frame.CryoSleep
    pooled=keys.map(keys.value_counts())<96
    keys=keys.mask(pooled,"other-"+frame.HomePlanet)
    keys=keys.mask(keys.map(keys.value_counts())<96,"other")
    return np.unique(keys,return_inverse=True)[1]


def ranks(values):
    out=np.full(values.shape,.5)
    for j,col in enumerate(values.T):
        ok=np.isfinite(col)
        if ok.any():
            out[ok,j]=(rankdata(col[ok])-.5)/ok.sum()
    return out


def map_to_grid(train,query,grid):
    kernel=np.exp(-cdist(train,grid,"sqeuclidean")/.12)
    v=np.ones(len(grid))
    target=len(train)/len(grid)
    for _ in range(500):
        mass=(1./(kernel@v))@kernel*v
        if np.max(abs(mass/target-1))<2.e-7:
            break
        v*=target/mass
        v/=np.exp(np.mean(np.log(v)))
    else:
        raise RuntimeError("unbalanced reference measure")
    k=np.exp(-cdist(query,grid,"sqeuclidean")/.12)*v
    return (k@grid)/k.sum(1,keepdims=True)


def process(frame):
    raw=frame[[*BILLS,"Age","CabinNumber"]].to_numpy(float)
    grid=qmc.Sobol(raw.shape[1],scramble=True,seed=438).random_base2(7)
    ctx=context_groups(frame)
    output={k:np.empty_like(raw) for k in ("marginal","joint","shuffled")}
    for cell in np.unique(ctx):
        ids=np.flatnonzero(ctx==cell)
        z=ranks(raw[ids])
        rng=np.random.default_rng(859+cell)
        shuffled=np.column_stack([rng.permutation(col) for col in z.T])
        output["marginal"][ids]=z
        output["joint"][ids]=map_to_grid(z,z,grid)
        output["shuffled"][ids]=map_to_grid(shuffled,z,grid)
    return output


def calibrate(base,p,y,g,fold):
    design=np.column_stack([logit(v.clip(.001,.999)) for v in (base,p)])
    out=np.empty(len(y))
    for f in np.unique(fold):
        ix=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
            t,q=ix[a],ix[b]
            assert not np.intersect1d(g[t],g[q]).size
            out[q]=LogisticRegression(C=.5,max_iter=400).fit(design[t],y[t]).predict_proba(design[q])[:,1]
    return out


def main():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    roster=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Transported","Name"))
                     for s in ("train","test")],ignore_index=True)
    f=compact_features(roster)
    np.testing.assert_array_equal(roster.PassengerId.str.split("_").str[0].astype(int).to_numpy()[row],g)
    x=features(frame)[0]
    rep=process(f)
    designs={"raw":x}
    for k in rep:
        cols=[x,rep["marginal"][row]]
        if k!="marginal":
            cols.append(rep[k][row])
        designs[k]=np.column_stack(cols)
    history,refs={},[]
    OUT.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold=z["base"],z["fold"]
        scores={}
        for k,design in designs.items():
            out=np.full(len(y),np.nan)
            for f in np.unique(fold):
                t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(g[t],g[q]).size
                out[q]=model(210).fit(design[t],y[t]).predict_proba(design[q])[:,1]
            assert np.isfinite(out).all()
            scores[k]=out
            scores["combined_"+k]=calibrate(base,out,y,g,fold)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**scores)
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in scores.items():
            history.setdefault(k,[]).append((p>=.5)==y)
        print(seed,[(k,int((h[-1]&~ref).sum()),int((~h[-1]&ref).sum())) for k,h in history.items()],flush=True)
    stat={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}
    write(REPORT/"ranks.json",stat)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
