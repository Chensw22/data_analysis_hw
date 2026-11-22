"""Infer persistent changes in event risk along booking and cabin order using only training outcomes."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from group_crossfit_residual_discovery_n23 import teachers
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,ROOT,SEEDS,read_panel
from relational_error_corrector import write


AMPLITUDE=np.repeat([.25,.6,1.],3)
LENGTH=np.tile([4.,20.,100.],3)


def smooth(locations, high, low):
    # high/low are vectors of emission likelihood ratios to an independent teacher.
    pos=np.full(len(AMPLITUDE),.5)
    keep=np.empty_like(high)
    evidence=np.zeros_like(pos)
    correlation=np.exp(-np.diff(locations[:,None],axis=0)/LENGTH)
    for i in range(len(locations)):
        if i:
            pos=.5+correlation[i-1]*(pos-.5)
        total=pos*high[i]+(1-pos)*low[i]
        pos=pos*high[i]/total
        keep[i]=pos
        evidence+=np.log(total)
    # Incorporate right-hand observations without conditioning twice on the current node.
    posterior=keep.copy()
    for i in range(len(locations)-2,-1,-1):
        stay=(1+correlation[i])/2
        prior=.5+correlation[i]*(keep[i]-.5)
        posterior[i]=keep[i]*(stay*posterior[i+1]/prior+(1-stay)*(1-posterior[i+1])/(1-prior))
    return evidence,posterior


def state_effect(axis, coord, train, query, target, p, permute):
    total=np.zeros(len(AMPLITUDE))
    estimated=np.zeros((len(coord),len(AMPLITUDE)))
    present=np.isfinite(coord)
    train_mask=np.zeros(len(coord),bool)
    train_mask[train]=True
    rng=np.random.default_rng(6921)
    for band in np.unique(axis[present]):
        ids=np.flatnonzero((axis==band)&present)
        positions,nodes=np.unique(coord[ids],return_inverse=True)
        if permute:
            nodes=rng.permutation(len(positions))[nodes]
        use=train_mask[ids]
        # Only read observed outcomes and teacher probabilities inside outer training.
        r=(target[ids[use]]-p[ids[use]])[:,None]
        high=np.zeros((len(positions),len(AMPLITUDE)))
        low=np.zeros_like(high)
        np.add.at(high,nodes[use],np.log1p(r*AMPLITUDE))
        np.add.at(low,nodes[use],np.log1p(-r*AMPLITUDE))
        evidence,post=smooth(positions,np.exp(high),np.exp(low))
        total+=evidence
        estimated[ids]=(2*post[nodes]-1)*AMPLITUDE
    # Half the prior mass goes to the independent model, which changes no predictions.
    weight=softmax(np.r_[0,total-np.log(len(total))])
    return estimated[query]@weight[1:],float(weight[0])


def run():
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Name","Transported"])
                   for s in ["train","test"]],ignore_index=True)
    groups=raw.PassengerId.str.split("_").str[0]
    g=groups.iloc[rows].to_numpy()
    x=compact_features(raw.iloc[rows]).reset_index(drop=True)
    x["BookingSize"]=groups.map(groups.value_counts()).iloc[rows].to_numpy()
    cab=raw.Cabin.iloc[rows].str.split("/",expand=True)
    axes={"booking":(np.repeat("all",len(y)),g.astype(float)),
          "cabin":((cab[0]+"/"+cab[2]).fillna("unknown").to_numpy(),pd.to_numeric(cab[1]).to_numpy())}
    out=ART/"nov23_ordered_states"
    out.mkdir(exist_ok=True)
    summary={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        probs={name+suffix:base.copy() for name in axes for suffix in ["","_permuted"]}
        masses={}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(g[t],g[q]).size
            pt,pq=teachers(x,y,g,t,q,seed+int(k))
            point=np.full(len(y),np.nan)
            point[t]=pt.clip(.001,.999)
            with np.load(ART/f"nov23_residual_discovery/{seed}.npz") as check:
                np.testing.assert_allclose(pq.mean(1),check["teacher"][q])
            for name,(axis,coordinate) in axes.items():
                for shuffled in [False,True]:
                    key=name+("_permuted" if shuffled else "")
                    adjustment,mass=state_effect(axis,coordinate,t,q,y,point,shuffled)
                    probs[key][q]+=adjustment*base[q]*(1-base[q])
                    masses.setdefault(key,[]).append(mass)
        original=(base>=.5)==y
        metrics={}
        for name,p in probs.items():
            assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            hit=(p>=.5)==y
            metrics[name]=dict(accuracy=float(hit.mean()),fixed=int((hit&~original).sum()),
                               harmed=int((~hit&original).sum()),persistent1070=int(hit[~hits.any(0)].sum()),
                               unresolved466=int(hit[hard].sum()),posterior_no_state=masses[name])
        summary[seed]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**probs)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_ordered_states/result.json",dict(results=summary,
          definition="Two latent risk regimes with exchangeable signs; amplitudes 0.25/0.6/1 times p(1-p), correlation lengths 4/20/100, integrated likelihood with 50% prior mass on independence. No outcomes from query bookings enter either filter."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
