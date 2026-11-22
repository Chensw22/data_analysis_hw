"""Discover nonlinear manifest formulas from training residuals, with a search-budget-matched control."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conditional_calibration_n23 import offsets
from exaone_validation_data import CATEGORICAL,compact_features
from group_crossfit_residual_discovery_n23 import teachers
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,ROOT,SEEDS,read_panel
from relational_error_corrector import write

OPS={"plus":lambda a,b:(a+b)/2,"difference":lambda a,b:(a-b)/2,"product":np.multiply,
     "max":np.maximum,"min":np.minimum,"distance":lambda a,b:np.abs(a-b)/2}


def terminals(frame,t):
    cols={}
    for c in frame:
        s=frame[c]
        if c in CATEGORICAL:
            for v in sorted(s.iloc[t].unique()):
                cols[f"{c}={v}"]=s.eq(v).to_numpy(float)
        else:
            values=s.to_numpy(float)
            if np.isnan(values).any():
                cols[c+"_missing"]=np.isnan(values).astype(float)
            if c in ["RoomService","FoodCourt","Spa","VRDeck","ShoppingMall"]:
                cols[c+"_positive"]=(values>0).astype(float)
                values=np.log1p(values)
            values=np.nan_to_num(values,nan=float(np.nanmedian(values[t])))
            lo,hi=np.quantile(values[t],[.02,.98])
            if hi>lo:
                cols[c]=((values-lo)/(hi-lo)*2-1).clip(-1,1)
    return [(k,v) for k,v in cols.items() if np.std(v[t])>.005]


def discover(atoms,t,labels,initial,mode,seed):
    rng=np.random.default_rng(seed)
    scores=initial.copy()
    chosen=[]
    for _ in range(5):
        resid=labels[t]-expit(scores[t])
        p=expit(scores[t])
        def characterize(term):
            values=term[1][t]
            centered=values-values.mean()
            norm=np.sqrt(np.sum(p*(1-p)*centered**2)+12)
            return abs(resid@centered)/norm
        parents=atoms
        champion=None
        for depth in range(3):
            proposals=[]
            for j in range(72):
                a,v=parents[rng.integers(len(parents))]
                b,w=atoms[rng.integers(len(atoms))]
                op=list(OPS)[rng.integers(len(OPS))]
                proposals.append((f"{op}({a},{b})",OPS[op](v,w)))
            merit=np.array([characterize(term) for term in proposals])
            best=proposals[np.argmax(merit)]
            if champion is None or merit.max()>champion[0]:
                champion=(merit.max(),best)
            # Total number of generated and scored formulas remains identical.
            keep=np.argsort(merit)[-12:] if mode=="evolved" else rng.choice(len(merit),12,replace=False)
            parents=[proposals[i] for i in keep]
        name,vals=champion[1]
        mu,sd=vals[t].mean(),max(vals[t].std(),.05)
        column=(vals-mu)/sd
        step=(resid@column[t])/(np.sum(p*(1-p)*column[t]**2)+12)
        scores+=.65*step*column
        chosen.append((name,column))
    return chosen


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Transported","Name"])
                   for s in ["train","test"]],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0]
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    frame["BookingSize"]=group.map(group.value_counts()).to_numpy()[row]
    group=group.iloc[row].to_numpy()
    out=ART/"nov23_symbolic_residuals"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        probs={k:np.full(len(y),np.nan) for k in ["evolved","random","linear"]}
        formulas=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            pt,pq=teachers(frame,y,group,t,q,seed+int(k))
            with np.load(ART/f"nov23_residual_discovery/{seed}.npz") as check:
                np.testing.assert_allclose(check["teacher"][q],pq.mean(1))
            offset=np.zeros(len(y))
            offset[t]=logit(pt.clip(.001,.999))
            offset[q]=logit(base[q].clip(.001,.999))
            parts=GroupKFold(2,shuffle=True,random_state=131).split(t,groups=group[t])
            predicted={name:[] for name in probs}
            for split,(a,b) in enumerate(parts):
                build,cal=t[a],t[b]
                atoms=terminals(frame,build)
                for mode in probs:
                    selected=atoms if mode=="linear" else discover(atoms,build,y,offset,mode,933+split)
                    if mode!="linear":
                        formulas.append(dict(fold=int(k),split=split,mode=mode,terms=[s[0] for s in selected]))
                    x=np.column_stack([np.ones(len(y))]+[s[1] for s in selected])
                    pp=offsets(x,offset,y,cal,q,np.full(x.shape[1],12.))
                    predicted[mode].append(pp)
            for mode,pp in predicted.items():
                probs[mode][q]=np.mean(pp,axis=0)
        old=(base>=.5)==y
        metrics={}
        for mode,p in probs.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            metrics[mode]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                               persistent1070=int(ok[~hits.any(0)].sum()),unresolved466=int(ok[hard].sum()))
        results[seed]=dict(metrics=metrics,formulas=formulas)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**probs)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_symbolic_residuals/result.json",dict(results=results,
          protocol="Use group-cross-fitted teachers inside outer training. Discover 5 expressions with 3 rounds of 72 proposals on half the training bookings; fit their weights on the other half, swap and average. Teacher predictions inside discovery are cross-fitted on outer training; outer query bookings never participate."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
