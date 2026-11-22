"""Learn which passengers differ within bookings, conditioning out the unknown party intercept."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from projection_shapes_n23 import smooth_basis
from relational_error_corrector import write


def partitions(groups,y,shuffle=False):
    members=pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
    remap=np.arange(len(y))
    if shuffle:
        # Randomize passenger membership under the same group sizes and label
        # counts so the amount of conditional training evidence is identical.
        rng=np.random.default_rng(520)
        for label in (False,True):
            indices=np.flatnonzero(y==label)
            remap[indices]=rng.permutation(indices)
    buckets={}
    for p in members:
        ids=remap[p]
        k=int(y[ids].sum())
        if 0<k<len(ids):
            buckets.setdefault((len(ids),k),[]).append(ids)
    result=[]
    for (n,k),ids in buckets.items():
        states=np.zeros((len(list(combinations(range(n),k))),n))
        for i,idx in enumerate(combinations(range(n),k)):
            states[i,list(idx)]=1
        result.append((np.stack(ids),states))
    return result


def fit(x,y,groups,kind):
    yy=y.astype(float)
    blocks=[] if kind=="ordinary" else partitions(groups,y,shuffle=kind=="shuffled")
    n_cond=sum(ids.size for ids,_ in blocks)
    # Individual likelihood identifies the level for singletons and entirely
    # concordant parties. The auxiliary loss has the same total row weight.
    weight=len(y)/max(n_cond,1)
    ridge=np.full(x.shape[1],4.)
    ridge[0]=.02
    def objective(beta):
        logits=x@beta
        loss=(np.logaddexp(0,logits)-yy*logits).sum()
        residual=expit(logits)-yy
        for ids,states in blocks:
            scores=logits[ids]@states.T
            loss+=weight*(logsumexp(scores,axis=1)-(logits[ids]*yy[ids]).sum(1)).sum()
            residual[ids]+=weight*(softmax(scores,axis=1)@states-yy[ids])
        return loss+.5*((beta**2)*ridge).sum(), x.T@residual+ridge*beta
    result=minimize(objective,np.zeros(x.shape[1]),method="L-BFGS-B",jac=True,options={"maxiter":450})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, n_cond


def run():
    row,y,base,_,_,_,_=panel()
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported"))
    frame=compact_features(raw.iloc[row])
    booking=raw.PassengerId.str.split("_").str[0]
    group=booking.to_numpy()[row]
    size=booking.map(booking.value_counts()).to_numpy()[row]
    repeat=~((base>=.5)==y).any(0)
    out=ART/"nov23_broad_conditional_counts"
    out.mkdir(exist_ok=True)
    hit={}
    counts={}
    for i,seed in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            row_check,labels,fold=z["row"],z["y"],z["fold"]
        np.testing.assert_array_equal(row_check,row)
        np.testing.assert_array_equal(labels,y)
        pred={k:np.full(len(y),np.nan) for k in ("ordinary","conditional","shuffled","transfer")}
        counts[seed]=[]
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            _,phi=design(frame,t)
            phi=np.column_stack([phi,np.log(size)])
            a,b=smooth_basis(phi[t],phi[q])
            a,b=np.column_stack([np.ones(len(t)),a]),np.column_stack([np.ones(len(q)),b])
            for kind in ("ordinary","conditional","shuffled"):
                beta,n=fit(a,y[t],group[t],kind)
                pred[kind][q]=expit(b@beta)
                if kind=="conditional":
                    counts[seed].append(n)
                if kind=="shuffled":
                    assert n==counts[seed][-1]
        delta=logit(pred["conditional"].clip(.001,.999))-logit(pred["ordinary"].clip(.001,.999))
        pred["transfer"]=expit(logit(base[i].clip(.001,.999))+delta)
        for k,p in pred.items():
            assert np.isfinite(p).all()
            hit.setdefault(k,[]).append((p>=.5)==y)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base[i],**pred)
        print(seed,{k:round(float(((p>=.5)==y).mean()),4) for k,p in pred.items()},flush=True)
    ref=(base>=.5)==y
    result={}
    for k,v in hit.items():
        good=np.stack(v)
        result[k]={}
        for label,m in {"all":np.ones(len(y),bool),"solo":size==1,"accompanied":size>1,"ever":~ref.all(0),
                        "persistent":repeat}.items():
            result[k][label]=dict(rows=int(m.sum()),accuracy=good[:,m].mean(1).tolist(),
                 fixed=(good&~ref)[:,m].sum(1).tolist(),harmed=(~good&ref)[:,m].sum(1).tolist(),
                 stable_correct=int((m&good.all(0)).sum()))
    write(ROOT/"reports/nov23_broad_shared_passenger_modes/conditional_counts.json",
          dict(results=result,mixed_party_training_rows=counts))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
