"""Learn which manifest fields define informative analogues for each passenger."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax, logsumexp, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_passenger_transport_n23 import manifest_coordinates
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import REPORT, scopes


def coordinates(frame):
    values, numeric = [], []
    for col in frame.columns:
        v=frame[col]
        if pd.api.types.is_numeric_dtype(v):
            values.append(v.rank(pct=True).fillna(-1).to_numpy())
            numeric.append(True)
        else:
            values.append(pd.factorize(v)[0])
            numeric.append(False)
    return np.column_stack(values),np.array(numeric)


def donor_data(x,numeric,raw,group,query,train):
    rng=np.random.default_rng(731)
    train=rng.permutation(train)
    ids=[]
    for chunk in np.array_split(query, 12):
        gap=cdist(raw[chunk],raw[train],"sqeuclidean")
        gap[group[chunk,None]==group[train]]=np.inf
        # Preserve a few nonlocal alternatives to the equal-field neighborhood.
        perm=rng.random(gap.shape)
        perm[~np.isfinite(gap)]=np.inf
        near=np.argsort(gap,axis=1,kind="stable")[:,:80]
        np.put_along_axis(perm,near,np.inf,axis=1)
        extra=np.argpartition(perm,15,axis=1)[:,:16]
        ids.append(train[np.c_[near,extra]])
    ids=np.vstack(ids)
    assert not (group[query,None]==group[ids]).any()
    a,b=x[query,None,:],x[ids]
    diff=np.where(numeric, np.minimum(abs(a-b),1.)**2, a!=b)
    return np.ascontiguousarray(diff,dtype=float),ids


def odds(z):
    return logit(z.clip(.001,.999))


def metric_fit(h,dist,labels,y):
    n,nref,d=dist.shape
    # The reference donor has fixed weight comparable to a remote analogue and
    # equal mass for both labels. It keeps one-sided neighborhoods finite.
    pseudo=-3.
    same=labels==y[:,None]
    def objective(theta):
        w=theta.reshape(h.shape[1],d)
        weights=softmax(h@w,axis=1)
        score=-24*np.einsum("ijd,id->ij",dist,weights)
        den=np.logaddexp(logsumexp(score,axis=1),pseudo)
        keep=np.where(same,score,-np.inf)
        num=np.logaddexp(logsumexp(keep,axis=1),pseudo-np.log(2))
        ds=np.exp(score-den[:,None])-np.exp(keep-num[:,None])
        dw=-24*np.einsum("ij,ijd->id",ds,dist)
        g=h.T@(weights*(dw-(dw*weights).sum(1,keepdims=True)))
        penalty=8.
        return (den-num).sum()+.5*penalty*(w*w).sum(), (g+penalty*w).ravel()
    sol=minimize(objective,np.zeros(h.shape[1]*d),jac=True,method="L-BFGS-B",
                 options={"maxiter":160,"ftol":1e-8})
    assert sol.success,sol.message
    return sol.x.reshape(h.shape[1],d)


def vote(h,w,dist,labels):
    score=-24*np.einsum("ijd,id->ij",dist,softmax(h@w,axis=1))
    mass=np.exp(score)
    prior=np.exp(-3.)
    return ((mass*labels).sum(1)+.5*prior)/(mass.sum(1)+prior)


def run():
    row,y,frame,g,_=load()
    x,numeric=coordinates(frame)
    raw=manifest_coordinates(frame)
    # The context is estimated from each passenger's own fields. A constant
    # context learns the same set of metric weights for every passenger.
    context=(raw-raw.mean(0))/raw.std(0).clip(.04)
    context=np.column_stack([np.ones(len(y)),context.clip(-4,4)/np.sqrt(context.shape[1])])
    masks=scopes(y,frame)
    history,refs={},[]
    dest=ART/"nov23_passenger_metric"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)): np.testing.assert_array_equal(z[k],v)
            base,fold=z["base"],z["fold"]
        outputs={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            a,nt=donor_data(x,numeric,raw,g,t,t)
            b,nq=donor_data(x,numeric,raw,g,q,t)
            rng=np.random.default_rng(528)
            for mode,train_h,test_h in (
                ("equal",context[t,:1],context[q,:1]),
                ("global",context[t,:1],context[q,:1]),
                ("personal",context[t],context[q]),
                ("shuffled",context[rng.permutation(t)],context[rng.permutation(q)])):
                w=(np.zeros((1,x.shape[1])) if mode=="equal" else metric_fit(train_h,a,y[nt],y[t]))
                outputs.setdefault(mode,np.full(len(y),np.nan))[q]=vote(test_h,w,b,y[nq])
        for mode in list(outputs):
            design=np.c_[odds(base),odds(outputs[mode])]
            cal=np.zeros(len(y))
            for f in np.unique(fold):
                idx=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(idx,groups=g[idx]):
                    t,q=idx[a],idx[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    fit=LogisticRegression(C=.5,max_iter=300).fit(design[t],y[t])
                    cal[q]=fit.predict_proba(design[q])[:,1]
            outputs["combined_"+mode]=cal
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in outputs.items():
            assert np.isfinite(p).all()
            history.setdefault(k,[]).append((p>=.5)==y)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**outputs)
        print(seed,[(k,int((v[-1]&~ref).sum()),int((~v[-1]&ref).sum())) for k,v in history.items()],flush=True)
    write(REPORT/"metric.json",{k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()})


if __name__=="__main__":
    with threadpool_limits(2):
        run()
