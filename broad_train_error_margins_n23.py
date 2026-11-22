"""Transfer information about recurring training mistakes without needing a test passenger's label."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_dispute_leaf_evidence_n23 import ART,ROOT,SEEDS,cohorts
from class_analogue_experts_n23 import design
from exaone_validation_data import CATEGORICAL,compact_features
from projection_shapes_n23 import smooth_basis
from relational_error_corrector import write

OUT=ART/"nov23_broad_train_error_margins"
REPORT=ROOT/"reports/nov23_broad_train_error_margins"


def teacher_misses(frame,x,y,groups,t):
    votes=np.full((len(t),6),np.nan)
    for rep,seed in enumerate((749,1369)):
        for a,b in GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=groups[t]):
            fit,cal=t[a],t[b]
            assert not np.intersect1d(groups[fit],groups[cal]).size
            cat=CatBoostClassifier(iterations=270,learning_rate=.07,depth=6,thread_count=2,
                                   verbose=False,allow_writing_files=False,random_seed=417)
            cat.fit(frame.iloc[fit],y[fit].astype(int),cat_features=CATEGORICAL)
            trees=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=5,n_jobs=1,random_state=329).fit(x[fit],y[fit])
            hist=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=2,
                       early_stopping=False,random_state=399).fit(x[fit],y[fit])
            votes[b,rep*3:(rep+1)*3]=np.column_stack([cat.predict_proba(frame.iloc[cal])[:,1],
                  trees.predict_proba(x[cal])[:,1],hist.predict_proba(x[cal])[:,1]])
    assert np.isfinite(votes).all()
    return (votes>=.5)!=y[t,None]


def features(frame,size,t,q):
    x,phi=design(frame,t)
    dense=np.column_stack([phi,np.log(size)])
    a,b=smooth_basis(dense[t],dense[q])
    z=(x-x[t].mean(0))/x[t].std(0).clip(.25)
    rng=np.random.default_rng(212)
    omega=rng.normal(size=(x.shape[1],128))/np.sqrt(x.shape[1])
    trig=np.sqrt(2)*np.cos(z.clip(-5,5)@omega+rng.uniform(0,2*np.pi,128))
    return x,np.column_stack([np.ones(len(t)),a,trig[t]]),np.column_stack([np.ones(len(q)),b,trig[q]])


def optimize(x,y,misses,kind):
    # Counts from six honest teacher votes define disjoint training strata.
    # A held-out passenger never needs to be assigned to one of them.
    counts=misses.sum(1)
    bucket=np.digitize(counts,[1,4,6])
    if kind=="constant":
        bucket=np.zeros(len(y),dtype=int)
    _,bucket=np.unique(bucket,return_inverse=True)
    size=np.bincount(bucket)
    prior=size/len(y)
    sign=2*y.astype(float)-1
    n=x.shape[1]
    bonus=kind in ("margin","constant")
    penalty=np.full(n,3.)
    penalty[0]=.01
    def objective(params):
        beta=params[:n]
        margin=sign*(x@beta)
        if bonus:
            margin=margin+params[n:][bucket]
        losses=np.logaddexp(0,-margin)
        weights=np.ones(len(y))
        loss=losses.sum()
        if kind=="robust":
            group_loss=np.bincount(bucket,weights=losses)/size
            logweights=np.log(prior)+group_loss
            normal=logsumexp(logweights)
            loss=len(y)*normal
            weights=np.exp(group_loss[bucket]-normal)
        gradient=-expit(-margin)*weights
        dbeta=x.T@(gradient*sign)+penalty*beta
        loss+=.5*((beta**2)*penalty).sum()
        if bonus:
            v=params[n:]
            loss+=.5*(size*v**2).sum()
            dbeta=np.r_[dbeta,np.bincount(bucket,weights=gradient)+size*v]
        return loss,dbeta
    init=np.zeros(n+(len(size) if bonus else 0))
    bounds=[(None,None)]*n+([(0,None)]*len(size) if bonus else [])
    optimum=minimize(objective,init,jac=True,method="L-BFGS-B",bounds=bounds,options={"maxiter":1600,"ftol":2e-8})
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return optimum.x[:n],optimum.x[n:].tolist()


def run():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Transported","Name"))
                   for s in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0]
    OUT.mkdir(exist_ok=True)
    hits={}
    refs=[]
    checks={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            row,y,fold,base=[ref[k] for k in ("row","y","fold","base")]
        size=group.map(group.value_counts()).to_numpy()[row]
        gs=group.to_numpy()[row]
        frame=compact_features(raw.iloc[row])
        pred={}
        checks[seed]=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(gs[t],gs[q]).size
            teacher_x,x,z=features(frame,size,t,q)
            errs=teacher_misses(frame,teacher_x,y,gs,t)
            # Preserve the exact six-vote joint pattern and each class's frequency.
            fake=errs.copy()
            rng=np.random.default_rng(605)
            for label in (False,True):
                part=np.flatnonzero(y[t]==label)
                fake[part]=errs[rng.permutation(part)]
            margins={}
            for label,kind,source in (("ordinary","ordinary",errs),("constant","constant",errs),
                                      ("margin","margin",errs),("random_margin","margin",fake),
                                      ("robust","robust",errs),("random_robust","robust",fake)):
                beta,v=optimize(x,y[t],source,kind)
                pred.setdefault(label,np.full(len(y),np.nan))[q]=expit(z@beta)
                if kind=="margin":
                    margins[label]=v
            checks[seed].append(dict(vote_hist=np.bincount(errs.sum(1),minlength=7).tolist(),training_allowance=margins))
        for key,control in [("margin","constant"),("robust","ordinary")]:
            delta=logit(pred[key].clip(.001,.999))-logit(pred[control].clip(.001,.999))
            pred[key+"_transfer"]=expit(logit(base.clip(.001,.999))+delta)
        for name,p in pred.items():
            assert np.isfinite(p).all()
            hits.setdefault(name,[]).append((p>=.5)==y)
        refs.append((base>=.5)==y)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**pred)
        print(seed,{k:round(float(((p>=.5)==y).mean()),4) for k,p in pred.items()},flush=True)
    masks=cohorts(row,y)
    r=np.stack(refs)
    summary={}
    for k,v in hits.items():
        good=np.stack(v)
        summary[k]=dict(accuracy=good.mean(1).tolist(),fixed=(good&~r).sum(1).tolist(),harmed=(~good&r).sum(1).tolist(),
                        stable={n:int((m&good.all(0)).sum()) for n,m in masks.items()})
    write(REPORT/"result.json",dict(results=summary,folds=checks))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
