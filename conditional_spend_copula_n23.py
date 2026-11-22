"""Combine hurdle probabilities, adaptive marginal bins, and conditional Gaussian copulas."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit,ndtri
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from grouped_sparse_rules_n23 import remaining
from relational_error_corrector import write

BILLS=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]


def rank_map(ref,values):
    result=np.full_like(values,np.nan)
    seen=np.isfinite(values)
    if len(ref)>0:
        left=np.searchsorted(ref,values[seen],side="left")
        right=np.searchsorted(ref,values[seen],side="right")
        result[seen]=ndtri((.5+.5*(left+right))/(1+len(ref)))
    return result


def copula_density(z,corr,include_amounts):
    if not include_amounts:
        z=z[:,5:]
        corr=corr[5:,5:]
    masks=np.isfinite(z)
    unique,inverse=np.unique(masks,axis=0,return_inverse=True)
    result=np.zeros(len(z))
    for k,mask in enumerate(unique):
        if mask.sum()<2:
            continue
        selected=inverse==k
        sub=corr[np.ix_(mask,mask)]
        delta=np.linalg.inv(sub)-np.eye(mask.sum())
        part=z[selected][:,mask]
        result[selected]=-.5*(np.linalg.slogdet(sub)[1]+np.sum((part@delta)*part,axis=1))
    return result


def fit_block(data,y,t,q):
    evidence=np.zeros((len(q),2))
    thresholds=[]
    for j in range(5):
        present=data[t,j]
        present=present[np.isfinite(present)]
        thresholds.append(np.unique(np.quantile(present,np.linspace(0,1,13)[1:-1])) if len(present) else [])
    for cls in [False,True]:
        tr=data[t[y[t]==cls]]
        qt=data[q]
        train=np.zeros_like(tr)
        test=np.zeros_like(qt)
        marginal=np.zeros(len(q))
        for j in range(data.shape[1]):
            ref=np.sort(tr[:,j][np.isfinite(tr[:,j])])
            train[:,j]=rank_map(ref,tr[:,j]);test[:,j]=rank_map(ref,qt[:,j])
            if j<5:
                # Conditional positive-amount bin mass; zero/missing states are in the classifier.
                bins=thresholds[j]
                n=np.bincount(np.searchsorted(bins,ref),minlength=len(bins)+1)
                mass=(n+2)/(n.sum()+2*len(n))
                observed=np.isfinite(qt[:,j])
                marginal[observed]+=np.log(mass[np.searchsorted(bins,qt[observed,j])])
        corr=np.eye(data.shape[1])
        for i in range(len(corr)):
            for j in range(i):
                keep=np.isfinite(train[:,i])&np.isfinite(train[:,j]);n=keep.sum()
                if n>=8 and np.std(train[keep,i])>1e-7 and np.std(train[keep,j])>1e-7:
                    corr[i,j]=corr[j,i]=n/(n+70)*np.corrcoef(train[keep,i],train[keep,j])[0,1]
        val,vec=np.linalg.eigh(corr)
        corr=(vec*np.maximum(val,.15))@vec.T
        sd=np.sqrt(np.diag(corr))
        corr=corr/np.outer(sd,sd)
        # Condition on known age and berth instead of inserting their likelihood twice.
        dependence=copula_density(test,corr,True)-copula_density(test,corr,False)
        evidence+=(1 if cls else -1)*np.column_stack([marginal,dependence])
    return evidence


def run():
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=rows.row.to_numpy(),rows.target.to_numpy(bool)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Transported","Name"]).iloc[dev]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    x=compact_features(raw).reset_index(drop=True)
    f=x.drop(columns=BILLS).copy()
    for j in BILLS:
        f[j+"_state"]=np.select([x[j].isna(),x[j]>0],[-1,1],default=0)
    positive=x[BILLS].where(x[BILLS]>0)
    coords=np.column_stack([np.log1p(positive),x.Age,x.CabinNumber])
    positive_count=positive.notna().sum(axis=1).to_numpy()
    strata=x.HomePlanet.astype(str).to_numpy()+np.where(positive_count<=1,"_single","_multiple")
    hard=remaining(dev,y)
    output=ART/"nov23_spend_copula"
    output.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            fold,base=ref["fold"],ref["base"]
            verify(ref,dev,y,fold)
        probs={name:np.full(len(y),np.nan) for name in ["state_prior","independent","conditional_copula","base_dependence"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            fit=CatBoostClassifier(iterations=450,depth=5,learning_rate=.05,thread_count=2,
                                   random_seed=219,verbose=False,allow_writing_files=False)
            fit.fit(f.iloc[t],y[t],cat_features=CATEGORICAL)
            prior=fit.predict_proba(f.iloc[q])[:,1]
            evidence=fit_block(coords,y,t,q)
            for key in np.unique(strata[q]):
                inside=strata[q]==key
                support=t[strata[t]==key]
                if min(np.sum(y[support]),np.sum(~y[support]))<40:
                    continue
                # Partially pool small local cohorts toward the whole training distribution.
                n=len(support);a=n/(n+350)
                evidence[inside]=(1-a)*evidence[inside]+a*fit_block(coords,y,support,q[inside])
            odds=logit(prior.clip(.001,.999))
            probs["state_prior"][q]=prior
            probs["independent"][q]=expit(odds+evidence[:,0])
            probs["conditional_copula"][q]=expit(odds+evidence.sum(axis=1))
            probs["base_dependence"][q]=expit(logit(base[q].clip(.001,.999))+.25*evidence[:,1])
        ok_base=(base>=.5)==y
        metrics={}
        for name,p in probs.items():
            assert np.isfinite(p).all()
            for label,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                ok=(score>=.5)==y
                metrics[label]=dict(accuracy=float(ok.mean()),fixed=int((ok&~ok_base).sum()),
                    harmed=int((~ok&ok_base).sum()),hard356_correct=int(ok[hard].sum()))
        result[str(seed)]=metrics
        np.savez_compressed(output/f"{seed}.npz",row=dev,fold=fold,y=y,base=base,**probs)
        print(seed,metrics,flush=True)
    write(ROOT/"reports/nov23_spend_copula/result.json",dict(results=result,
        note="Per-class empirical normal scores use positive amounts; correlation shrinkage, PSD regularization and conditioning on observed age/berth. Prior distinguishes missing/zero/positive."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
