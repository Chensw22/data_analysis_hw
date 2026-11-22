"""Test class-conditional log-ratio densities on each observed positive-spending simplex."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.linalg import helmert, solve_triangular
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import CATEGORICAL,compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

BILLS=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]


def design(frame):
    bills=frame[BILLS].to_numpy(float)
    positive=bills>0
    which=positive.astype(int)@(2**np.arange(len(BILLS)))
    spend=np.nansum(bills,axis=1)
    prior=frame.drop(columns=BILLS).copy()
    prior["observed_log_budget"]=np.log1p(spend)
    for j,c in enumerate(BILLS):
        prior[c+"_state"]=np.where(np.isnan(bills[:,j]),-1,positive[:,j].astype(int))
    ratios=np.zeros((len(frame),4))
    for code in np.unique(which):
        ix=np.flatnonzero(which==code)
        active=positive[ix[0]]
        k=active.sum()-1
        if k>0:
            # Orthonormal isometric log-ratio coordinates on this strictly positive face.
            ratios[ix,:k]=np.log(bills[ix][:,active])@helmert(k+1).T
    direct=prior.copy()
    for j in range(4):
        direct[f"balance_{j}"]=ratios[:,j]
    return prior,direct,which,ratios,np.log1p(spend),positive.sum(1)>=2


def mean_fit(h,z,prior,strength):
    return np.linalg.solve(h.T@h+np.eye(h.shape[1])*strength,h.T@z+strength*prior)


def log_density(z,mu,cov):
    chol=np.linalg.cholesky(cov)
    delta=solve_triangular(chol,(z-mu).T,lower=True)
    return -.5*(delta**2).sum(0)-np.log(np.diag(chol)).sum()


def contributions(y,face,ratios,budget,demographics,t,q,seed):
    full=np.zeros(len(q))
    independent=full.copy()
    placebo=full.copy()
    rng=np.random.default_rng(seed)
    for code in np.unique(face[q]):
        k=int(code).bit_count()-1
        if k<1:
            continue
        train=t[face[t]==code]
        pos=np.flatnonzero(face[q]==code)
        if len(train)<12 or len(pos)==0:
            continue
        mu=np.mean(budget[train])
        sd=max(.25,np.std(budget[train]))
        amount=(budget-mu)/sd
        # Shrink class-specific mean-budget curves toward the common face-level fit.
        h=np.column_stack([np.ones(len(y)),amount,np.maximum(amount,0),demographics])
        z=ratios[:,:k]
        b0=mean_fit(h[train],z[train],np.zeros((h.shape[1],k)),2.)
        res=z[train]-h[train]@b0
        global_cov=res.T@res/len(train)+np.eye(k)*.05
        # The placebo keeps the outcome count per face and broad budget stratum.
        copy=y[train].copy()
        block=np.searchsorted(np.quantile(budget[train],[.33,.67]),budget[train])
        for j in np.unique(block):
            rows=np.flatnonzero(block==j)
            copy[rows]=rng.permutation(copy[rows])
        for labels,target in [(y[train],full),(copy,placebo)]:
            for cls,sign in [(True,1),(False,-1)]:
                idx=train[labels==cls]
                fit=mean_fit(h[idx],z[idx],b0,18.)
                resid=z[idx]-h[idx]@fit
                cov=(resid.T@resid+28*global_cov)/(len(idx)+28)+np.eye(k)*.03
                predicted=h[q[pos]]@fit
                target[pos]+=sign*log_density(z[q[pos]],predicted,cov)
                if target is full:
                    independent[pos]+=sign*log_density(z[q[pos]],predicted,np.diag(np.diag(cov)))
    return {"full":full,"diagonal":independent,"label_placebo":placebo}


def learn(x,y,t,q):
    model=CatBoostClassifier(iterations=480,depth=6,learning_rate=.045,l2_leaf_reg=6,
                             thread_count=2,random_seed=345,allow_writing_files=False,verbose=False)
    model.fit(x.iloc[t],y[t],cat_features=CATEGORICAL)
    return model.predict_proba(x.iloc[q])[:,1]


def run():
    panel=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row=panel.row.to_numpy()
    y,old,history=read_panel()
    np.testing.assert_array_equal(panel.target,y)
    stubborn=~old.any(0)
    hard=stubborn&~history.any(axis=(0,1))
    assert stubborn.sum()==1070 and hard.sum()==466
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"]).iloc[row]
    g=raw.PassengerId.str.split("_").str[0].to_numpy()
    x=compact_features(raw).reset_index(drop=True)
    prior,direct,face,ratios,budget,eligible=design(x)
    demo=np.column_stack([x.HomePlanet.eq(p) for p in ["Earth","Europa"]]+
                         [x.Side.eq("P"),x.Age.fillna(28).to_numpy()/40])
    out=ART/"nov23_simplex_likelihood"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as cache:
            base,fold=cache["base"],cache["fold"]
            verify(cache,row,y,fold)
        predictions={k:np.full(len(y),np.nan) for k in ["budget_prior","direct_balances","full","diagonal","label_placebo"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(g[t],g[q]).size
            p=learn(prior,y,t,q)
            predictions["budget_prior"][q]=p
            predictions["direct_balances"][q]=learn(direct,y,t,q)
            for name,delta in contributions(y,face,ratios,budget,demo,t,q,seed+int(k)).items():
                values=expit(logit(p)+delta.clip(-7,7))
                values[~eligible[q]]=p[~eligible[q]]
                predictions[name][q]=values
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,eligible=eligible,**predictions)
    summarize()


def summarize():
    y,old,history=read_panel()
    stubborn=~old.any(0)
    hard=stubborn&~history.any(axis=(0,1))
    results={}
    consistent={}
    candidate_names=["budget_prior","direct_balances","full","diagonal","label_placebo"]
    for seed in SEEDS:
        with np.load(ART/f"nov23_simplex_likelihood/{seed}.npz") as cache:
            predictions={name:cache[name] for name in candidate_names}
            base,eligible=cache["base"],cache["eligible"]
            np.testing.assert_array_equal(y,cache["y"])
        metrics={}
        for name,p in predictions.items():
            assert np.isfinite(p).all()
            for label,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                ok=(score>=.5)==y
                basebad=(base>=.5)!=y
                reference=predictions["budget_prior"] if label==name else .75*base+.25*predictions["budget_prior"]
                priorbad=(reference>=.5)!=y
                metrics[label]=dict(accuracy=float(ok.mean()),fixed=int((ok&basebad).sum()),harmed=int((~ok&~basebad).sum()),
                                    eligible_correct=int((ok&eligible).sum()),fixed_vs_prior_eligible=int((eligible&ok&priorbad).sum()),
                                    harmed_vs_prior_eligible=int((eligible&~ok&~priorbad).sum()),
                                    hard1070_correct=int((ok&stubborn).sum()),hard466_correct=int((ok&hard).sum()))
                consistent.setdefault(label,[]).append(ok)
        results[str(seed)]=metrics
    stable={name:dict(hard1070_all=int((np.all(hits,axis=0)&stubborn).sum()),
                      hard466_all=int((np.all(hits,axis=0)&hard).sum()),
                      hard466_any=int((np.any(hits,axis=0)&hard).sum())) for name,hits in consistent.items()}
    write(ROOT/"reports/nov23_simplex_likelihood/result.json",dict(results=results,cross_split=stable,eligible=int(eligible.sum()),
          total=len(y),eligible1070=int((eligible&stubborn).sum()),eligible466=int((eligible&hard).sum()),
          criterion="Two or more observed strictly positive bills; partial bills retain a separate missing indicator. Densities are fitted only on the same positive-service pattern and omit patterns with fewer than 12 training rows."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
