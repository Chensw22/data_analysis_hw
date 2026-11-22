"""Sparse variational GP classification for whole-booking passenger validation."""
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from torch import nn

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write


class SparseGP(nn.Module):
    def __init__(self,centers):
        super().__init__()
        self.register_buffer("centers",centers)
        self.register_buffer("identity",torch.eye(len(centers),dtype=centers.dtype))
        self.scale=nn.Parameter(torch.zeros(centers.shape[1],dtype=centers.dtype))
        self.amplitude=nn.Parameter(torch.tensor(.5,dtype=centers.dtype))
        self.location=nn.Parameter(torch.zeros(len(centers),dtype=centers.dtype))
        self.lower=nn.Parameter(torch.zeros_like(self.identity))
        self.log_diag=nn.Parameter(torch.zeros_like(self.location))
        self.bias=nn.Parameter(torch.zeros((),dtype=centers.dtype))

    def kernel(self,a,b):
        length=self.scale.clamp(-2,2).exp()
        dist=torch.cdist(a/length,b/length).square()
        return self.amplitude.clamp(-2,3).exp()*torch.exp(-dist/2)

    def forward(self,x):
        covariance=self.kernel(self.centers,self.centers)
        chol=torch.linalg.cholesky(covariance+.0005*self.identity)
        A=torch.linalg.solve_triangular(chol,self.kernel(self.centers,x),upper=False).T
        diag=self.log_diag.clamp(-5,3)
        variational_chol=torch.tril(self.lower,diagonal=-1)+torch.diag(diag.exp())
        mean=self.bias+A@self.location
        prior_var=self.amplitude.clamp(-2,3).exp()
        var=(prior_var-A.square().sum(1)+(A@variational_chol).square().sum(1)).clamp_min(1e-8)
        kl=(variational_chol.square().sum()+self.location.square().sum()-len(diag)-2*diag.sum())/2
        return mean,var,kl


def fit(x,y,t,q):
    dtype=torch.float64
    x_t=torch.tensor(x,dtype=dtype)
    target=torch.from_numpy(y.astype(float))
    centers,_=kmeans_plusplus(x[t],n_clusters=80,random_state=674)
    model=SparseGP(torch.tensor(centers,dtype=dtype))
    opt=torch.optim.Adam(model.parameters(),lr=.018)
    nodes,w=np.polynomial.hermite.hermgauss(12)
    nodes=torch.tensor(np.sqrt(2)*nodes,dtype=dtype)
    w=torch.tensor(w/np.sqrt(np.pi),dtype=dtype)
    rng=np.random.default_rng(161)
    for _ in range(42):
        for ids in np.array_split(rng.permutation(t),int(np.ceil(len(t)/512))):
            mu,v,kl=model(x_t[ids])
            logits=mu[:,None]+torch.sqrt(v)[:,None]*nodes
            cost=nn.functional.softplus(logits)-target[ids,None]*logits
            loss=(cost@w).mean()+kl/len(t)+.00015*model.scale.square().sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            opt.step()
    with torch.no_grad():
        mu,v,_=model(x_t[q])
        p=torch.sigmoid(mu[:,None]+torch.sqrt(v)[:,None]*nodes)@w
        return p.numpy(),v.numpy(),float(model.amplitude.clamp(-2,3).exp())


def run():
    torch.set_num_threads(2)
    row,y,b,bank,_,_,_=panel()
    basehits=(b>=.5)==y
    hits=(bank>=.5)==y[None,:,None]
    same=(hits.mean(1)>=basehits.mean(1)[:,None]-.01).all(0)
    weak=~basehits.any(0)&hits.all(0).any(1)&~hits.all(0)[:,same].any(1)
    deep=~basehits.any(0)&~hits.any(axis=(0,2))
    assert (weak.sum(),deep.sum())==(335,252)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                   for s in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0]
    size=np.log1p(group.map(group.value_counts()).iloc[row].to_numpy())
    group=group.iloc[row].to_numpy()
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    out=ART/"nov23_variational_gp"
    out.mkdir(exist_ok=True)
    records={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        prob=np.full(len(y),np.nan)
        var=prob.copy()
        fraction=prob.copy()
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            _,phi=design(frame,t)
            x=np.column_stack([phi,size])
            prob[q],var[q],prior_variance=fit(x,y,t,q)
            fraction[q]=var[q]/prior_variance
            print(seed,"fold",f,flush=True)
        assert np.isfinite([prob,var,fraction]).all() and (var>0).all()
        results={}
        for name,score in [("gp",prob),("quarter",.75*base+.25*prob),
                           ("variance_blend",base+(prob-base)*.4/(1+fraction))]:
            hit=(score>=.5)==y
            original=(base>=.5)==y
            results[name]=dict(accuracy=float(hit.mean()),fixed=int((hit&~original).sum()),
                               harmed=int((~hit&original).sum()),weak335=int(hit[weak].sum()),deep252=int(hit[deep].sum()))
        records[seed]=dict(metrics=results,latent_variance_median=float(np.median(var)),
                           error_rank_auc_from_variance=float(roc_auc_score((prob>=.5)!=y,fraction)))
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,p=prob,
                            latent_variance=var,fraction=fraction)
        write(ROOT/"reports/nov23_variational_gp/result.json",dict(results=records))
        print(seed,results,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
