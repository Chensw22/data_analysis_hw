"""Nonstationary numeric covariance for a finite-rank probabilistic manifest classifier."""
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

BILLS=["RoomService","Spa","VRDeck","FoodCourt","ShoppingMall"]


def coordinates(frame,party_size,t):
    expenses=frame[BILLS]
    numeric=np.column_stack([frame.Age,frame.CabinNumber,np.log1p(expenses),np.log1p(party_size)])
    med=np.nanmedian(numeric[t],axis=0)
    scale=np.diff(np.nanquantile(numeric[t],[.05,.95],axis=0),axis=0)[0].clip(1.)
    continuous=np.where(np.isfinite(numeric),numeric-med,0.)/scale
    flags=np.column_stack([np.isnan(numeric),expenses.gt(0).to_numpy()])
    fields=[frame[c].eq(level).to_numpy(float) for c in CATEGORICAL for level in sorted(frame[c].iloc[t].unique())]
    categorical=np.column_stack([*fields,flags*.5])
    return continuous,categorical


def radii(x,cat,party,t):
    fit=np.column_stack([x[t],cat[t]])
    all_rows=np.column_stack([x,cat])
    result=np.empty(len(x))
    for first in range(0,len(x),128):
        block=slice(first,first+128)
        dist=cdist(all_rows[block],fit,"sqeuclidean")
        dist[party[block,None]==party[t][None,:]]=np.inf
        result[block]=np.sqrt(np.partition(dist,63,axis=1)[:,63])
    # Duplicate manifests may have zero neighbor radius; bandwidth stays positive.
    median=np.median(result[t])
    return np.clip(result,.35*median,2.5*median)


def gp(features,y,t,q):
    prior_variance=6.
    model=LogisticRegression(C=prior_variance,fit_intercept=True,max_iter=400,tol=1e-6)
    model.fit(features[t],y[t])
    train_p=model.predict_proba(features[t])[:,1]
    latent=model.decision_function(features[q])
    # Laplace approximation at the MAP. Intercept has a flat prior; its
    # posterior uncertainty is included with the finite landmark coefficients.
    aug=np.column_stack([features,np.ones(len(features))])
    penalty=np.full(aug.shape[1],1/prior_variance)
    penalty[-1]=0.
    H=aug[t].T@(aug[t]*(train_p*(1-train_p))[:,None])+np.diag(penalty)
    factor=cholesky(H,lower=True)
    var=(solve_triangular(factor,aug[q].T,lower=True)**2).sum(axis=0)
    return expit(latent/np.sqrt(1+np.pi*var/8))


def predict(frame,party,size,y,t,q,seed):
    x,cat=coordinates(frame,size,t)
    r=radii(x,cat,party,t)
    med=np.median(r[t])
    centers=np.random.default_rng(seed).choice(t,512,replace=False)
    d=cdist(x,x[centers],"sqeuclidean")
    cats=cdist(cat,cat[centers],"sqeuclidean")
    dim=x.shape[1]
    out={}
    for name,width in [("global",np.full_like(r,med)),("adaptive",r)]:
        s2=width[:,None]**2+width[centers]**2
        # Isotropic Gibbs covariance on numeric fields times a stationary
        # Gaussian on categorical indicators. The determinant term is needed
        # for a valid covariance when bandwidth varies between passengers.
        K=np.exp(-d/s2-.5*cats)*(2*width[:,None]*width[centers]/s2)**(dim/2)
        gram=K[centers]
        L=cholesky((gram+gram.T)/2+1e-6*np.eye(len(centers)),lower=True)
        phi=solve_triangular(L,K.T,lower=True).T
        out[name]=gp(phi,y,t,q)
    return out


def run():
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row=inventory.row.to_numpy()
    y,old,history=read_panel()
    np.testing.assert_array_equal(inventory.target,y)
    bad_all=~old.any(0)
    hard=bad_all&~history.any(axis=(0,1))
    assert bad_all.sum()==1070 and hard.sum()==466
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Name","Transported"])
                   for s in ["train","test"]],ignore_index=True)
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    ticket=raw.PassengerId.str.split("_").str[0]
    size=ticket.map(ticket.value_counts()).to_numpy()[row]
    party=ticket.to_numpy()[row]
    out=ART/"nov23_gibbs_manifest"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        panel={s:np.full(len(y),np.nan) for s in ["global","adaptive"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(party[t],party[q]).size
            for name,p in predict(frame,party,size,y,t,q,seed+int(k)).items():
                panel[name][q]=p
        summary={}
        for name,p in panel.items():
            assert np.isfinite(p).all()
            for key,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                ok=(score>=.5)==y
                bad=(base>=.5)!=y
                summary[key]=dict(accuracy=float(ok.mean()),fixed=int((ok&bad).sum()),
                                  harmed=int((~ok&~bad).sum()),previous1070_correct=int(ok[bad_all].sum()),
                                  previous466_correct=int(ok[hard].sum()))
        results[str(seed)]=summary
        np.savez_compressed(out/f"{seed}.npz",row=row,fold=fold,y=y,base=base,**panel)
        print(seed,summary,flush=True)
    write(ROOT/"reports/nov23_gibbs_manifest/result.json",dict(results=results,
        note="Both arms share all 512 sampled landmarks, inputs, Gaussian coefficient prior, outer training and Laplace integration. Only bandwidth differs. Radius is the 64th closest outer-training passenger in a different booking."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
