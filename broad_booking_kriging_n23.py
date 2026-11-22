"""Estimate cross-booking error covariance and predict held-out residuals by simple kriging."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import nnls
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import ART,ROOT,SEEDS,REPORT,load,assign
from broad_dispute_leaf_evidence_n23 import cohorts
from relational_error_corrector import write

OUT=ART/"nov23_broad_booking_kriging"
NAMES=("booking_40","booking_400","deck","cabin_8","side_contrast_8","cabin_80","side_contrast_80")


def geometry():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId","Cabin"])
                   for s in ("train","test")],ignore_index=True)
    cabin=raw.Cabin.str.split("/",expand=True)
    records=pd.DataFrame({"g":raw.PassengerId.str.split("_").str[0].astype(int),"deck":cabin[0],
                          "side":cabin[2],"position":pd.to_numeric(cabin[1])}).dropna()
    part=records.groupby(["g","deck","side"]).position.agg(["size","median"]).reset_index()
    part=part.sort_values(["g","size","deck","side"],ascending=[True,False,True,True]).drop_duplicates("g")
    return part.set_index("g")


def components(g,geo):
    table=geo.reindex(g)
    pos=table["median"].to_numpy()
    deck=table.deck.fillna("?").to_numpy()
    side=table.side.map({"P":-1,"S":1}).fillna(0).to_numpy()
    valid=np.isfinite(pos)
    db=np.abs(g[:,None]-g)
    dc=np.abs(np.nan_to_num(pos)[:,None]-np.nan_to_num(pos))
    same=(deck[:,None]==deck)&valid[:,None]&valid
    cross=side[:,None]*side
    factors=[np.exp(-db/40),np.exp(-db/400),same]
    for scale in (8,80):
        f=same*np.exp(-dc/scale)
        factors.extend((f,f*cross))
    return np.stack(factors).astype(np.float32)


def train_field(K,t,q,value,mean_prob):
    # Fit off-diagonal covariance only. Each training record here is one
    # booking, so a passenger can never be paired with their own party.
    tri=np.triu_indices(len(t),1)
    i,j=t[tri[0]],t[tri[1]]
    x=K[:,i,j].T
    close=(x[:,0]>.6)|(x[:,3]>.6)
    rng=np.random.default_rng(951)
    remote=np.flatnonzero(~close)
    sample=np.r_[np.flatnonzero(close),rng.choice(remote,min(50000,len(remote)),replace=False)]
    x=x[sample].astype(float)
    a,b=tri[0][sample],tri[1][sample]
    empirical=value[a]*value[b]
    # Normal equations with a small ridge penalty, solved as a nonnegative
    # least squares problem; component amplitudes are covariance variances.
    normal=x.T@x+2*np.eye(len(K))
    root=np.linalg.cholesky(normal)
    coeff=nnls(root.T,np.linalg.solve(root,x.T@empirical))[0]
    total=np.einsum("i,ijk->jk",coeff,K,dtype=np.float64)
    nugget=max(float(np.mean(value**2)-total[t,t].mean()),.25*float(np.mean(value**2)),1e-4)
    system=total[np.ix_(t,t)]+nugget*np.eye(len(t))
    alpha=cho_solve(cho_factor(system),value)
    delta=total[np.ix_(q,t)]@alpha
    # Same covariance fit, but calibration residuals are reassigned among
    # bookings of similar predicted probability to erase their location.
    shuffled=value.copy()
    for block in np.array_split(np.argsort(mean_prob,kind="stable"),12):
        shuffled[block]=rng.permutation(value[block])
    null=total[np.ix_(q,t)]@cho_solve(cho_factor(system),shuffled)
    return delta,null,coeff


def run():
    row,y,_,groups,region=load()
    masks=cohorts(row,y)
    geo=geometry()
    OUT.mkdir(exist_ok=True)
    hits={}
    coefs={}
    refs=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
        with np.load(ART/f"nov23_bill_matched_register/{seed}.npz") as z:
            reg=z["register"]
            for key,expected in [("row",row),("fold",fold),("y",y)]:
                np.testing.assert_array_equal(z[key],expected)
        saved={}
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            unique,code=np.unique(groups[outer],return_inverse=True)
            K=components(unique,geo)
            for mode in ("random","order","cabin"):
                split=assign(outer,groups,region,mode,3,442)
                for k in np.unique(split):
                    a,b=np.flatnonzero(split!=k),np.flatnonzero(split==k)
                    t,q=outer[a],outer[b]
                    gu,inv=np.unique(code[a],return_inverse=True)
                    n=np.bincount(inv)
                    query,back=np.unique(code[b],return_inverse=True)
                    assert not np.intersect1d(gu,query).size
                    for teacher,p in (("baseline",base),("registration",reg)):
                        score=logit(p[outer].clip(.002,.998))[:,None]
                        cal=LogisticRegression(C=1,max_iter=250).fit(score[a],y[t])
                        pc=cal.predict_proba(score)[:,1]
                        risk=np.minimum(pc,1-pc)
                        direction=pc>=.5
                        residuals={"signed":y[t].astype(float)-pc[a],
                                   "difficulty":(direction[a]!=y[t]).astype(float)-risk[a]}
                        prefix=teacher+"_"+mode+"_"
                        saved.setdefault(prefix+"calibrated",np.full(len(y),np.nan))[q]=pc[b]
                        meanp=np.bincount(inv,weights=pc[a])/n
                        for key,resid in residuals.items():
                            values=np.bincount(inv,weights=resid)/n
                            delta,noise,c=train_field(K,gu,query,values,meanp)
                            coefs.setdefault(prefix+key,[]).append(c.tolist())
                            for label,change in ((key,delta),(key+"_shuffled",noise)):
                                if key=="signed":
                                    prob=pc[b]+change[back]
                                else:
                                    err=risk[b]+change[back]
                                    prob=np.where(direction[b],1-err,err)
                                saved.setdefault(prefix+label,np.full(len(y),np.nan))[q]=prob.clip(.001,.999)
        for label,p in saved.items():
            assert np.isfinite(p).all()
            hits.setdefault(label,[]).append((p>=.5)==y)
        refs.append((base>=.5)==y)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**saved)
        print(seed,{k:round(float(((p>=.5)==y).mean()),4) for k,p in saved.items() if k.startswith("baseline_") and not k.endswith("shuffled")},flush=True)
    ref=np.stack(refs)
    records={}
    for label,seq in hits.items():
        hit=np.stack(seq)
        prefix="_".join(label.split("_")[:2])
        learned=np.stack(hits[prefix+"_calibrated"])
        records[label]=dict(accuracy=hit.mean(1).tolist(),fixed=(hit&~ref).sum(1).tolist(),
                            harmed=(~hit&ref).sum(1).tolist(),net_vs_calibration=(hit.astype(int)-learned).sum(1).tolist(),
                            stable={k:int((hit.all(0)&m).sum()) for k,m in masks.items()})
    write(REPORT/"spatial_covariance.json",dict(results=records,components=NAMES,fitted_covariances=coefs))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
