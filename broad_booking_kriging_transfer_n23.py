"""Use honest residuals from every available training booking to correct unseen passengers."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import nnls
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_booking_kriging_n23 import geometry
from broad_geographic_transfer_n23 import ART,ROOT,SEEDS,REPORT,load,summary
from broad_dispute_leaf_evidence_n23 import cohorts
from group_crossfit_residual_discovery_n23 import teachers
from relational_error_corrector import write


def factor(g,d,s,x,i,j):
    delta=np.abs(g[i]-g[j])
    same=(d[i]==d[j])&(d[i]!=-1)&(d[j]!=-1)
    distance=np.abs(x[i]-x[j])
    cross=s[i]*s[j]
    near=same*np.exp(-distance/8)
    far=same*np.exp(-distance/80)
    return np.stack([np.exp(-delta/40),np.exp(-delta/400),same,near,near*cross,far,far*cross],axis=-1)


def smooth(group_ids,geo,t,q,value,prob):
    frame=geo.reindex(group_ids)
    deck=pd.Categorical(frame.deck).codes
    side=frame.side.map({"P":-1,"S":1}).fillna(0).to_numpy()
    x=frame["median"].fillna(0).to_numpy()
    def features(i,j):
        return factor(group_ids,deck,side,x,i,j)
    # All near pairs and a uniform reservoir of up to 50,000 remote pairs.
    # Generate triangular pairs by blocks to avoid a dense seven-kernel bank.
    rng=np.random.default_rng(951)
    near_i,near_j=[],[]
    far_i=np.empty(0,int);far_j=np.empty(0,int);keys=np.empty(0)
    for start in range(0,len(t),128):
        end=min(start+128,len(t))
        i,j=np.where(np.arange(start,end)[:,None]<np.arange(len(t))[None,:])
        i+=start
        dist=np.abs(group_ids[t[i]]-group_ids[t[j]])
        nearby=(dist< -40*np.log(.6)) | ((deck[t[i]]==deck[t[j]])&(deck[t[i]]>=0)&
                                              (np.abs(x[t[i]]-x[t[j]])< -8*np.log(.6)))
        near_i.append(i[nearby]);near_j.append(j[nearby])
        far_i=np.r_[far_i,i[~nearby]];far_j=np.r_[far_j,j[~nearby]]
        keys=np.r_[keys,rng.random((~nearby).sum())]
        if len(keys)>50000:
            take=np.argpartition(keys,49999)[:50000]
            far_i,far_j,keys=far_i[take],far_j[take],keys[take]
    i=np.concatenate(near_i+[far_i]);j=np.concatenate(near_j+[far_j])
    feats=features(t[i],t[j])
    root=np.linalg.cholesky(feats.T@feats+2*np.eye(feats.shape[1]))
    c=nnls(root.T,np.linalg.solve(root,feats.T@(value[i]*value[j])))[0]
    def covariance(left,right):
        result=np.empty((len(left),len(right)))
        for a in range(0,len(left),96):
            part=left[a:a+96]
            result[a:a+len(part)]=features(part[:,None],right[None,:])@c
        return result
    system=covariance(t,t)
    noise=max(float(np.mean(value**2)-np.diag(system).mean()),.25*float(np.mean(value**2)),.0001)
    system.flat[::len(t)+1]+=noise
    chol=cho_factor(system,overwrite_a=True)
    fake=value.copy()
    for band in np.array_split(np.argsort(prob,kind="stable"),12):
        fake[band]=rng.permutation(value[band])
    delta=covariance(q,t)@cho_solve(chol,np.column_stack([value,fake]))
    return delta[:,0],delta[:,1],c


def run():
    rows,y,frame,groups,_=load()
    masks=cohorts(rows,y)
    geo=geometry()
    plans=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for key,v in (("row",rows),("y",y)):
                np.testing.assert_array_equal(z[key],v)
            plans.append((str(seed),z["fold"],z["base"],None))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        hist=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
        for key,v in (("row",rows),("true",y),("fold",z["fold"]),("v131",z["v131"])):
            np.testing.assert_array_equal(hist[key],v)
        plans.append(("v131",z["fold"],z["v108"],z["v131"]))
    report={}
    hits={}
    out=ART/"nov23_broad_geographic_transfer"
    for name,fold,base,inc in plans:
        bank={}
        coefs=[]
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            seed=20261112+int(k) if inc is not None else int(name)+int(k)
            p,qt=teachers(frame,y,groups,t,q,seed)
            odds=logit(p.clip(.002,.998))[:,None]
            cal=LogisticRegression(C=1.).fit(odds,y[t])
            p=cal.predict_proba(odds)[:,1]
            teacher=cal.predict_proba(logit(qt.mean(1).clip(.002,.998))[:,None])[:,1]
            weights=np.unique(groups[t],return_inverse=True)
            known,idx=weights
            count=np.bincount(idx)
            residual=np.bincount(idx,weights=y[t].astype(float)-p)/count
            avg=np.bincount(idx,weights=p)/count
            query,back=np.unique(groups[q],return_inverse=True)
            all_groups=np.r_[known,query]
            qr=np.arange(len(known),len(all_groups))
            subset=np.sort(np.random.default_rng(222).permutation(len(known))[:len(known)//3])
            for label,tr in (("one_third",subset),("full",np.arange(len(known)))):
                delta,fake,c=smooth(all_groups,geo,tr,qr,residual[tr],avg[tr])
                if label=="full":
                    coefs.append(c.tolist())
                for feature,prob in ((label,base[q]+delta[back]),(label+"_placebo",base[q]+fake[back]),
                                     (label+"_native",teacher+delta[back])):
                    bank.setdefault(feature,np.full(len(y),np.nan))[q]=prob.clip(.001,.999)
            bank.setdefault("teacher",np.full(len(y),np.nan))[q]=teacher
        if inc is not None:
            override=inc!=(base>=.5)
            for label in ("one_third","full","one_third_placebo","full_placebo"):
                bank[label]=np.where(override,inc.astype(float),bank[label])
            reference=inc.astype(float)
        else:
            reference=base
            for key,p in bank.items():
                hits.setdefault(key,[]).append((p>=.5)==y)
        report[name]=dict(results=summary(y,bank,reference,fold,masks),covariances=coefs)
        np.savez_compressed(out/f"all_training_{name}.npz",row=rows,y=y,fold=fold,base=reference,**bank)
        print(name,report[name]["results"]["full"],flush=True)
    stable={n:{k:int((np.logical_and.reduce(v)&m).sum()) for k,m in masks.items()} for n,v in hits.items()}
    write(REPORT/"all_training_residuals.json",dict(plans=report,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
