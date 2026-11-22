"""Learn whether correlated patterns of independent model opinions add held-booking information."""
import numpy as np
from scipy.special import expit,logit,logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_diverse_rescue_n23 import ACTIONS,MORE,predictions
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,REPORT,OUT,SEEDS,historical,evaluate
from broad_weak_complement_n23 import combine
from relational_error_corrector import write


def relative_symbols(base,p,t):
    x=logit(base.clip(.002,.998))
    design=np.column_stack([np.ones(len(x)),x,*[np.maximum(x-k,0) for k in (-1.,0.,1.)]])
    opinions=logit(p.clip(.002,.998))
    coef=np.linalg.solve(design[t].T@design[t]+np.eye(design.shape[1]),design[t].T@opinions[t])
    delta=opinions-design@coef
    cut=np.quantile(delta[t],[.25,.5,.75],axis=0)
    return np.sum(delta[:,None,:]>cut[None,:,:],axis=1)


def fit_density(symbols,t,q,nstates):
    d=symbols.shape[1]
    x=np.eye(4)[symbols[t]].reshape(len(t),-1)
    query=np.eye(4)[symbols[q]].reshape(len(q),-1)
    # Common quarter-cell prior and exchangeable latent components.
    rg=np.random.default_rng(741)
    w=rg.dirichlet(np.ones(nstates),size=len(t))
    for _ in range(40):
        counts=w.T@x
        prob=(counts.reshape(nstates,d,4)+2)/(w.sum(0)[:,None,None]+8)
        prior=(w.sum(0)+2)/(len(t)+2*nstates)
        emission=np.log(prob).reshape(nstates,-1)
        joint=x@emission.T+np.log(prior)
        next_w=np.exp(joint-logsumexp(joint,axis=1,keepdims=True))
        if np.max(abs(next_w-w))<1e-5:
            break
        w=next_w
    return logsumexp(query@emission.T+np.log(prior),axis=1)


def run_one(base,models,y,fold,g):
    out={k:np.full(len(y),np.nan) for k in ("score","independent","latent","shuffled")}
    for k in np.unique(fold):
        held=np.flatnonzero(fold==k)
        blocks=[held[b] for _,b in GroupKFold(3,shuffle=True,random_state=627).split(held,groups=g[held])]
        for i,q in enumerate(blocks):
            a,b=blocks[(i+1)%3],blocks[(i+2)%3]
            pred={k:[] for k in out}
            for t,c in ((a,b),(b,a)):
                assert not np.intersect1d(g[q],g[np.r_[t,c]]).size and not np.intersect1d(g[t],g[c]).size
                symbols=relative_symbols(base,models,t)
                apply=np.r_[c,q]
                place=y.copy()
                rng=np.random.default_rng(591)
                for band in np.array_split(t[np.argsort(base[t])],12):
                    place[band]=rng.permutation(y[band])
                for key,count,label in (("score",0,y),("independent",1,y),("latent",3,y),("shuffled",3,place)):
                    alternatives=np.empty((len(apply),0))
                    if count:
                        pos=fit_density(symbols,t[label[t]],apply,count)
                        neg=fit_density(symbols,t[~label[t]],apply,count)
                        # Condition source scores on the prior. Calibrate the
                        # resulting log likelihood ratio on a separate group.
                        ratio=pos-neg
                        centered=(ratio-ratio[:len(c)].mean())/max(.5,ratio[:len(c)].std())
                        alternatives=expit(logit(base[apply].clip(.002,.998))+centered)[:,None]
                    pred[key].append(combine(base[apply],alternatives,y[apply],np.arange(len(c)),
                                              np.arange(len(c),len(apply))))
            for key,parts in pred.items():
                out[key][q]=np.mean(parts,axis=0)
    return out


def run():
    row,y,_,g,_=load()
    orig,target,masks=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(target,y)
    results,hist={},{}
    for name in [*map(str,SEEDS),"v131"]:
        inc=name=="v131"
        with np.load(OUT/f"{name}.npz") as z:
            for key,data in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[key],data)
            base,ref,fold=z["base"],z["ref"],z["fold"]
            cols=[z["teacher"],z["native"]]
        with np.load(ART/f"nov23_broad_service_transport/{name}.npz") as z:
            np.testing.assert_array_equal(fold,z["fold"])
            cols.append(z["learned"])
        with np.load(ART/f"nov23_broad_weak_disagreement/objective_{name}.npz") as z:
            np.testing.assert_array_equal(fold,z["fold"])
            cols.append(z["direct_smooth_count"])
        members=np.empty((len(y),5))
        for k in np.unique(fold):
            idx=np.flatnonzero(fold==k)
            with np.load(ART/f"nov23_broad_weak_disagreement/refits_{name}_{k}.npz") as z:
                np.testing.assert_array_equal(z["q"],idx)
                np.testing.assert_array_equal(z["row"],row)
                members[idx]=z["members"].mean(0)
        bank=np.column_stack([*cols,members])
        plans=[("portable",bank,base,ref),("registration_control",bank[:,:-1],members[:,-1],members[:,-1]>=.5)]
        if not inc:
            anchor,p,fold2=predictions(row,y,int(name),{**ACTIONS,**MORE})
            np.testing.assert_array_equal(base,anchor)
            np.testing.assert_array_equal(fold,fold2)
            plans.append(("all25",p,base,ref))
        results[name]={}
        for scope,models,anchor,original in plans:
            scores={"update_"+s:v for s,v in run_one(anchor,models,y,fold,g).items()}
            results[name][scope],hits=evaluate(y,anchor,original,scores,fold,inc and scope=="portable",masks)
            if not inc:
                for s,v in hits.items():
                    hist.setdefault(scope+"_"+s,[]).append(v)
            np.savez_compressed(OUT/f"latent_{name}_{scope}.npz",row=row,y=y,fold=fold,anchor=anchor,original=original,**scores)
            print(name,scope,[(s,v["fixed"]-v["harmed"]) for s,v in results[name][scope].items()],flush=True)
        write(REPORT/"latent.json",dict(plans=results))
    stable={}
    for k,rows in hist.items():
        all_hit=np.logical_and.reduce([h for h,_ in rows])
        stable[k]={n:int((m&all_hit).sum()) for n,m in masks.items()}
        stable[k]["gain_ci_pp"]=interval(np.mean([d for _,d in rows],0),g)
    write(REPORT/"latent.json",dict(plans=results,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
