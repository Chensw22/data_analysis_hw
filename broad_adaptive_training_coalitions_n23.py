"""Select training-block coalitions for each query using their fold-local predictive evidence."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_billing_response_surface_n23 import REPORT, surface_inputs
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_support_trajectories_n23 import LEVEL,SUBSETS
from released_correction_audit_n23 import development

MEMBER=np.array([[j in subset for j in range(4)] for subset in SUBSETS]).astype(float)


def marginal_by_block(p):
    index={tuple(s):i for i,s in enumerate(SUBSETS)}
    logits=logit(p.clip(.001,.999))
    cols=[]
    for j in range(4):
        changes=np.column_stack([logits[:,index[tuple(sorted((*s,j)))]]-logits[:,i]
                                 for i,s in enumerate(SUBSETS) if j not in s])
        cols.append(np.c_[logits[:,index[(j,)]]-logits[:,-1],changes.mean(1),changes.std(1),
                          np.mean(changes>0,axis=1)-.5])
    return np.stack(cols,axis=1)


def local_evidence(x,y,g,t,q,seed):
    blocks=[t[b] for _,b in GroupKFold(4,shuffle=True,random_state=seed).split(t,groups=g[t])]
    split=np.concatenate(blocks)
    np.testing.assert_array_equal(np.sort(split),t)
    cols=[]
    d=cdist(x[q],x[split],"sqeuclidean")
    radius=np.maximum(np.partition(d,63,axis=1)[:,63],1e-8)
    wt=np.exp(-d/radius[:,None])
    for ids,w in zip(blocks,np.split(wt,np.cumsum(list(map(len,blocks)))[:-1],axis=1)):
        assert not np.intersect1d(g[ids],g[q]).size
        mass=w.sum(1)
        chance=(2+w@y[ids])/(4+mass)
        cols.append(np.c_[np.log1p(mass),logit(chance)])
    out=np.stack(cols,axis=1)
    return out-out.mean(1,keepdims=True)


def fit_gate(features,base,p,y,t,q,include_base):
    # Conditional on a nonempty subset, independent block-inclusion logits
    # give subset weights proportional to exp(sum of its included logits).
    sd=features[t].std((0,1)).clip(.1)
    x=np.concatenate([np.ones((*features.shape[:2],1)),(features/sd).clip(-5,5)],axis=2)
    z=logit(base.clip(.001,.999))
    d=x.shape[2]
    def calculate(params,ids):
        eta=x[ids]@params[:d]
        w=softmax(eta@MEMBER.T,axis=1)
        mean=(w*p[ids]).sum(1).clip(.00001,.99999)
        raw=logit(mean)
        design=np.c_[np.ones(len(ids)),raw,z[ids]] if include_base else np.c_[np.ones(len(ids)),raw]
        score=design@params[d:]
        return score,design,mean,w
    start=np.r_[np.zeros(d),[0,.5,1.] if include_base else [0,1.]]
    penalty=np.r_[np.full(d,10.),np.full(len(start)-d,2.)]
    def objective(params):
        score,design,mean,w=calculate(params,t)
        err=expit(score)-y[t]
        deriv=(w*(p[t]-mean[:,None]))@MEMBER
        scale=err*params[d+1]/(mean*(1-mean))
        gradient=np.einsum("i,ij,ijk->k",scale,deriv,x[t])
        return (np.logaddexp(0,score).sum()-score@y[t]+.5*penalty@((params-start)**2),
                np.r_[gradient,design.T@err]+penalty*(params-start))
    sol=minimize(objective,start,jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-11})
    assert sol.success,sol.message
    score,_,_,w=calculate(sol.x,q)
    return expit(score),w@LEVEL


def run():
    row,y,frame,g,_=load()
    labels,released,old_masks=development()
    np.testing.assert_array_equal(labels,y)
    miss=np.logical_and.reduce([p!=y for p in released.values()])
    state=strata(frame)
    core=miss&(((state==2)&y)|((state==0)&~y))
    masks=dict(miss=miss,core=core,repeat=old_masks["repeat"],core_solo=core&(frame.BookingSize.to_numpy()==1))
    assert [v.sum() for v in masks.values()]==[941,696,1070,435]
    r,b=surface_inputs(frame)
    x=np.c_[r/np.sqrt(r.shape[1]),b/np.sqrt(b.shape[1])]
    history,refs,actions,means={},[],{},{}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as c:
            np.testing.assert_array_equal(c["row"],row);np.testing.assert_array_equal(c["y"],y)
            base,fold=c["base"],c["fold"]
        with np.load(ART/f"nov23_broad_support_trajectories/{seed}.npz") as c:
            for k,v in (("row",row),("y",y),("fold",fold),("subsets",LEVEL)):
                np.testing.assert_array_equal(c[k],v)
            p=c["bank"].T
        refs.append((base>=.5)==y)
        shape=marginal_by_block(p)
        signed=2*(base>=.5).astype(float)-1
        oriented=shape*signed[:,None,None]
        scores,expect={},{}
        for f in np.unique(fold):
            t=np.flatnonzero(fold!=f)
            q=np.flatnonzero(fold==f)
            # Original splits use zero-padded group strings, whose sorted order
            # is the same as these verified numeric group IDs.
            local=local_evidence(x,y,g,t,q,seed+int(f))
            between=local[:,:,1]*oriented[q,:,1]
            both=np.concatenate([oriented[q],local,between[:,:,None]],axis=2)
            rng=np.random.default_rng(428)
            fake=local[np.arange(len(q))[:,None],np.argsort(rng.random((len(q),4)),axis=1)]
            fake=np.concatenate([oriented[q],fake,(fake[:,:,1]*oriented[q,:,1])[:,:,None]],axis=2)
            for k,features in (("uniform",both[:,:,:0]),("marginal",oriented[q]),("local",both),("shuffled",fake)):
                for a,v in GroupKFold(3,shuffle=True,random_state=925).split(q,groups=g[q]):
                    assert not np.intersect1d(g[q[a]],g[q[v]]).size
                    for anchored in (False,True):
                        key=("combined_" if anchored else "native_")+k
                        pred,ex=fit_gate(features,base[q],p[q],y[q],a,v,anchored)
                        scores.setdefault(key,np.full(len(y),np.nan))[q[v]]=pred
                        expect.setdefault(key,np.full(len(y),np.nan))[q[v]]=ex
        actions[seed]={}
        means[seed]={}
        for k,z in scores.items():
            assert np.isfinite(z).all()
            h=(z>=.5)==y
            history.setdefault(k,[]).append(h)
            means[seed][k]=float(expect[k].mean())
            full=p[:,-1]>=.5
            changed=full!=(z>=.5)
            actions[seed][k]=dict(fixed_vs_full=int((changed&h).sum()),harmed_vs_full=int((changed&~h).sum()))
        folder=ART/"nov23_adaptive_training_coalitions"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**scores)
    results={k:summarize(np.stack(v),np.stack(refs),masks) for k,v in history.items()}
    write(REPORT/"coalitions.json",dict(results=results,compare_full=actions,expected_blocks=means))
    print(results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
