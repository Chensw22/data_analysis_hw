"""Measure prior sensitivity with identical distinct labelled passengers in every context."""
import hashlib
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer,torch
from broad_class_cost_path_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from released_correction_audit_n23 import development


def contexts(t,y,seed,m=512):
    rng=np.random.default_rng(seed)
    orig=np.concatenate([rng.choice(t[y[t]==label],m,replace=False) for label in (False,True)])
    selected=np.concatenate([rng.permutation(np.arange(m))[:m//2],
                             m+rng.permutation(np.arange(m))[:m//2]])
    sham=np.zeros(len(orig),bool)
    sham[selected]=True
    weights=[np.full(2*m,2),1+2*~y[orig],1+2*y[orig],1+2*sham,1+2*~sham]
    perm=rng.permutation(4*m)
    out=[np.repeat(orig,w)[perm] for w in weights]
    for ids in out:
        np.testing.assert_array_equal(np.unique(ids),np.unique(orig))
        assert len(ids)==4*m
    assert [int(y[s].sum()) for s in out]==[2*m,m,3*m,2*m,2*m]
    return out


def train_gate(base,ref,fold,logits,y,g):
    anchor=logit(base.clip(.001,.999))
    sign=np.where(base>=.5,1.,-1.)
    main=np.c_[anchor,logits[:,0]]
    # Sensitivity and curvature can change the size of the correction
    # differently depending on which decision the incumbent originally made.
    def factors(lo,hi):
        spread=(hi-lo)/2
        bend=(hi+lo)/2-logits[:,0]
        return np.c_[spread,bend,spread*sign,bend*sign]
    path=factors(logits[:,1]+np.log(3),logits[:,2]-np.log(3))
    control=factors(logits[:,3],logits[:,4])
    out={}
    for f in np.unique(fold):
        ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
            t,q=ix[a],ix[b]
            assert not np.intersect1d(g[t],g[q]).size
            for name,x in (("plain",main),("prior",np.c_[main,path]),
                           ("replication_control",np.c_[main,control])):
                sd=x[t].std(0).clip(.2)
                p=LogisticRegression(C=.4,max_iter=400).fit(x[t]/sd,y[t]).predict(x[q]/sd)
                out.setdefault(name,ref.copy())[q]=p
    return out


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    labels,release,_=development()
    np.testing.assert_array_equal(y,labels)
    dest=ART/"nov23_replicated_prior"
    dest.mkdir(exist_ok=True)
    history,refs,plans,diagnostics={},[],{},{}
    for seed in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold,ref=z["base"],z["fold"],z["ref"]
        np.testing.assert_array_equal(ref,release[131] if seed=="v131" else base>=.5)
        logits=np.empty((len(y),5))
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for j,s in enumerate(contexts(t,y,183+int(f))):
                logits[q,j]=logit(infer(frame,y,s,q,43+int(f)).clip(.001,.999))
        out=train_gate(base,ref,fold,logits,y,g)
        out["ordinary"]=logits[:,0]>=0
        out["prior_average"]=logits[:,1:3].mean(1)>=0
        out["control_average"]=logits[:,3:].mean(1)>=0
        reference=ref==y
        if seed!="v131":refs.append(reference)
        result={}
        for k,v in out.items():
            h=v==y
            result[k]=dict(accuracy=float(h.mean()),fixed=int((h&~reference).sum()),harmed=int((~h&reference).sum()))
            if seed!="v131":history.setdefault(k,[]).append(h)
        slope=(logits[:,2]-logits[:,1])/(2*np.log(3))
        diagnostics[seed]=dict(prior_slope_quantiles=np.quantile(slope,[.1,.5,.9]).tolist(),
                ranks_reversed=int((slope<0).sum()))
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,base=base,fold=fold,ref=ref,logits=logits)
        plans[seed]=result
        print(seed,result["prior"],diagnostics[seed],flush=True)
    write(REPORT/"replicated.json",dict(plans=plans,sensitivity=diagnostics,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
