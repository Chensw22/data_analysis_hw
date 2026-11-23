"""Expose a frozen tabular model to complementary equal-size supports with different class priors."""
import hashlib
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer,torch
from broad_class_cost_path_n23 import REPORT,decisions
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from released_correction_audit_n23 import development


def split_support(t,y,seed,m=1280):
    rng=np.random.default_rng(seed)
    neg,pos=[rng.choice(t[y[t]==label],size=m,replace=False) for label in (False,True)]
    def arms(p,n):
        return [np.sort(np.r_[pos[:p],neg[:n]]),np.sort(np.r_[pos[p:],neg[n:]])]
    balanced=arms(m//2,m//2)
    skewed=arms(m//4,3*m//4)
    np.testing.assert_array_equal(np.sort(np.concatenate(balanced)),np.sort(np.concatenate(skewed)))
    for a,b in (balanced,skewed):
        assert len(a)==len(b)==m and not np.intersect1d(a,b).size
    return balanced+skewed


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,frame,g,_=load()
    labels,release,_=development()
    np.testing.assert_array_equal(y,labels)
    masks=scopes(y,frame)
    masks.update(repeat_true=masks["repeat"]&y,repeat_false=masks["repeat"]&~y)
    dest=ART/"nov23_tabicl_prior_context"
    dest.mkdir(exist_ok=True)
    hist,refs,plans,placement={},[],{},{}
    for seed in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{seed}.npz") as z:
            base,fold,ref=z["base"],z["fold"],z["ref"]
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
        np.testing.assert_array_equal(ref,release[131] if seed=="v131" else base>=.5)
        sides=np.empty((len(y),4))
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for j,train in enumerate(split_support(t,y,119+int(f))):
                sides[q,j]=logit(infer(frame,y,train,q,43+int(f)).clip(.001,.999))
        neutral=sides[:,:2].mean(1)
        logits=np.column_stack([sides[:,2],neutral,sides[:,3]])
        out,placement[seed]=decisions(base,ref,fold,logits,y,g)
        balanced_prob=expit(sides[:,:2]).mean(1)
        skew_prob=expit(sides[:,2:]-np.log([1/3,3])).mean(1)
        out.update(balanced_prob=balanced_prob>=.5,restored_prob=skew_prob>=.5)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,base=base,fold=fold,ref=ref,sides=sides,**out)
        reference=ref==y
        if seed!="v131":refs.append(reference)
        stats={}
        for k,v in out.items():
            h=v==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~reference).sum()),harmed=int((~h&reference).sum()))
            if seed!="v131":hist.setdefault(k,[]).append(h)
        plans[seed]=stats
        print(seed,[(k,stats[k]["fixed"],stats[k]["harmed"]) for k in ("ordinary","restored_prob","separate_costs")],flush=True)
    write(REPORT/"tabicl.json",dict(plans=plans,placement=placement,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
