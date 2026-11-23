"""Measure how many disputed decisions must be correct to reach 85% on the full roster."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_class_cost_path_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def get(folder,seed,row,y,base,ref,fold,key):
    with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
        for k,v in (("row",row),("y",y),("base",base),("ref",ref),("fold",fold)):
            np.testing.assert_array_equal(z[k],v)
        return z[key]


def requirements(mask,ref,y,repeat):
    outside=int(((ref==y)&~mask).sum())
    needed=int(np.floor(.85*len(y)))+1-outside
    return dict(count=int(mask.sum()),initial_correct=int(((ref==y)&mask).sum()),
                possible_fixed=int(((ref!=y)&mask).sum()),required_correct=needed,
                required_fraction=float(needed/mask.sum()),repeat_in_scope=int((mask&repeat).sum()))


def main():
    row,y,_,g,_=load()
    labels,releases,ms=development()
    np.testing.assert_array_equal(labels,y)
    masks={k:ms[k] for k in ("ever","repeat")}
    dest=ART/"nov23_cost_disagreement"
    dest.mkdir(exist_ok=True)
    hist,refs,settings,eligibility={},[],{},[]
    for seed in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,ref,fold=z["base"],z["ref"],z["fold"]
        np.testing.assert_array_equal(ref,releases[131] if seed=="v131" else base>=.5)
        def read(folder,key):
            return get(folder,seed,row,y,base,ref,fold,key)
        tree=read("class_cost_path","logits")
        icl=read("tabicl_prior_context","sides")
        tab=np.c_[icl[:,2],icl[:,:2].mean(1),icl[:,3]]
        one=(tree[:,0]>=0)!=(tree[:,2]>=0)
        two=(tab[:,0]>=0)!=(tab[:,2]>=0)
        free=ref==(base>=.5)
        regions={"tree":free&one,"tabicl":free&two,"both":free&one&two,"either":free&(one|two)}
        region=regions["either"]
        if seed!="v131":eligibility.append(region)
        summary={k:requirements(m,ref,y,masks["repeat"]) for k,m in regions.items()}
        seen=read("ambiguous_context","seen")
        amb=read("ambiguous_context","disputed")/np.maximum(seen,1)
        amb=np.where(seen>0,amb,base)
        priors=np.log([1/3,3])
        ordinary=np.c_[logit(base.clip(.001,.999)),tree[:,1],tab[:,1]]
        dx=np.c_[tree[:,[0,2]]-priors-tree[:,1,None],tab[:,[0,2]]-priors-tab[:,1,None]]
        sign=np.where(ref,1.,-1.)
        rich=np.c_[dx,sign[:,None]*dx]
        pred={}
        for f in np.unique(fold):
            ix=np.flatnonzero((fold==f)&region)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                t,q=ix[a],ix[b]
                assert not np.intersect1d(g[t],g[q]).size
                for name,x in (("ordinary",ordinary),("costs",np.c_[ordinary,rich]),
                    ("costs_and_disputed",np.c_[ordinary,rich,logit(amb.clip(.001,.999)),seen])):
                    sd=x[t].std(0).clip(.15)
                    fit=LogisticRegression(C=.25,max_iter=400).fit(x[t]/sd,y[t])
                    pred.setdefault(name,ref.copy())[q]=fit.predict(x[q]/sd)
        ref_hit=ref==y
        if seed!="v131":refs.append(ref_hit)
        decisions={}
        for k,v in pred.items():
            np.testing.assert_array_equal(v[~region],ref[~region])
            h=v==y
            decisions[k]=dict(region_correct=int((h&region).sum()),region_fraction=float(h[region].mean()),
                overall=float(h.mean()),fixed=int((h&~ref_hit).sum()),harmed=int((~h&ref_hit).sum()))
            if seed!="v131":hist.setdefault(k,[]).append(h)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**pred)
        settings[seed]=dict(regions=summary,models=decisions)
        print(seed,summary["either"],decisions["costs_and_disputed"],flush=True)
    stable_scope=np.logical_and.reduce(eligibility)
    metrics={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}
    write(REPORT/"joint.json",dict(plans=settings,grouped=metrics,
         always_eligible=int(stable_scope.sum()),ever_eligible=int(np.logical_or.reduce(eligibility).sum()),
         repeat_always_eligible=int((stable_scope&masks["repeat"]).sum())))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
