"""Teach a frozen classifier only on unseen support passengers that opposite cost models dispute."""
import hashlib
import numpy as np
from catboost import CatBoostClassifier,Pool
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer,torch
from broad_class_cost_path_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development


def classify(frame,y,learn,score):
    out=[]
    for r in (1/3,1,3):
        weight=np.where(y[learn],r,1.)
        model=CatBoostClassifier(iterations=200,depth=5,learning_rate=.075,random_seed=395,
                                  verbose=False,allow_writing_files=False,thread_count=2)
        model.fit(Pool(frame.iloc[learn],y[learn],cat_features=CATEGORICAL,weight=weight/weight.mean()))
        out.append(model.predict(Pool(frame.iloc[score],cat_features=CATEGORICAL),prediction_type="RawFormulaVal"))
    return np.column_stack(out)


def same_counts(group,mask,rng):
    out=[]
    for k in np.unique(group[mask]):
        n=int((mask&(group==k)).sum())
        out.extend(rng.choice(np.flatnonzero(group==k),n,replace=False))
    result=np.sort(out)
    np.testing.assert_array_equal(np.sort(group[result]),np.sort(group[mask]))
    return result


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    labels,release,_=development()
    np.testing.assert_array_equal(labels,y)
    dest=ART/"nov23_ambiguous_context"
    dest.mkdir(exist_ok=True)
    results,history,references,diagnostics={},{},[],{}
    for seed in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        np.testing.assert_array_equal(ref,release[131] if seed=="v131" else base>=.5)
        names=("class_count","confidence","disputed")
        sums={k:np.zeros(len(y)) for k in names}
        seen=np.zeros(len(y),int)
        log=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for a,b in GroupKFold(2,shuffle=True,random_state=629).split(t,groups=g[t]):
                learn,pool=t[a],t[b]
                assert not np.intersect1d(g[learn],g[pool]).size
                p=classify(frame,y,learn,np.r_[pool,q])
                flag=(p[:,0]>=0)!=(p[:,2]>=0)
                mask=flag[:len(pool)]
                active=q[flag[len(pool):]&(ref[q]==(base[q]>=.5))]
                prob=p[:len(pool),1]
                band=np.digitize(prob,np.quantile(prob,np.arange(1,16)/16))
                rng=np.random.default_rng(397)
                indices=(same_counts(y[pool],mask,rng),same_counts(2*band+y[pool],mask,rng),np.flatnonzero(mask))
                log.append(dict(train=int(mask.sum()),query=len(active),confidence_overlap=int(mask[indices[1]].sum())))
                if not len(active):
                    continue
                seen[active]+=1
                for k,ii in zip(names,indices):
                    trained=pool[ii]
                    assert not np.intersect1d(g[trained],g[active]).size
                    sums[k][active]+=infer(frame,y,trained,active,43+int(f))
        flips={}
        active=seen>0
        for k in names:
            p=base.copy()
            p[active]=sums[k][active]/seen[active]
            flips[k]=np.where(active,p>=.5,ref)
            out=ref.copy()
            design=np.c_[logit(base.clip(.001,.999)),logit(p.clip(.001,.999)),seen]
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&active)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    out[q]=LogisticRegression(C=.5,max_iter=400).fit(design[t],y[t]).predict(design[q])
            flips["combined_"+k]=out
        reference=ref==y
        stats={}
        if seed!="v131":references.append(reference)
        for k,v in flips.items():
            h=v==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~reference).sum()),harmed=int((~h&reference).sum()))
            if seed!="v131":history.setdefault(k,[]).append(h)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,base=base,ref=ref,fold=fold,seen=seen,**sums)
        diagnostics[seed]=dict(active=int(active.sum()),repeat_in_active=int((active&masks["repeat"]).sum()),arms=log)
        results[seed]=stats
        print(seed,stats["combined_disputed"],stats["combined_confidence"],flush=True)
    write(REPORT/"ambiguous.json",dict(plans=results,scope=diagnostics,grouped={
        k:summarize(np.stack(v),np.stack(references),masks) for k,v in history.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
