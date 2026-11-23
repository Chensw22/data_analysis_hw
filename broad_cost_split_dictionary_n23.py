"""Keep class-specialized tree splits and relearn their leaf values for ordinary accuracy."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit,logit
from catboost import CatBoostClassifier,Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_class_cost_path_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development


def leaves(frames,counts):
    matrix=np.column_stack(frames)
    widths=np.concatenate(counts)
    offsets=np.r_[0,widths.cumsum()[:-1]]
    assert ((matrix>=0)&(matrix<widths[None,:])).all()
    idx=(matrix+offsets).astype(np.int32).ravel()
    ptr=np.arange(matrix.shape[0]+1,dtype=np.int32)*matrix.shape[1]
    return csr_matrix((np.ones(len(idx)),idx,ptr),shape=(len(matrix),int(widths.sum())))


def recalibrate(base,ref,fold,prob,y,g):
    eligible=ref==(base>=.5)
    x=np.c_[logit(base.clip(.001,.999)),logit(prob.clip(.001,.999))]
    out=ref.copy()
    for f in np.unique(fold):
        ids=np.flatnonzero((fold==f)&eligible)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            out[q]=LogisticRegression(C=.5,max_iter=400).fit(x[t],y[t]).predict(x[q])
    return out


def predict(frame,y,t,q):
    inputs=Pool(frame.iloc[np.r_[t,q]],cat_features=CATEGORICAL)
    bank={}
    rng=np.random.default_rng(953)
    perm=rng.permutation(y[t])
    for seed in (219,513,785):
        # The middle forest is shared by all three comparators.
        r={219:1/3,513:1.,785:3.}[seed]
        for mode in (("neutral",) if r==1 else ("neutral","weighted","shuffled")):
            w=np.ones(len(t)) if mode=="neutral" else np.where(y[t] if mode=="weighted" else perm,r,1.)
            fit=CatBoostClassifier(iterations=112,depth=5,learning_rate=.09,verbose=False,
                  random_seed=seed,one_hot_max_size=32,thread_count=2,allow_writing_files=False)
            fit.fit(Pool(frame.iloc[t],y[t],cat_features=CATEGORICAL,weight=w/w.mean()))
            leaf=fit.calc_leaf_indexes(inputs,thread_count=2)
            total=fit.predict(inputs,prediction_type="RawFormulaVal")
            bank[mode,seed]=(leaf,fit.get_tree_leaf_counts(),total)
    out={}
    for mode in ("neutral","weighted","shuffled"):
        values=[bank["neutral" if seed==513 else mode,seed] for seed in (219,513,785)]
        # The two specialist log-prior offsets cancel in the equal logit average.
        out[mode+"_scalar"]=expit(np.mean([v[2][len(t):] for v in values],axis=0))
        x=leaves([v[0] for v in values],[v[1] for v in values])
        fit=LogisticRegression(C=.06,max_iter=450,solver="lbfgs").fit(x[:len(t)],y[t])
        assert fit.n_iter_.max()<450
        out[mode+"_leaves"]=fit.predict_proba(x[len(t):])[:,1]
    return out


def main():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    labels,release,_=development()
    np.testing.assert_array_equal(y,labels)
    dest=ART/"nov23_cost_split_dictionary"
    dest.mkdir(exist_ok=True)
    results,refs,history={},[],{}
    for seed in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{seed}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        np.testing.assert_array_equal(ref,release[131] if seed=="v131" else base>=.5)
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for k,v in predict(frame,y,t,q).items():
                scores.setdefault(k,np.full(len(y),np.nan))[q]=v
        ref_hit=ref==y
        if seed!="v131":refs.append(ref_hit)
        methods={}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            methods[k]=p>=.5
            methods["combined_"+k]=recalibrate(base,ref,fold,p,y,g)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        result={}
        for k,v in methods.items():
            h=v==y
            result[k]=dict(accuracy=float(h.mean()),fixed=int((h&~ref_hit).sum()),harmed=int((~h&ref_hit).sum()))
            if seed!="v131":history.setdefault(k,[]).append(h)
        results[seed]=result
        print(seed,{k:v for k,v in result.items() if k in ("weighted_leaves","combined_weighted_leaves")},flush=True)
    write(REPORT/"partitions.json",dict(plans=results,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
