"""Compare complementary whole-booking and within-booking annotation at matched budgets."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_error_learning_curve_n23 import paired_changes
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_annotation_unit"


def arms(t, g, size, seed):
    rng = np.random.default_rng(seed)
    units = pd.Series(t).groupby(g[t]).agg(list)
    strata_map = {}
    for group, members in units.items():
        strata_map.setdefault((size[members[0]], len(members)), []).append(group)
    picks = {"together": [[], []], "dispersed": [[], []]}
    for (_, n), keys in strata_map.items():
        keys = rng.permutation(keys)
        for j in range(0,len(keys)-1,2):
            a,b = [rng.permutation(units[k]) for k in keys[j:j+2]]
            k = n//2 + (rng.integers(2) if n%2 and n>1 else 0)
            picks["together"][0].extend(a)
            picks["together"][1].extend(b)
            if n==1:
                picks["dispersed"][0].extend(a)
                picks["dispersed"][1].extend(b)
            else:
                picks["dispersed"][0].extend(np.r_[a[:k],b[:n-k]])
                picks["dispersed"][1].extend(np.r_[a[k:],b[n-k:]])
        if len(keys)%2:
            arm = rng.integers(2)
            for v in picks.values():
                v[arm].extend(units[keys[-1]])
    out = {kind:[np.sort(a) for a in both] for kind,both in picks.items()}
    for a,b in out.values():
        np.testing.assert_array_equal(np.sort(np.r_[a,b]),t)
        assert not np.intersect1d(a,b).size
    for a,b in zip(out["together"],out["dispersed"]):
        np.testing.assert_array_equal(np.sort(size[a]),np.sort(size[b]))
        np.testing.assert_array_equal(a[size[a]==1],b[size[b]==1])
    return out


def run():
    row,y,frame,g,_ = load()
    size = frame.BookingSize.to_numpy()
    labels,versions,old = development()
    np.testing.assert_array_equal(y,labels)
    miss = np.logical_and.reduce([v!=y for v in versions.values()])
    state = strata(frame)
    core = miss & (((state==2)&y) | ((state==0)&~y))
    masks = dict(all=np.ones(len(y),bool),miss=miss,core=core,other=~core,repeat=old["repeat"],
                 solo=size==1,multi=size>1,core_solo=core&(size==1),core_multi=core&(size>1))
    assert (miss.sum(),core.sum())==(941,696)
    preds,stats,bases = {},[],[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as c:
            np.testing.assert_array_equal(c["row"],row)
            np.testing.assert_array_equal(c["y"],y)
            fold,base = c["fold"],c["base"]
        bases.append(base>=.5)
        scores = {kind:np.full((2,len(y)),np.nan) for kind in ("together","dispersed")}
        for f in np.unique(fold):
            t,q = np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            subsets=arms(t,g,size,int(seed) + int(f))
            for kind,both in subsets.items():
                for i,a in enumerate(both):
                    seen,count=np.unique(g[a],return_counts=True)
                    full_size=pd.Series(size[a],index=g[a]).groupby(level=0).first().loc[seen].to_numpy()
                    stats.append(dict(seed=int(seed),fold=int(f),arm=i,kind=kind,rows=len(a),groups=len(seen),
                                      complete=int((count==full_size).sum()),label_rate=float(y[a].mean())))
                    model=CatBoostClassifier(iterations=460,depth=6,learning_rate=.06,thread_count=2,
                        random_seed=1211,verbose=False,allow_writing_files=False)
                    model.fit(frame.iloc[a],y[a],cat_features=CATEGORICAL)
                    scores[kind][i,q]=model.predict_proba(frame.iloc[q])[:,1]
        for kind,p in scores.items():
            assert np.isfinite(p).all()
            preds.setdefault(kind,[]).append(p)
        out=ART/"nov23_annotation_unit"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**scores)
        print(seed,[(k,((p.mean(0)>=.5)==y).mean()) for k,p in scores.items()],flush=True)
    results = {}
    comparisons = {}
    for key,combine in (("one_random_arm",False),("ensemble",True)):
        hit = {}
        for kind,p in preds.items():
            p=np.stack(p)
            hit[kind]=((p.mean(1)>=.5)==y)[:,None,:] if combine else (p>=.5)==y
            ok=hit[kind]
            base_ok=(np.stack(bases)==y)[:,None,:]
            results[f"{key}_{kind}"]={s:dict(n=int(m.sum()),accuracy=ok[:,:,m].mean((1,2)).tolist(),
                    fixed=(ok & ~base_ok)[:,:,m].sum(2).mean(1).tolist(),
                    harmed=(~ok & base_ok)[:,:,m].sum(2).mean(1).tolist(),
                    all_predictions_correct=int((ok.all((0,1))&m).sum())) for s,m in masks.items()}
        a,b=hit["together"],hit["dispersed"]
        comparisons[key]=paired_changes(a.reshape(-1,len(y)),b.reshape(-1,len(y)),g,masks)
        for s,m in masks.items():
            comparisons[key][s].update(fixed=(~a&b)[:,:,m].sum(2).mean(1).tolist(),
                                        harmed=(a&~b)[:,:,m].sum(2).mean(1).tolist())
    write(REPORT/"results.json",dict(training=stats,results=results,dispersed_minus_together=comparisons))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
