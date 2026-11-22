"""Learn sparse class-specific rule votes from supervised whole-booking tree paths."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import hard_rows, ART, ROOT, SEEDS
from exaone_validation_data import compact_features, CATEGORICAL
from grouped_five_way_coverage_n23 import EXPERIMENTS, verify
from relational_error_corrector import write


def remaining(rows, y):
    mask=hard_rows(rows,y)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            fold,base=ref["fold"],ref["base"]
            verify(ref,rows,y,fold)
        for folder,key in EXPERIMENTS.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,rows,y,fold)
                p=z[key]
                mask &= ((p>=.5)!=y) & (((.75*base+.25*p)>=.5)!=y)
        with np.load(ART/f"nov23_group_disputes/{seed}.npz") as z:
            verify(z,rows,y,fold)
            for key in ["two_level_table","shallow_tree"]:
                mask &= (z[key]>=.5)!=y
    assert mask.sum()==356
    return mask


def rule_basis(forest, x):
    simple=set()
    interactions=set()
    for estimator in forest.estimators_.ravel():
        tree=estimator.tree_
        def visit(node,path):
            col=tree.feature[node]
            if col<0:
                return
            for branch,child in [(True,tree.children_left[node]),(False,tree.children_right[node])]:
                test=(int(col),float(tree.threshold[node]),branch)
                simple.add(test)
                extended=path+(test,)
                if len(set(clause[0] for clause in extended))>1:
                    interactions.add(tuple(sorted(extended)))
                visit(child,extended)
        visit(0,())
    conditions={}
    for clause in sorted(simple):
        col,threshold,left=clause
        conditions[clause]=x[:,col]<=threshold if left else x[:,col]>threshold
    main=csr_matrix(np.column_stack(list(conditions.values())).astype(float))
    crosses=csr_matrix(np.column_stack([np.logical_and.reduce([conditions[c] for c in rule])
                                         for rule in sorted(interactions)]).astype(float))
    return main,hstack([main,crosses],format='csr')


def run():
    records=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=records.row.to_numpy(),records.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in ["Transported","Name"])
                   for part in ["train","test"]],ignore_index=True)
    bookings=raw.PassengerId.str.split("_").str[0]
    groups=bookings.iloc[dev].to_numpy()
    x=compact_features(raw).iloc[dev].copy()
    x["BookingSize"]=bookings.map(bookings.value_counts()).iloc[dev].to_numpy()
    x=pd.get_dummies(x,columns=CATEGORICAL).fillna(-1).to_numpy(np.float32)
    mask=remaining(dev,y)
    out=ART/"nov23_sparse_rules"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,dev,y,fold)
        probs={name:np.full(len(y),np.nan) for name in ["tree","thresholds","rules"]}
        diagnostics=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not set(groups[t])&set(groups[q])
            forest=GradientBoostingClassifier(n_estimators=120,learning_rate=.045,max_depth=3,min_samples_leaf=20,
                                               max_features=.8,subsample=.78,random_state=312)
            forest.fit(x[t],y[t])
            probs["tree"][q]=forest.predict_proba(x[q])[:,1]
            main,all_rules=rule_basis(forest,x)
            selected={}
            for name,design in [("thresholds",main),("rules",all_rules)]:
                # Positive coefficients count votes for transported, negative coefficients
                # for nontransported. The L1 penalty removes redundant paths.
                fit=LogisticRegression(solver="liblinear",l1_ratio=1,C=.25,max_iter=2500,random_state=312)
                fit.fit(design[t],y[t])
                probs[name][q]=fit.predict_proba(design[q])[:,1]
                if np.max(fit.n_iter_)>=fit.max_iter:
                    raise RuntimeError("Sparse logistic solver has not converged")
                selected[name]=int(np.count_nonzero(fit.coef_))
            diagnostics.append(selected)
        metrics={}
        for name,p in probs.items():
            assert np.isfinite(p).all()
            for label,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                ok=(score>=.5)==y
                bad=(base>=.5)!=y
                metrics[label]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),harmed=int((~bad&~ok).sum()),
                                    remaining356_correct=int(ok[mask].sum()))
        np.savez_compressed(out/f"{seed}.npz",row=dev,fold=fold,y=y,base=base,**probs)
        results[str(seed)]=dict(metrics=metrics,selected=diagnostics)
        print(seed,metrics,flush=True)
    baseline=[]; learned=[]; standalone=[]
    for seed in SEEDS:
        with np.load(out/f"{seed}.npz") as z:
            baseline.append((z["base"]>=.5)==y)
            learned.append(((.75*z["base"]+.25*z["rules"])>=.5)==y)
            standalone.append((z["rules"]>=.5)==y)
    a,b=np.asarray(baseline),np.asarray(learned)
    delta=np.mean(b.astype(float)-a,axis=0)
    _,idx=np.unique(groups,return_inverse=True)
    sizes=np.bincount(idx)
    gains=np.bincount(idx,weights=delta)
    rng=np.random.default_rng(2231)
    draws=[]
    for _ in range(100):
        freq=rng.multinomial(len(sizes),np.full(len(sizes),1/len(sizes)),size=40)
        draws.extend(100*(freq@gains)/(freq@sizes))
    results["stability"]=dict(mean_delta_pp=float(100*delta.mean()),
        paired_booking_95_pp=np.quantile(draws,[.025,.975]).tolist(),
        always_fixed=int((~a.any(0)&b.all(0)).sum()),always_harmed=int((a.all(0)&~b.any(0)).sum()),
        standalone_any_seed_356=int((np.any(standalone,axis=0)&mask).sum()),
        standalone_every_seed_356=int((np.all(standalone,axis=0)&mask).sum()))
    write(ROOT/"reports/nov23_sparse_rules/result.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
