"""Conditional logistic regression on complete matched sets of six passenger bills."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, logsumexp
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_boosting_path_n23 import calibrated
from broad_error_vote_tables_n23 import bank
from broad_geographic_transfer_n23 import BILLS
from broad_partition_evidence_n23 import ROOT, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development

SIZE = 6


def possible_assignments(k):
    ans = np.zeros((len(list(combinations(range(SIZE), k))), SIZE))
    for i, subset in enumerate(combinations(range(SIZE), k)):
        ans[i, list(subset)] = 1.
    return ans


def matched_sets(bills, seed, matched):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bills))
    if not matched:
        return order[:len(order)//SIZE*SIZE].reshape(-1,SIZE)
    available = np.ones(len(order),bool)
    out = []
    for person in order:
        if not available[person]:
            continue
        ids = np.flatnonzero(available)
        if len(ids) < SIZE:
            break
        # Randomize exact ties instead of grouping passengers by source row order.
        d = np.square(bills[ids]-bills[person]).sum(1) + rng.uniform(0,1e-9,len(ids))
        d[ids==person] = -1.
        selected = ids[np.argpartition(d,SIZE-1)[:SIZE]]
        out.append(selected)
        available[selected] = False
    return np.stack(out)


def fit_conditional(design, y, sets):
    cells = []
    n = y[sets].sum(1)
    for k in range(1,SIZE):
        ids = sets[n==k]
        if len(ids):
            cells.append((ids, possible_assignments(k)))
    def loss(coef):
        score = design @ coef
        grad = np.zeros(len(y))
        total = 0.
        for ids, possibilities in cells:
            logits = score[ids] @ possibilities.T
            norm = logsumexp(logits,axis=1)
            total += np.sum(norm - (score[ids]*y[ids]).sum(1))
            grad[ids] = np.exp(logits-norm[:,None]) @ possibilities - y[ids]
        return total + 3.*(coef@coef), np.asarray(design.T @ grad).ravel() + 6.*coef
    opt = minimize(loss, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter":400,"ftol":1e-10})
    assert opt.success, opt.message
    return opt.x, sum(i.size for i,_ in cells)


def run():
    rows,y,frame,groups,_ = load()
    target,old,historical = development()
    np.testing.assert_array_equal(y,target)
    missed = np.logical_and.reduce([p!=y for p in old.values()])
    state = strata(frame)
    paid = missed & (state==2) & y
    zero = missed & (state==0) & ~y
    masks = dict(release=missed,paid=paid,zero=zero,exception=paid|zero,repeat=historical["repeat"])
    assert [m.sum() for m in (missed,paid,zero)]==[941,415,281]
    raw = frame[BILLS].to_numpy(float)
    features = pd.get_dummies(frame.drop(columns=BILLS)).astype(float).fillna(-1).to_numpy()
    all_hits,base_hits,counts = {},[],{}
    for seed in SEEDS:
        fold,base,ref,_,_ = bank(str(seed),rows,y)
        base_hits.append(ref==y)
        outputs = {}
        counts[seed] = []
        for f in np.unique(fold):
            t,q = np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t],groups[q]).size
            trees = RandomTreesEmbedding(n_estimators=64,max_depth=4,random_state=252,n_jobs=1)
            train_x = trees.fit_transform(features[t])
            query_x = trees.transform(features[q])
            # Explicit paid/zero/missing indicators retain economic distinctions at small amounts.
            a = StandardScaler().fit_transform(np.log1p(np.nan_to_num(raw[t])))
            bill = np.c_[a,np.isnan(raw[t])*2.,(raw[t]>0)*2.]
            scores = {}
            info = {}
            for model in ("bill_sets","random_sets"):
                sets = matched_sets(bill,882,model=="bill_sets")
                assert np.unique(sets).size == sets.size
                coef,used = fit_conditional(train_x,y[t],sets)
                scores[model] = query_x @ coef
                info[model] = dict(used=used,total=len(t),mean_square_bill_gap=float(np.square(
                    bill[sets]-bill[sets].mean(axis=1,keepdims=True)).sum(axis=2).mean()))
            counts[seed].append(info)
            for a,b in GroupKFold(3,shuffle=True,random_state=439).split(q,groups=groups[q]):
                assert not np.intersect1d(groups[q[a]],groups[q[b]]).size
                for name,z in scores.items():
                    state_cols = np.c_[state[q]==0,state[q]==1]
                    register = np.c_[z, state_cols, z[:,None]*state_cols]
                    anchor = logit(base[q].clip(.001,.999))
                    designs = {"alone_"+name:register, "joint_"+name:np.c_[anchor,register],
                               "anchor":np.c_[anchor,state_cols]}
                    for key,design in designs.items():
                        out = outputs.setdefault(key,np.full(len(y),np.nan))
                        out[q[b]] = calibrated(design[a],design[b],y,q[a])
        for key,v in outputs.items():
            assert np.isfinite(v).all()
            all_hits.setdefault(key,[]).append((v>=.5)==y)
    base_hits=np.stack(base_hits)
    result={k:summarize(np.stack(v),base_hits,masks) for k,v in all_hits.items()}
    result["baseline"]=summarize(base_hits,base_hits,masks)
    write(ROOT/"reports/nov23_classification_context/results.json",dict(models=result,matching=counts))
    print(result,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
