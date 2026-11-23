"""Test whether a passenger's recorded bills need different response functions across manifest contexts."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import BILLS, ROOT, ART, SEEDS, load, strata, write
from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_context_interaction_audit"


def designs(frame, state, train, query):
    register = pd.get_dummies(frame.drop(columns=BILLS),columns=CATEGORICAL,dtype=float)
    gate_cols = [c for c in register if c=="Age" or c.startswith(("CryoSleep_","Deck_"))]
    main = register.to_numpy(float)
    bill = np.log1p(frame[BILLS].to_numpy(float))
    gate = register[gate_cols].to_numpy(float)
    sham = gate.copy()
    rng = np.random.default_rng(416)
    for idx in (train,query):
        for s in np.unique(state[idx]):
            subset = idx[state[idx]==s]
            sham[subset] = gate[rng.permutation(subset)]
    x = np.column_stack([main,bill])
    r,b = set(range(main.shape[1])),set(range(main.shape[1],x.shape[1]))
    cols = set(range(x.shape[1],x.shape[1]+gate.shape[1]))
    return b, {"additive":(x,[r,b]),"unrestricted":(x,None),
               "age_sleep_deck":(np.c_[x,gate],[r,b|cols]),"shuffled":(np.c_[x,sham],[r,b|cols])}


def paired(p, previous, y, masks):
    hit, prior = (p>=.5)==y,(previous>=.5)==y
    return {k:dict(correct=int((hit&m).sum()),fixed=int((hit&~prior&m).sum()),
                   harmed=int((~hit&prior&m).sum())) for k,m in masks.items()}


def run():
    row,y,frame,g,_ = load()
    labels,released,historical = development()
    np.testing.assert_array_equal(labels,y)
    miss = np.logical_and.reduce([v!=y for v in released.values()])
    state = strata(frame)
    paid,zero = miss&(state==2)&y,miss&(state==0)&~y
    masks = dict(all=np.ones(len(y),bool),release=miss,core=paid|zero,
                 paid=paid,zero=zero,repeat=historical["repeat"])
    assert [masks[k].sum() for k in ("all","release","core")]==[6954,941,696]
    results, histories = {},{}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            base,fold = z["base"],z["fold"]
        scores = dict(baseline=base)
        contrast = {}
        for f in np.unique(fold):
            train,query = np.flatnonzero(fold!=f), np.flatnonzero(fold==f)
            assert not np.intersect1d(g[train],g[query]).size
            bill, models = designs(frame,state,train,query)
            for k,(x,constraints) in models.items():
                m = HistGradientBoostingClassifier(interaction_cst=constraints, max_iter=200,
                    max_leaf_nodes=15, learning_rate=.065, min_samples_leaf=20,
                    l2_regularization=4, early_stopping=False, random_state=442)
                m.fit(x[train], y[train])
                scores.setdefault(k,np.full(len(y),np.nan))[query] = m.predict_proba(x[query])[:,1]
                # A separable logit model has a zero rectangular contrast even
                # when bill combinations are exchanged between unrelated records.
                a,b = x[query], x[np.roll(query,1)]
                ab,ba = a.copy(),b.copy()
                cols=sorted(bill)
                ab[:,cols],ba[:,cols]=b[:,cols],a[:,cols]
                gaps=m.decision_function(a)+m.decision_function(b)-m.decision_function(ab)-m.decision_function(ba)
                if k=="additive":
                    assert abs(gaps).max()<1.e-10
                contrast.setdefault(k,[]).append(float(np.abs(gaps).mean()))
        for k in ("additive","age_sleep_deck","unrestricted","shuffled"):
            x = np.column_stack([logit(base.clip(.001,.999)),logit(scores["additive"].clip(.001,.999))])
            if k!="additive":
                x = np.c_[x,logit(scores[k].clip(.001,.999))-x[:,1]]
            out = np.full(len(y),np.nan)
            for f in np.unique(fold):
                outer = np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(outer,groups=g[outer]):
                    t,q = outer[a],outer[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    out[q]=LogisticRegression(C=.5,max_iter=500).fit(x[t],y[t]).predict_proba(x[q])[:,1]
            scores["stack_"+k]=out
        metrics = {}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            hit=(p>=.5)==y
            previous=scores["stack_additive" if k.startswith("stack_") else "additive"]
            metrics[k]=dict(accuracy=float(hit.mean()),vs_additive=paired(p,previous,y,masks),
                            vs_baseline=paired(p,base,y,masks))
            histories.setdefault(k,[]).append(hit)
        results[seed]=dict(metrics=metrics,mean_abs_rectangular_contrast=contrast)
        cache=ART/"nov23_context_interaction_audit"
        cache.mkdir(exist_ok=True)
        np.savez_compressed(cache/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**scores)
        print(seed,[(k,round(v["accuracy"],4)) for k,v in metrics.items()],flush=True)
    stable = {k:{s:int((np.logical_and.reduce(h)&m).sum()) for s,m in masks.items()} for k,h in histories.items()}
    uncertainty = {}
    for key,other in (("stack_age_sleep_deck","stack_additive"),("stack_age_sleep_deck","stack_shuffled"),
                      ("age_sleep_deck","additive")):
        gain = np.mean(histories[key],axis=0)-np.mean(histories[other],axis=0)
        uncertainty[key+"_vs_"+other] = {k:dict(delta_pp=float(100*gain[m].mean()),
            booking_bootstrap_95_pp=interval(gain[m],g[m])) for k,m in masks.items() if k in ("core","all")}
    write(REPORT/"results.json",dict(plans=results,stable=stable,contrasts=uncertainty))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
