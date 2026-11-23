"""Separate systematic changes in passenger predictions from training-subset variability."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_billing_response_surface_n23 import REPORT
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from broad_support_trajectories_n23 import LEVEL, SUBSETS
from released_correction_audit_n23 import development


def effects(scores):
    index={tuple(v):i for i,v in enumerate(SUBSETS)}
    shifts=[]
    for level in (1,2,3):
        changes=[]
        for i,s in enumerate(SUBSETS):
            if len(s)==level:
                for j in set(range(4))-set(s):
                    changes.append(scores[index[tuple(sorted((*s,j)))]]-scores[i])
        dif=np.stack(changes)
        shifts.extend([dif.std(0), (dif>0).mean(0)-.5, np.mean(abs(dif),axis=0)])
    return np.column_stack(shifts)


def decomposition(p,y,m):
    # Choose one training subset uniformly. Its Brier score equals the squared
    # error of the mean prediction plus the conditional variance exactly.
    p=p[:,m]
    label=y[m]
    mean=p.mean(0)
    risk=np.mean((p-label)**2)
    err=np.mean((mean-label)**2)
    variance=p.var(0).mean()
    np.testing.assert_allclose(err+variance,risk,atol=1e-12)
    return dict(random_subset_accuracy=float(((p>=.5)==label).mean()),
                ensemble_accuracy=float(((mean>=.5)==label).mean()),mean_brier=float(risk),
                ensemble_brier=float(err),variance=float(variance))


def run():
    row,y,frame,g,_=load()
    labels,release,scopes=development()
    np.testing.assert_array_equal(labels,y)
    miss=np.logical_and.reduce([p!=y for p in release.values()])
    state=strata(frame)
    core=miss&(((state==2)&y)|((state==0)&~y))
    masks=dict(all=np.ones(len(y),bool),core=core,other=~core,miss=miss,repeat=scopes["repeat"])
    assert [masks[k].sum() for k in ("core","miss","repeat")]==[696,941,1070]
    report,history,base_hits,summary={}, {}, [], {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row);np.testing.assert_array_equal(z["y"],y)
            base,fold=z["base"],z["fold"]
        with np.load(ART/f"nov23_broad_support_trajectories/{seed}.npz") as z:
            for k,v in (("row",row),("y",y),("fold",fold),("subsets",LEVEL)):
                np.testing.assert_array_equal(z[k],v)
            p=z["bank"]
        assert p.shape==(15,len(y)) and np.isfinite(p).all()
        report[seed]={k:[decomposition(p[LEVEL==n],y,m) for n in (1,2,3,4)] for k,m in masks.items()}
        base_hits.append((base>=.5)==y)
        avg=np.stack([p[LEVEL==level].mean(0) for level in (1,2,3,4)],axis=1)
        # Compare how often full agrees with truth when at least one individual
        # half-fit dissents vs when the averaged half-fit itself dissents.
        opp={}
        full=p[-1]>=.5
        for k,sel in (("any_half",((p[LEVEL==2]>=.5)!=full).any(0)),("ensemble_half",(avg[:,1]>=.5)!=full)):
            good=(full!=y)
            opp[k]={s:dict(recoverable=int((sel&good&m).sum()),harmed_if_all=int((sel&~good&m).sum())) for s,m in masks.items()}
        summary[seed]=opp
        odds=logit(p.clip(.001,.999))
        main=np.c_[logit(base.clip(.001,.999)),logit(avg.clip(.001,.999))]
        spread=effects(odds)
        scores={k:np.full(len(y),np.nan) for k in ("full","averages","dispersion","shuffled")}
        for f in np.unique(fold):
            idx=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(idx,groups=g[idx]):
                t,q=idx[a],idx[b]
                assert not np.intersect1d(g[t],g[q]).size
                bins=state[t]*6+np.digitize(base[t],np.quantile(base[t],np.arange(1,6)/6))
                perm=np.arange(len(t))
                rng=np.random.default_rng(490)
                for c in np.unique(bins):
                    ix=np.flatnonzero(bins==c)
                    perm[ix]=rng.permutation(ix)
                for k,train,test in (("full",main[t][:,[0,-1]],main[q][:,[0,-1]]),
                                     ("averages",main[t],main[q]),
                                     ("dispersion",np.c_[main[t],spread[t]],np.c_[main[q],spread[q]]),
                                     ("shuffled",np.c_[main[t],spread[t][perm]],np.c_[main[q],spread[q]])):
                    model=LogisticRegression(C=.5,max_iter=500).fit(train,y[t])
                    scores[k][q]=model.predict_proba(test)[:,1]
        for k,s in scores.items():
            assert np.isfinite(s).all()
            history.setdefault(k,[]).append((s>=.5)==y)
    res={k:summarize(np.stack(v),np.stack(base_hits),masks) for k,v in history.items()}
    write(REPORT/"support_variance.json",dict(curves=report,opportunities=summary,combinations=res))
    for k in ("all","core","other"):
        halves=np.mean([[report[s][k][1][v] for v in ("random_subset_accuracy","ensemble_accuracy","variance")] for s in SEEDS],axis=0)
        full=np.mean([report[s][k][3]["ensemble_accuracy"] for s in SEEDS])
        print(k,halves,full)
    print(res)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
