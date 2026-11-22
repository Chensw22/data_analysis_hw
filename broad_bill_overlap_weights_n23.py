"""Separate reliance on any purchase from reliance on the detailed five-service basket."""
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART, ROOT, SEEDS, historical, load, strata, evaluate, write, model
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS
from broad_refit_dissent_n23 import features

REPORT = ROOT/"reports/nov23_broad_basket_balance"
OUT = ART/"nov23_broad_basket_balance"


def basket(frame):
    b=frame[BILLS].to_numpy(float)
    amounts=np.nan_to_num(b)
    total=amounts.sum(1)
    shares=amounts/np.maximum(total[:,None],1.)
    return np.column_stack([np.log1p(amounts), np.isnan(b), shares, np.log1p(total),
                            (amounts>0).sum(1)])


def estimate_bill_score(b,y,g,t,q):
    p=np.full(len(t),np.nan)
    for a,v in GroupKFold(3,shuffle=True,random_state=304).split(t,groups=g[t]):
        assert not np.intersect1d(g[t[a]],g[t[v]]).size
        p[v]=model(120).fit(b[t[a]],y[t[a]]).predict_proba(b[t[v]])[:,1]
    assert np.isfinite(p).all()
    test=model(120).fit(b[t],y[t]).predict_proba(b[q])[:,1]
    return p,test


def placebos(weights,state,y,rng):
    # Preserve totals by true label and by coarse bill status. The true arm
    # alone can prioritize an individual whose precise purchases are anomalous.
    fake=weights.copy()
    for cell in np.unique(state*2+y):
        ids=np.flatnonzero((state*2+y)==cell)
        fake[ids]=rng.permutation(weights[ids])
    return fake


def main():
    row,y,frame,g,_=load()
    checked,labels,masks=historical()
    np.testing.assert_array_equal(row,checked)
    np.testing.assert_array_equal(y,labels)
    state=strata(frame)
    scope=dict(**masks,repeat_paid=masks["repeat"]&(state==2)&y,
               repeat_zero=masks["repeat"]&(state==0)&~y)
    x,b=features(frame)[0],basket(frame)
    OUT.mkdir(exist_ok=True)
    record,stable,votes={},{},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_bill_exception/{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            base,ref,fold,coarse = [z[k] for k in ("base","ref","fold","neutral")]
        scores={k:np.full(len(y),np.nan) for k in ("basket","weighted","placebo","restored")}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            p,scores["basket"][q]=estimate_bill_score(b,y,g,t,q)
            w=np.where(y[t],1-p,p)
            w/=w.mean()
            for key,weight in (("weighted",w),("placebo",placebos(w,state[t],y[t],np.random.default_rng(523)))):
                scores[key][q]=model(240).fit(x[t],y[t],sample_weight=weight).predict_proba(x[q])[:,1]
            scores["restored"][q]=expit(logit(scores["weighted"][q])+logit(scores["basket"][q]))
        for key,aux in (("coarse",coarse),("weighted",scores["weighted"]),("placebo",scores["placebo"])):
            matrix=np.column_stack([logit(v.clip(.001,.999)) for v in (base,scores["basket"],aux)])
            result=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,bv in GroupKFold(3,shuffle=True,random_state=351).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[bv]
                    assert not np.intersect1d(g[t],g[q]).size
                    result[q]=calibrated(matrix[t],matrix[q],y,t)
            scores["update_"+key]=result
        scores["coarse_native"]=coarse
        results,hits=evaluate(y,base,ref,scores,fold,name=="v131",scope)
        for k,(hit,gain) in hits.items():
            paid=state==2
            results[k]["paid_fixed_harmed"]=[int((paid&(gain==v)).sum()) for v in (1,-1)]
            results[k]["paid_new_false_positives"]=int((paid&~y&(gain<0)).sum())
            if name!="v131":
                votes.setdefault(k,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,base=base,ref=ref,fold=fold,**scores)
        record[name]=results
        print(name,results["weighted"]["fixed"],results["weighted"]["harmed"],
              results["update_weighted"]["accuracy"],flush=True)
    consensus={k:np.logical_and.reduce(h) for k,h in votes.items()}
    for k,v in consensus.items():
        stable[k]={s:int((m&v).sum()) for s,m in scope.items()}
    overlap={s:dict(new_only=int((m&consensus["weighted"]&~consensus["coarse_native"]).sum()),
                    coarse_only=int((m&~consensus["weighted"]&consensus["coarse_native"]).sum()),
                    both=int((m&consensus["weighted"]&consensus["coarse_native"]).sum()))
             for s,m in scope.items()}
    write(REPORT/"results.json",dict(plans=record,stable=stable,vs_coarse=overlap))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
