"""Rank baseline mistakes above the correct cases most liable to be falsely reversed."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART,ROOT,SEEDS,historical,load,strata,write
from broad_refit_dissent_n23 import features

OUT=ART/"nov23_error_tail_ranking"
REPORT=ROOT/"reports/nov23_error_tail_ranking"
BUDGETS=(.01,.05,.1,.2)


def pair_loss(positive,negative,threshold,fraction):
    # Average squared pairwise margin loss for each negative in O(n log n).
    a=np.sort(positive)
    b=1+negative
    nr=len(a)
    k=np.searchsorted(a,b)
    prefix=np.r_[0.,np.cumsum(a)]
    squares=np.r_[0.,np.cumsum(a*a)]
    losses=(k*b*b-2*b*prefix[k]+squares[k])/(2*nr)
    if fraction==1:
        w=np.full(len(b),1/len(b))
        value=losses.mean()
        dt=0.
    else:
        smooth=.025
        exceed=(losses-threshold)/smooth
        value=threshold+smooth*np.logaddexp(0,exceed).mean()/fraction
        w=expit(exceed)/(fraction*len(b))
        dt=1-w.sum()
    dn=w*(k*b-prefix[k])/nr
    order=np.argsort(b)
    sb,sw=b[order],w[order]
    sw_total=np.r_[np.cumsum(sw[::-1])[::-1],0.]
    sb_total=np.r_[np.cumsum((sw*sb)[::-1])[::-1],0.]
    first=np.searchsorted(sb,positive,side="right")
    dp=-(sb_total[first]-positive*sw_total[first])/nr
    return value,dp,dn,dt


def train(x,err,fraction):
    xp,xn=x[err],x[~err]
    def objective(theta):
        w=theta[:-1]
        value,gp,gn,gt=pair_loss(xp@w,xn@w,theta[-1],fraction)
        return value+.025*(w@w),np.r_[gp@xp+gn@xn+.05*w,gt]
    fit=minimize(objective,np.r_[np.zeros(x.shape[1]),.5],jac=True,method="L-BFGS-B",
                 options={"maxiter":450,"ftol":1e-10})
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x[:-1]


def panel(name,row,y,frame):
    files={"observed_agreement":[s+"_"+k for s in ("sleep","earth","side","pay","retail") for k in ("agreement","direct")],
           "broad_neural_path":["epoch_80"],"broad_boosting_path":["trees_800"],
           "broad_basket_balance":["weighted"],"service_assignment":["factor_real"]}
    signals=[]
    for src,keys in files.items():
        with np.load(ART/f"nov23_{src}/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            if not signals:
                base,ref,fold=z["base"].clip(.001,.999),z["ref"],z["fold"]
            else:
                np.testing.assert_array_equal(z["ref"],ref)
                np.testing.assert_array_equal(z["fold"],fold)
                np.testing.assert_allclose(z["base"].clip(.001,.999),base,atol=1e-10)
            signals.extend([logit(z[k].clip(.001,.999)) for k in keys])
    side=np.where(ref,-1.,1.)
    risk=np.minimum(base,1-base)
    contra=np.column_stack(signals)*side[:,None]
    raw=features(frame)[0]
    x=np.c_[logit(risk),side,contra,(contra>0).mean(1),np.tanh(contra)*side[:,None],
            raw,raw*side[:,None]]
    return base,ref,fold,x


def percentile(x):
    return (rankdata(x)-.5)/len(x)


def metrics(action,ref,y,scope):
    err=ref!=y
    fixed=action&err
    harmed=action&~err
    return dict(fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                accuracy=float(((ref^action)==y).mean()),
                precision=float(fixed.sum()/action.sum()) if action.any() else None,
                recovered={k:int((fixed&m).sum()) for k,m in scope.items()})


def evaluate_splits(x,target,groups,fold,eligible):
    names=("base","global","tail")
    ranks={k:np.zeros(len(target)) for k in names}
    risks={k:np.zeros(len(target)) for k in names}
    for f in np.unique(fold):
        ids=np.flatnonzero((fold==f)&eligible)
        blocks=[ids[v] for _,v in GroupKFold(5,shuffle=True,random_state=851).split(ids,groups=groups[ids])]
        for k,q in enumerate(blocks):
            c=blocks[(k+1)%5]
            t=np.concatenate([block for j,block in enumerate(blocks) if j not in (k,(k+1)%5)])
            assert not np.intersect1d(groups[t],groups[c]).size
            assert not np.intersect1d(groups[q],groups[np.r_[t,c]]).size
            sc=StandardScaler().fit(x[t])
            xt,xc,xq=[sc.transform(x[part]).clip(-5,5) for part in (t,c,q)]
            for name in names:
                if name=="base":
                    rc,rq=x[c,0],x[q,0]
                else:
                    w=train(xt,target[t],1. if name=="global" else .1)
                    rc,rq=xc@w,xq@w
                pc,pq=percentile(rc),percentile(rq)
                # Estimate precision separately, without the ranker's training labels.
                cal=LogisticRegression(C=.8).fit(logit(pc)[:,None],target[c])
                ranks[name][q]=pq
                risks[name][q]=cal.predict_proba(logit(pq)[:,None])[:,1]
    return ranks,risks


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    scope=dict(**masks,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy(),
               repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    OUT.mkdir(exist_ok=True)
    plans,hist={},{}
    for name in [*map(str,SEEDS),"v131"]:
        base,ref,fold,x=panel(name,row,y,frame)
        eligible=ref==(base>=.5)
        wrong=y!=ref
        ranks,risks=evaluate_splits(x,wrong,g,fold,eligible)
        records={}
        bank={}
        for key,score in ranks.items():
            rule=(risks[key]>.5)&eligible
            options={"calibrated":rule}
            for share in BUDGETS:
                options[str(share)]=(score>=1-share)&eligible
            vals={k:metrics(act,ref,y,scope) for k,act in options.items()}
            vals["ranking"]=dict(auc=float(roc_auc_score(wrong[eligible],score[eligible])),
                         standardized_partial_auc=float(roc_auc_score(wrong[eligible],score[eligible],max_fpr=.1)))
            records[key]=vals
            bank[key+"_rank"]=score;bank[key+"_risk"]=risks[key]
            if name!="v131":
                for k,action in options.items():
                    hist.setdefault(key+"_"+k,[]).append((ref^action)==y)
        plans[name]=records
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,base=base,fold=fold,ref=ref,**bank)
        print(name,[(k,r["calibrated"]["accuracy"],r["0.1"]["precision"]) for k,r in records.items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in hist.items()}
    write(REPORT/"ranking.json",dict(plans=plans,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
