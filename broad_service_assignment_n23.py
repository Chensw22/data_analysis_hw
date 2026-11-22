"""Factor transport probability into an unordered basket and its service assignment."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART, ROOT, SEEDS, historical, load, strata, evaluate, write, model
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import CATEGORICAL

OUT = ART/"nov23_service_assignment"
REPORT = ROOT/"reports/nov23_broad_basket_balance"


def encode(frame):
    bills=frame[BILLS].to_numpy()
    known=np.isfinite(bills)
    purchase=np.nan_to_num(bills)
    amounts=np.sort(purchase,axis=1)[:,::-1]
    # Conditioning variables exclude the names of the services receiving money.
    # Which fields are observed is fixed for all assignments to a basket.
    reg=pd.get_dummies(frame.drop(columns=BILLS),columns=CATEGORICAL).fillna(-1.).to_numpy(float)
    blind=np.column_stack([reg,np.log1p(amounts),known,np.log1p(amounts.sum(1))])
    context=pd.get_dummies(frame[["HomePlanet","Deck","Side","Destination"]]).to_numpy(float)
    context=np.c_[np.ones(len(bills)),context,frame.Age.fillna(28)/35,
                  np.log1p(frame.BookingSize),np.log1p(amounts.sum(1))/7,(amounts>0).sum(1)/3]
    # Every step assigns a block of equal positive amounts to an unordered
    # subset of the remaining observed services. A last forced choice adds no information.
    events={}
    for i,vals in enumerate(purchase):
        available=known[i].copy()
        previous=np.zeros(5,bool)
        for amount in np.unique(vals[vals>0])[::-1]:
            chosen=vals==amount
            n=int(chosen.sum())
            if available.sum()>n:
                x=np.r_[context[i],np.log1p(amount)/7,amount/amounts[i].sum(),previous]
                events.setdefault(n,[]).append((i,x,available.copy(),chosen))
            available&=~chosen
            previous|=chosen
    chunks=[]
    for n,rows in events.items():
        idx,x,available,chosen=(np.asarray(v) for v in zip(*rows))
        choices=np.array([np.isin(np.arange(5),c) for c in combinations(range(5),n)],float)
        allowed=available.astype(int)@choices.T==n
        assert (allowed.sum(1)>1).all() and (chosen<=available).all()
        chunks.append((idx,x,allowed,chosen,choices))
    return blind,chunks


def subset(chunks,rows):
    result=[]
    for idx,x,allowed,chosen,choices in chunks:
        use=np.isin(idx,rows)
        if use.any():
            result.append((idx[use],x[use],allowed[use],chosen[use],choices))
    return result


def distribution(x,allowed,choices,coef):
    logits=(x@coef)@choices.T
    logits[~allowed]=-np.inf
    norm=logsumexp(logits,axis=1)
    return norm,np.exp(logits-norm[:,None])


def fit_choice(chunks,rows):
    parts=subset(chunks,rows)
    d=chunks[0][1].shape[1]
    def objective(param):
        coef=param.reshape(d,5)
        value=2*(param@param)
        grad=4*coef
        for _,x,allowed,chosen,choices in parts:
            norm,p=distribution(x,allowed,choices,coef)
            value+=np.sum(norm-np.sum((x@coef)*chosen,axis=1))
            grad+=x.T@(p@choices-chosen)
        return value,grad.ravel()
    fit=minimize(objective,np.zeros(d*5),jac=True,method="L-BFGS-B",
                 options={"maxiter":220,"ftol":1e-9})
    assert fit.success,fit.message
    return fit.x.reshape(d,5)


def likelihood(parts,coef,n):
    result=np.zeros(n)
    for idx,x,allowed,chosen,choices in parts:
        z,_=distribution(x,allowed,choices,coef)
        np.add.at(result,idx,np.sum((x@coef)*chosen,axis=1)-z)
    return result


def permute(y,t,state,counts,rng):
    labels=y.copy()
    cells=state*10+counts
    for k in np.unique(cells[t]):
        ids=t[cells[t]==k]
        labels[ids]=rng.permutation(y[ids])
    return labels


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    blind,chunks=encode(frame)
    state=strata(frame)
    paid=state==2
    scopes=dict(**masks,repeat_paid=masks["repeat"]&paid&y,
                repeat_zero=masks["repeat"]&(state==0)&~y)
    counts=frame[BILLS].gt(0).sum(axis=1).to_numpy()
    OUT.mkdir(exist_ok=True)
    records,histories={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_basket_balance/{name}.npz") as z:
            for k,v in (("row",row),("y",y)): np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        core=np.zeros(len(y))
        ratios={k:np.zeros(len(y)) for k in ("real","sham")}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            core[q]=model(240).fit(blind[t],y[t]).predict_proba(blind[q])[:,1]
            held=subset(chunks,q)
            for k,labels in (("real",y),("sham",permute(y,t,state,counts,np.random.default_rng(515)))):
                ll=[likelihood(held,fit_choice(chunks,t[labels[t]==label]),len(y))[q] for label in (False,True)]
                ratios[k][q]=ll[1]-ll[0]
        scores=dict(unordered=core)
        for k,delta in ratios.items():
            assert np.all(delta[~paid]==0)
            scores["factor_"+k]=expit((logit(core)+delta).clip(-25,25))
            p=base.copy()
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&paid)
                for a,b in GroupKFold(3,shuffle=True,random_state=412).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    x=np.column_stack([logit(v.clip(.001,.999)) for v in (base,core)]+[delta])
                    p[q]=calibrated(x[t],x[q],y,t)
            scores["update_"+k]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        for k,(hit,diff) in hits.items():
            result[k]["paying_false_positive_additions"]=int((paid&~y&(diff<0)).sum())
            result[k]["paying_false_negative_fixes"]=int((paid&y&(diff>0)).sum())
            if name!="v131":
                histories.setdefault(k,[]).append(hit)
        records[name]=result
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores,**ratios)
        print(name,[(k,v["fixed"],v["harmed"]) for k,v in result.items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in histories.items()}
    write(REPORT/"assignment.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
