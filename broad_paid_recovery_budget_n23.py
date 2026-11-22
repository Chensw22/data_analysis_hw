"""Optimistic limits and group-held-out decisions for positive-only payer corrections."""
import numpy as np
from sklearn.model_selection import GroupKFold
from broad_service_transfer_n23 import ART,REPORT,BILLS,SEEDS,historical,load,prepare,write

SOURCES={
    "broad_basket_balance":["basket","weighted","placebo","coarse_native","restored"],
    "service_assignment":["unordered","factor_real","real"],
    "paid_register_likelihood":["bill","forward","reverse"],
    "broad_service_transfer":["update_native","update_other","update_aligned","update_pooled"]
}


def ranks(name,row,y):
    bank={}
    for source,keys in SOURCES.items():
        with np.load(ART/f"nov23_{source}/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            if not bank:
                fold,base,ref=z["fold"],z["base"],z["ref"]
                bank["base"]=base
            for k,v in (("fold",fold),("base",base),("ref",ref)):
                np.testing.assert_array_equal(z[k],v)
            for key in keys:
                bank[source+"/"+key]=z[key]
    return list(bank),np.asarray(list(bank.values())).T,fold,base,ref


def options(ids,values,y,hard=None):
    opts=[]
    for j,col in enumerate(values.T):
        for direction in (1,-1):
            v=direction*col[ids]
            order=np.argsort(-v,kind="stable")
            last=np.r_[v[order][1:]!=v[order][:-1],True]
            ix=ids[order]
            cuts=v[order][last]
            yes=np.cumsum(y[ix],dtype=int)[last]
            no=np.cumsum(~y[ix],dtype=int)[last]
            hits=None if hard is None else np.cumsum(hard[ix],dtype=int)[last]
            opts.append((j,direction,cuts,yes,no,hits))
    return opts


def best(opts):
    best_fix,best_harm,rule=0,0,None
    for j,direction,cuts,yes,no,_ in opts:
        gain=yes-no
        i=int(np.argmax(gain))
        if (gain[i],-no[i])>(best_fix-best_harm,-best_harm):
            best_fix,best_harm,rule=int(yes[i]),int(no[i]),(j,direction,float(cuts[i]))
    return best_fix,best_harm,rule


def budget(per_service,limit,metric):
    # For each exact number of false positives, retain the most fixes possible.
    dp=np.full(limit+1,-99999,int)
    dp[0]=0
    for opts in per_service:
        local=np.full(limit+1,-99999,int)
        local[0]=0
        for _,_,_,yes,no,hits in opts:
            keep=no<=limit
            np.maximum.at(local,no[keep],(hits if metric=="repeat" else yes)[keep])
        nxt=dp.copy()
        for n in np.flatnonzero(local>=0):
            nxt[n:]=np.maximum(nxt[n:],dp[:limit+1-n]+local[n])
        dp=nxt
    return np.maximum.accumulate(dp)


def applied(ids,rule,values):
    if rule is None:
        return np.zeros(len(ids),bool)
    j,direction,cut=rule
    return direction*values[ids,j]>=cut


def solve():
    row,y,frame,group,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    dom=prepare(frame)[0]
    hard=masks["repeat"]&y&(dom>=0)
    assert hard.sum()==428
    plans,all_hits={},[]
    for name in [*map(str,SEEDS),"v131"]:
        keys,values,fold,base,ref=ranks(name,row,y)
        assert np.isfinite(values).all()
        eligible=(dom>=0)&~ref&(ref==(base>=.5))
        census=[]
        selections=[]
        oracle=np.zeros(len(y),bool)
        for j,s in enumerate(BILLS):
            ids=np.flatnonzero(eligible&(dom==j))
            opts=options(ids,values,y,hard)
            selections.append(opts)
            yes,no,rule=best(opts)
            oracle[ids]=applied(ids,rule,values)
            census.append(dict(service=s,eligible=len(ids),missed=int(y[ids].sum()),oracle_fixed=yes,
                               oracle_harmed=no,ranker=None if rule is None else keys[rule[0]]))
        pred=np.zeros(len(y),bool)
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=911).split(ids,groups=group[ids]):
                train,query=ids[a],ids[b]
                assert not np.intersect1d(group[train],group[query]).size
                for j in range(5):
                    t=train[eligible[train]&(dom[train]==j)]
                    q=query[eligible[query]&(dom[query]==j)]
                    if len(t) and len(q):
                        _,_,rule=best(options(t,values,y))
                        pred[q]=applied(q,rule,values)
        assert not np.any((pred|oracle)&~eligible)
        base_right=int((ref==y).sum())
        report={}
        for k,act in (("hindsight",oracle),("held_bookings",pred)):
            fixed,harmed=int((act&y).sum()),int((act&~y).sum())
            report[k]=dict(fixed=fixed,harmed=harmed,accuracy=(base_right+fixed-harmed)/len(y),
                           repeat_paid_fixed=int((act&hard).sum()))
        assert sum(d["oracle_fixed"] for d in census)==report["hindsight"]["fixed"]
        assert sum(d["oracle_harmed"] for d in census)==report["hindsight"]["harmed"]
        limits=[0,25,50,100,200]
        front={k:dict(zip(map(str,limits),budget(selections,max(limits),k)[limits].tolist()))
               for k in ("all","repeat")}
        plans[name]=dict(comparison=report,needed=int(np.floor(.85*len(y))+1-base_right),
                         frontier=front,groups=census)
        if name!="v131":
            all_hits.append(pred)
    write(REPORT/"recovery_budget.json",dict(rankers=keys,plans=plans,
          stable_learned_paid=int((np.logical_and.reduce(all_hits)&hard).sum())))


if __name__=="__main__":
    solve()
