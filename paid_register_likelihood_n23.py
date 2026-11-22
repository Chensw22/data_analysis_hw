"""Factor the probability of a paid passenger's registration under each possible outcome."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, historical, strata, ART, ROOT, SEEDS, BILLS, evaluate, write

REPORT=ROOT/"reports/nov23_paid_register_likelihood"
OUT=ART/"nov23_paid_register_likelihood"


def register(row,x,g):
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{name}.csv",usecols=["PassengerId","Age","CryoSleep","HomePlanet","Cabin"])
                   for name in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0].astype(int)
    np.testing.assert_array_equal(group.to_numpy()[row],g)
    data={key:x[key].astype(str).to_numpy() for key in ("HomePlanet","Destination","Deck","Side","CryoSleep","VIP")}
    for key,arr,cuts in (("age",x.Age,[13,20,30,45,60]),("cabin_number",x.CabinNumber,[300,600,900,1200,1500]),
                         ("size",x.BookingSize,[2,3,5])):
        data[key]=np.where(arr.isna(),-1,np.digitize(arr,cuts))
    asleep=raw.CryoSleep.eq(True).astype(int)
    peer_sleep=(asleep.groupby(group).transform("sum")-asleep).to_numpy()[row]
    data["peer_sleep"]=np.minimum(peer_sleep,2)
    deck=raw.Cabin.str.split("/").str[0].fillna("unknown")
    companions=deck.groupby(group).transform("nunique").to_numpy()[row]
    data["deck_count"]=np.minimum(companions,3)
    return np.column_stack([pd.factorize(v)[0] for v in data.values()]),list(data)


def neighbors(distance):
    k=min(96,distance.shape[1])
    idx=np.argpartition(distance,k-1,axis=1)[:,:k]
    d=np.take_along_axis(distance,idx,axis=1)
    w=np.exp(-(d-d.min(1)[:,None])/(d.max(1)-d.min(1)).clip(.05)[:,None])
    return idx,w


def posterior(bills,reg,y,t,q):
    dist=cdist(bills[q],bills[t],"sqeuclidean")
    ids,w=neighbors(dist)
    prior=logit(((w*y[t[ids]]).sum(1)+1)/(w.sum(1)+2))
    out={"bill":prior}
    for name,order in (("forward",range(reg.shape[1])),("reverse",reversed(range(reg.shape[1])))):
        d=dist.copy()
        odds=np.zeros(len(q))
        for col in order:
            ids,w=neighbors(d)
            c=reg[t,col]
            target=reg[q,col]
            # Only the past fields enter the neighbor metric. The current
            # field's observed value supplies the likelihood for each outcome.
            size=int(reg[:,col].max()+1)
            match=c[ids]==target[:,None]
            post=[]
            pooled=((w*match).sum(1)+1)/(w.sum(1)+size)
            for cls in (False,True):
                wc=w*(y[t[ids]]==cls)
                post.append(((wc*match).sum(1)+6*pooled)/(wc.sum(1)+6))
            odds+=np.log(post[1]/post[0])
            d+=np.not_equal(target[:,None],c[None,:])
        out[name]=odds
    return out


def run():
    rows,y,x,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(rows,orig)
    np.testing.assert_array_equal(y,labels)
    paid=strata(x)==2
    scopes={**masks,"paid_missed":masks["repeat"]&paid&y}
    reg,fields=register(rows,x,g)
    bills=x[BILLS].to_numpy()
    z=np.column_stack([np.log1p(np.nan_to_num(bills))/2,np.isnan(bills),bills>0])
    report,history={},{}
    OUT.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_bill_exception/{name}.npz") as cache:
            for key,value in (("row",rows),("y",y)):
                np.testing.assert_array_equal(cache[key],value)
            fold,ref,base=cache["fold"],cache["ref"],cache["base"]
        bank={k:np.zeros(len(y)) for k in ("bill","forward","reverse")}
        for f in np.unique(fold):
            q=np.flatnonzero((fold==f)&paid)
            t=np.flatnonzero((fold!=f)&paid&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for k,p in posterior(z,reg,y,t,q).items():
                bank[k][q]=p
        direction=(bank["forward"]+bank["reverse"])/2
        standalone=expit(np.clip(bank["bill"]+direction,-20,20))
        scores={"update_native":np.where(paid,standalone,base)}
        eligible=paid&~ref&(ref==(base>=.5))
        risk=logit(np.clip(base,.001,.999))
        metrics={}
        for policy,inputs in (("base",[risk]),("bill",[risk,bank["bill"]]),
                              ("register",[risk,bank["bill"],bank["forward"],bank["reverse"]])):
            ds=np.column_stack(inputs)
            p=base.copy()
            selected=np.zeros(len(y),bool)
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&eligible)
                for a,b in GroupKFold(3,shuffle=True,random_state=342).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    m=LogisticRegression(C=.3,max_iter=300).fit(ds[t],y[t])
                    p[q]=m.predict_proba(ds[q])[:,1]
                    n=max(1,len(q)//10)
                    selected[q[np.argsort(-p[q],kind="stable")[:n]]]=True
            scores["update_"+policy]=p
            scores["update_top_"+policy]=np.where(selected,.75,base)
            metrics[policy]={"eligible":int(eligible.sum()),"top_n":int(selected.sum()),
                             "auc":float(roc_auc_score(y[eligible],p[eligible]))}
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        if name!="v131":
            for k,(ok,_) in hits.items():
                history.setdefault(k,[]).append(ok)
        np.savez_compressed(OUT/f"{name}.npz",row=rows,y=y,fold=fold,ref=ref,base=base,**bank,**scores)
        report[name]=dict(metrics=result,ranking=metrics)
        print(name,metrics,flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in history.items()}
    prior=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_broad_bill_exception/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],rows)
            prior.append((z["neutral"]>=.5)==y)
    both=np.logical_and.reduce(prior)
    overlap={}
    for k,values in history.items():
        selected=np.logical_and.reduce(values)&scopes["paid_missed"]
        overlap[k]=dict(solo=int((selected&(x.BookingSize==1)).sum()),
                        earth=int((selected&x.HomePlanet.eq("Earth")).sum()),
                        also_bill_balance=int((selected&both).sum()),
                        new_to_bill_balance=int((selected&~both).sum()))
    write(REPORT/"results.json",dict(plans=report,stable=stable,fields=fields,paid_recovered=overlap))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
