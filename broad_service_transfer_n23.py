"""Test whether payers' transport evidence transfers across dominant service types."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_service_assignment_n23 import ART,ROOT,SEEDS,BILLS,CATEGORICAL,historical,load,evaluate,write,model,calibrated

REPORT=ROOT/"reports/nov23_broad_basket_balance"


def prepare(frame):
    b=frame[BILLS].fillna(0).to_numpy()
    order=np.argsort(-b,axis=1,kind="stable")
    values=np.take_along_axis(b,order,axis=1)
    total=b.sum(1)
    domain=np.where(total>0,order[:,0],-1)
    magnitude=np.column_stack([np.log1p(values),np.log1p(total),values[:,:2]/np.maximum(1,total[:,None])])
    register=pd.get_dummies(frame.drop(columns=BILLS),columns=CATEGORICAL).fillna(-1).to_numpy(float)
    second=np.eye(len(BILLS))[order[:,1]]*(values[:,[1]]>0)
    side=np.column_stack([register,(b>0).sum(1),frame[BILLS].isna().sum(axis=1),second])
    return domain,magnitude,side


def profile(domain,mag,side,t,q):
    ids=np.r_[t,q]
    rank=np.empty((len(ids),mag.shape[1]))
    for k in np.unique(domain[ids]):
        select=domain[ids]==k
        native=t[domain[t]==k]
        for j in range(mag.shape[1]):
            ref=np.sort(mag[native,j])
            val=mag[ids[select],j]
            # Tied zero amounts take their common mid-rank.
            rank[select,j]=(.5+np.searchsorted(ref,val,"left")*.5+np.searchsorted(ref,val,"right")*.5)/(1+len(ref))
    return np.column_stack([side[ids],mag[ids],rank])


def scores_for_fold(domain,mag,side,t,q,y):
    x=profile(domain,mag,side,t,q)
    train,query=x[:len(t)],x[len(t):]
    scores={k:np.zeros(len(q)) for k in ("native","other","aligned","pooled")}
    shared=model(170).fit(train,y[t])
    for k in np.unique(domain[q]):
        source=domain[t]!=k
        take=domain[q]==k
        sources=t[source]
        target=t[~source]
        fit=model(170).fit(train[source],y[sources])
        native=model(170).fit(train[~source],y[target])
        p=fit.predict_proba(query[take])[:,1]
        scores["native"][take]=native.predict_proba(query[take])[:,1]
        scores["other"][take]=p
        # Source and target training priors are available before predicting any held-out label.
        shift=logit((y[target].sum()+1)/(len(target)+2))-logit((y[sources].sum()+1)/(len(sources)+2))
        scores["aligned"][take]=expit(logit(p)+shift)
        scores["pooled"][take]=shared.predict_proba(query[take])[:,1]
    return scores


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    dom,mag,side=prepare(frame)
    paid=dom>=0
    high=np.isin(dom,[BILLS.index(k) for k in ("RoomService","Spa","VRDeck")])
    rare=masks["repeat"]&high&y
    scopes=dict(**masks,repeat_paid=masks["repeat"]&paid&y,repeat_three=rare)
    table={k:dict(n=int((dom==j).sum()),positives=int(((dom==j)&y).sum()),
                  repeat_missed=int(((dom==j)&scopes["repeat_paid"]).sum()))
           for j,k in enumerate(BILLS)}
    records,histories={},{}
    out=ART/"nov23_broad_service_transfer"
    out.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_service_assignment/{name}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            base,ref,fold=z["base"],z["ref"],z["fold"]
        scores={}
        for f in np.unique(fold):
            allq=np.flatnonzero(fold==f)
            q=allq[paid[allq]]
            t=np.flatnonzero((fold!=f)&paid&~np.isin(g,g[allq]))
            assert not np.intersect1d(g[t],g[allq]).size
            for k,p in scores_for_fold(dom,mag,side,t,q,y).items():
                scores.setdefault("update_"+k,base.copy())[q]=p
        logits=[logit(scores["update_"+k].clip(.001,.999)) for k in ("native","aligned","pooled")]
        for key,components in (("native_gate",[logits[0]]),("transfer_gate",logits)):
            ds=np.column_stack([logit(base.clip(.001,.999)),*components,np.eye(5)[dom]])
            p=base.copy()
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&paid)
                for a,b in GroupKFold(3,shuffle=True,random_state=467).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(ds[t],ds[q],y,t)
            scores["update_"+key]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        stats={}
        for k,(hit,delta) in hits.items():
            if name!="v131":
                histories.setdefault(k,[]).append(hit)
            stats[k]=dict(three_fn_fixed=int((high&y&(delta>0)).sum()),
                          three_fp_added=int((high&~y&(delta<0)).sum()),
                          within_service_auc={n:float(roc_auc_score(y[dom==j],scores[k][dom==j])) for j,n in enumerate(BILLS)})
        records[name]=dict(models=result,detail=stats)
        np.savez_compressed(out/f"{name}.npz",row=row,y=y,base=base,ref=ref,fold=fold,**scores)
        print(name,result["update_transfer_gate"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in histories.items()}
    write(REPORT/"transfer.json",dict(plans=records,stable=stable,dominant_service=table))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
