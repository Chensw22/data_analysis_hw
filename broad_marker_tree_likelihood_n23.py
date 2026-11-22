"""Integrate all tree dependence structures for five observed passenger markers."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_companion_information_n23 import ART,REPORT,SEEDS,historical,load,strata,evaluate,write,calibrated
from broad_geographic_transfer_n23 import BILLS

OUT=ART/"nov23_marker_tree_likelihood"


def variables(frame):
    bill=frame[BILLS].fillna(0).to_numpy()
    primary=np.where(bill.max(1)>0,np.argmax(bill,axis=1)+1,0)
    marks=np.column_stack([pd.factorize(frame[k],sort=True)[0] for k in ("HomePlanet","CryoSleep","Side","Destination")]+[primary])
    fixed=pd.get_dummies(frame[["Deck","Age","VIP","CabinNumber","BookingSize"]],columns=["Deck","VIP"]).fillna(-1)
    x=np.column_stack([fixed,np.log1p(np.sort(bill,axis=1)),frame[BILLS].isna().sum(axis=1)])
    return x,marks


def complete_tree(logs,ratios):
    # Kirchhoff's determinant sums the pair-factor products over all labeled trees.
    n=logs.shape[1]
    mat=-ratios.copy()
    idx=np.arange(n)
    mat[:,idx,idx]=ratios.sum(2)
    sign,val=np.linalg.slogdet(mat[:,1:,1:])
    assert (sign>0).all()
    return logs.sum(1)+val-(n-2)*np.log(n)


def frequencies(x,marks,y,t,q):
    sizes=marks.max(0)+1
    n=len(sizes)
    pairs=list(combinations(range(n),2))
    sets=[(),*[(j,) for j in range(n)],*pairs,tuple(range(n))]
    codes={}
    widths={}
    for s in sets:
        widths[s]=int(np.prod(sizes[list(s)]))
        codes[s]=np.ravel_multi_index(marks[:,s].T,tuple(sizes[list(s)])) if s else np.zeros(len(y),int)
    model=ExtraTreesClassifier(n_estimators=48,max_features=.8,min_samples_leaf=30,random_state=753,n_jobs=2)
    model.fit(x[t],np.column_stack([y[t],marks[t]]))
    train,query=model.apply(x[t]),model.apply(x[q])
    tables={s:np.zeros((len(q),2)) for s in sets}
    averaged={s:np.zeros_like(arr) for s,arr in tables.items()}
    for k in range(train.shape[1]):
        nodes,group=np.unique(train[:,k],return_inverse=True)
        qt=np.searchsorted(nodes,query[:,k])
        np.testing.assert_array_equal(nodes[qt],query[:,k])
        support=np.bincount(group)[qt]
        for s in sets:
            m=widths[s]
            state=y[t].astype(int)*m+codes[s][t]
            index=codes[s][q,None]+np.array([0,m])
            if k==0:
                counts=np.bincount(state,minlength=2*m)
                averaged[s]=.09*counts[index]/len(t)+.01/(2*m)
            counts=np.bincount(group*(2*m)+state,minlength=len(nodes)*2*m).reshape(-1,2*m)
            tables[s]+=counts[qt[:,None],index]/support[:,None]
    for s in sets:
        tables[s]=.9*tables[s]/train.shape[1]+averaged[s]
    prior=tables[()]
    np.testing.assert_allclose(prior.sum(1),1.,atol=1e-12)
    ll={k:np.zeros((len(q),2)) for k in ("independent","tree","full")}
    for c in (0,1):
        single=np.column_stack([tables[(j,)][:,c] for j in range(n)])/prior[:,[c]]
        ratio=np.zeros((len(q),n,n))
        for i,j in pairs:
            ratio[:,i,j]=ratio[:,j,i]=tables[(i,j)][:,c]/(prior[:,c]*single[:,i]*single[:,j])
        ll["independent"][:,c]=np.log(single).sum(1)
        ll["tree"][:,c]=complete_tree(np.log(single),ratio)
        ll["full"][:,c]=np.log(tables[tuple(range(n))][:,c]/prior[:,c])
    out={k:expit(logit(prior[:,1])+v[:,1]-v[:,0]) for k,v in ll.items()}
    out["context"]=prior[:,1]
    return out


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    x,marks=variables(frame)
    direct=np.column_stack([x,*[np.eye(c.max()+1)[c] for c in marks.T]])
    scope=dict(**masks,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy(),
               repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    OUT.mkdir(exist_ok=True)
    plans,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_companion_information/{name}.npz") as z:
            for k,v in (("row",row),("y",y)): np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            sub=frequencies(x,marks,y,t,q)
            clf=HistGradientBoostingClassifier(max_iter=190,max_leaf_nodes=18,l2_regularization=4,
                         min_samples_leaf=22,early_stopping=False,random_state=918)
            sub["direct"]=clf.fit(direct[t],y[t]).predict_proba(direct[q])[:,1]
            for k,p in sub.items():
                scores.setdefault(k,np.full(len(y),np.nan))[q]=p.clip(1e-8,1-1e-8)
        for k,keys in (("direct",["direct"]),("separate",["direct","independent"]),
                       ("dependent",["direct","independent","tree","full"])):
            xcal=np.column_stack([logit(v.clip(.001,.999)) for v in [base,*[scores[v] for v in keys]]])
            p=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=571).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(xcal[t],xcal[q],y,t)
            scores["update_"+k]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scope)
        plans[name]=result
        if name!="v131":
            for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        print(name,result["tree"]["accuracy"],result["update_dependent"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in history.items()}
    write(REPORT/"marker_dependence.json",dict(plans=plans,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
