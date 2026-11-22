"""Recover passenger probabilities by integrating learned differences between independent bookings."""
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS,load
from exaone_validation_data import CATEGORICAL
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, evaluate, interval, teachers, write
from class_analogue_experts_n23 import design

OUT=ART/"nov23_broad_passenger_differences"
REPORT=ROOT/"reports/nov23_broad_passenger_differences"


def neighbors(view, groups, count, balanced):
    data=np.ascontiguousarray(view,dtype=float)
    rng=np.random.default_rng(825)
    order=rng.permutation(len(data)) if balanced else np.arange(len(data))
    distances,relative=cKDTree(data[order]).query(data,k=count+10)
    indices=order[relative]
    # With many exact ties (especially zero spending), a k-d tree could
    # repeatedly choose the same few anchors. Sample tied people uniformly.
    _,codes,sizes=np.unique(data,axis=0,return_inverse=True,return_counts=True)
    for code in np.flatnonzero((sizes>count+10) & balanced):
        block=np.flatnonzero(codes==code)
        for person in block:
            indices[person]=rng.choice(block,count+10,replace=False)
            distances[person]=0
    valid=groups[indices]!=groups[:,None]
    take=valid&(valid.cumsum(1)<=count)
    assert np.all(take.sum(1)==count)
    adjacent=indices[take].reshape(-1,count)
    distance=distances[take].reshape(-1,count)
    scale=np.maximum(distance[:,-1,None],.015)
    weight=np.exp(-.5*(distance/scale)**2)/count
    rows=np.repeat(np.arange(len(view)),count)
    return coo_matrix((weight.ravel(),(rows,adjacent.ravel())),shape=(len(view),len(view))).tocsr()


def network(x,context,spending,groups,balanced):
    nets=[neighbors(v,groups,10,balanced) for v in (x,context,spending)]
    # Keep the total contribution of each neighborhood definition equal.
    edges=sum(nets)/len(nets)
    edges=(edges+edges.T)*.5
    edges=edges.tocoo()
    unique=edges.row<edges.col
    return edges.row[unique],edges.col[unique],edges.data[unique]


def estimator():
    return HistGradientBoostingRegressor(max_iter=95,max_leaf_nodes=15,min_samples_leaf=45,
                 learning_rate=.075,l2_regularization=5.,early_stopping=False,random_state=827)


def differences(x,left,right,known,truth):
    delta=x[right]-x[left]
    center=(x[right]+x[left])*.5
    both=known[left]&known[right]
    signed=np.column_stack([delta,center])
    reverse=np.column_stack([-delta,center])
    true_change=truth[right[both]]-truth[left[both]]
    model=estimator().fit(np.vstack([signed[both],reverse[both]]),np.r_[true_change,-true_change])
    return .5*(model.predict(signed)-model.predict(reverse))


def integrate(left,right,w,delta,known,observed):
    n=len(known)
    adjacency=coo_matrix((np.r_[w,w],(np.r_[left,right],np.r_[right,left])),shape=(n,n)).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel()
    pull=known.astype(float)+.002
    system=diags(degree+pull)-adjacency
    source=np.bincount(right,weights=w*delta,minlength=n)-np.bincount(left,weights=w*delta,minlength=n)
    rhs=source+known*observed+.002*observed[known].mean()
    result,info=cg(system,rhs,M=diags(1/(degree+pull)),rtol=1e-8,maxiter=400)
    assert info==0 and np.linalg.norm(system@result-rhs)<1e-6*(1+np.linalg.norm(rhs))
    return result


def one_fold(frame,g,y,t,q,base,seed,balanced):
    assert not np.intersect1d(g[t],g[q]).size
    x,_=design(frame,t)
    x=np.column_stack([x,np.log(frame.BookingSize)])
    # Build one view of registration and location, and one of all five bills.
    # The coordinate scales are learned exclusively on training rows.
    context=pd.get_dummies(frame[CATEGORICAL+["Age","CabinNumber","BookingSize"]],columns=CATEGORICAL).fillna(-1).to_numpy(float)
    amounts=frame[BILLS]
    spend=np.column_stack([np.log1p(amounts.fillna(0)),amounts.gt(0),amounts.isna()])
    ids=np.r_[t,q]
    def scaled(v):
        return ((v[ids]-v[t].mean(0))/v[t].std(0).clip(.4)).clip(-5,5).astype(np.float32)
    entire=scaled(x)
    i,j,w=network(entire,scaled(context),scaled(spend),g[ids],balanced)
    known=np.arange(len(ids))<len(t)
    pt,_=teachers(frame,y,g,t,q,seed)
    prior=np.r_[pt,base[q]]
    result,check={},{}
    for key,offset in (("native",np.zeros(len(ids))),("update",prior)):
        observed=np.zeros(len(ids))
        observed[known]=y[t].astype(float)-offset[known]
        grad=differences(entire,i,j,known,observed)
        local=estimator().fit(entire[known],observed[known]).predict(entire)
        query_edges=~known[i]&~known[j]
        # Query outcomes below are used exclusively to report accuracy of
        # differences, after the learned edge field has been determined.
        target=y[ids].astype(float)-offset
        measured=(target[j]-target[i])[query_edges]
        for label,d in (("flat",np.zeros_like(grad)),("direct",local[j]-local[i]),("pair",grad)):
            score=integrate(i,j,w,d,known,observed)
            result[key+"_"+label]=(offset[~known]+score[~known]).clip(.001,.999)
            check[key+"_"+label]=float(np.mean((measured-d[query_edges])**2))
        result[key+"_no_graph"]=(offset[~known]+local[~known]).clip(.001,.999)
    return result,dict(mse=check,query_edges=int(query_edges.sum()),edges=len(i))


def run(balanced):
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(labels,y)
    directory = OUT/"balanced" if balanced else OUT
    directory.mkdir(parents=True,exist_ok=True)
    report_path = REPORT/("balanced.json" if balanced else "results.json")
    report,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for k,val in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],val)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        bank,checks={},[]
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(g,g[q]))
            outputs,check=one_fold(frame,g,y,t,q,base,115+int(k),balanced)
            checks.append(check)
            for label,p in outputs.items():
                bank.setdefault(label,np.full(len(y),np.nan))[q]=p
            print(name,"fold",k,flush=True)
        result,hits=evaluate(y,base,ref,bank,fold,name=="v131",masks)
        report[name]=dict(comparison=result,edge_checks=checks)
        if name!="v131":
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        write(report_path,dict(plans=report))
    stable={}
    for name,rows in history.items():
        good=np.logical_and.reduce([h for h,_ in rows])
        stable[name]={k:int((good&m).sum()) for k,m in masks.items()}
        stable[name]["paired_gain_ci95_pp"]=interval(np.mean([d for _,d in rows],0),g)
    write(report_path,dict(plans=report,stable=stable))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-ties",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(2):
        run(not args.original_ties)
