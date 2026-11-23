"""Fit passenger-level logistic loss while splitting trees on individual service observations."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from broad_opposing_passengers_n23 import evaluate
from broad_service_bag_trees_n23 import expand
from broad_service_pairs_n23 import REPORT


def design(data):
    cats=data.select_dtypes(exclude='number').columns
    numeric=data.select_dtypes(include='number')
    return np.column_stack([pd.get_dummies(data[cats],dtype=float).to_numpy(),
                            numeric.fillna(-1).to_numpy(),numeric.isna().to_numpy()]).astype(np.float32)


def assignments(tree,x,weights,owner,nodes,n_people):
    original=tree.apply(x)
    leaf=np.searchsorted(nodes,original)
    np.testing.assert_array_equal(nodes[leaf],original)
    a=np.zeros((n_people,len(nodes)))
    np.add.at(a,(owner,leaf),weights)
    np.testing.assert_allclose(a.sum(1),1.,atol=1e-10)
    return a,leaf


def boost(xt,xq,it,iq,wt,wq,y,nq,mode):
    # it and iq are local passenger indices; an observation's weight never exceeds
    # the single unit shared among all observations of the corresponding passenger.
    intercept=np.log((y.sum()+.5)/((~y).sum()+.5))
    ft=np.full(len(y),intercept)
    fq=np.full(nq,intercept)
    instance=np.full(len(it),intercept)
    instance_query=np.full(len(iq),intercept)
    for step in range(120):
        if mode=='row':
            p=expit(instance)
            v=(p*(1-p)).clip(.005)
            gradient=y[it].astype(float)-p
            weight=wt*v
            response=gradient/v
        else:
            p=expit(ft)
            v=(p*(1-p)).clip(.005)
            gradient=y.astype(float)-p
            weight=wt*v[it]
            response=(gradient/v)[it]
        tree=DecisionTreeRegressor(max_depth=4,min_samples_leaf=24,max_features=.8,random_state=354+step)
        tree.fit(xt,response,sample_weight=weight)
        nodes=np.flatnonzero(tree.tree_.children_left==-1)
        a,leaf=assignments(tree,xt,wt,it,nodes,len(y))
        b,query_leaf=assignments(tree,xq,wq,iq,nodes,nq)
        if mode=='row':
            grad=np.bincount(leaf,weights=wt*gradient,minlength=len(nodes))
            curvature=np.bincount(leaf,weights=weight,minlength=len(nodes))
            beta=grad/(curvature+2.)
        else:
            grad=a.T@gradient
            hessian=a.T@(v[:,None]*a)
            beta=(np.linalg.solve(hessian+2*np.eye(len(nodes)),grad) if mode=='joint' else grad/(hessian.diagonal()+2))
        update=.12*beta
        ft+=a@update
        fq+=b@update
        instance+=update[leaf]
        instance_query+=update[query_leaf]
    np.testing.assert_allclose(fq,np.bincount(iq,weights=wq*instance_query,minlength=nq),atol=1e-10)
    mean_p=np.bincount(iq,weights=wq*expit(instance_query),minlength=nq)
    return expit(fq),mean_p


def run_fold(frame,groups,y,t,q):
    owner,rows,prior=expand(frame)
    x=design(rows)
    train,query=np.isin(owner,t),np.isin(owner,q)
    order_t=np.argsort(t)
    order_q=np.argsort(q)
    it=order_t[np.searchsorted(t[order_t],owner[train])]
    iq=order_q[np.searchsorted(q[order_q],owner[query])]
    np.testing.assert_array_equal(t[it],owner[train])
    np.testing.assert_array_equal(q[iq],owner[query])
    out={}
    for k in ('row','diagonal','joint'):
        mixed,average=boost(x[train],x[query],it,iq,prior[train],prior[query],y[t],len(q),k)
        out['row_logit' if k=='row' else k]=mixed
        if k=='row':
            out['row_probability']=average
    return out,dict(passengers=len(t),service_records=len(it))


if __name__=='__main__':
    with threadpool_limits(1):
        evaluate('passenger_bag_boost',run_fold,REPORT)
