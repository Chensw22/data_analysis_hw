"""Train nonlinear service-specific predictions against the outcome of the whole passenger."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from exaone_validation_data import CATEGORICAL


def expand(frame):
    amounts=frame[BILLS].to_numpy(float)
    present=np.nan_to_num(amounts)
    missing=np.isnan(amounts).sum(1)
    n=len(frame)
    active=np.column_stack([np.ones(n,bool),present>0,missing>0])
    owner,service=np.nonzero(active)
    all_amounts=np.column_stack([np.zeros(n),present,np.zeros(n)])
    fee=all_amounts[owner,service]
    person=frame.drop(columns=BILLS).copy()
    person['TotalSpend']=present.sum(1)
    person['PositiveServices']=(present>0).sum(1)
    person['MissingServices']=missing
    data=person.iloc[owner].reset_index(drop=True)
    data['Service']=np.array(['Cabin',*BILLS,'Unknown'])[service]
    data['Fee']=fee
    data['Share']=fee/np.maximum(present.sum(1)[owner],1)
    weight=(1+fee)**(1/7)
    weight/=np.bincount(owner,weights=weight)[owner]
    np.testing.assert_allclose(np.bincount(owner,weights=weight),1.)
    return owner,data,weight


def fit(frame,y,weight):
    model=CatBoostClassifier(iterations=260,depth=5,learning_rate=.065,l2_leaf_reg=5,one_hot_max_size=32,
                             thread_count=2,random_seed=693,verbose=False,allow_writing_files=False)
    cats=[k for k in [*CATEGORICAL,'Service'] if k in frame]
    model.fit(frame,y,sample_weight=weight,cat_features=cats)
    return model


def run_fold(frame,groups,y,t,q):
    owner,data,prior=expand(frame)
    train=np.isin(owner,t)
    query=np.isin(owner,q)
    it,iq=owner[train],owner[query]
    xtrain,xquery=data[train],data[query]
    weights=prior[train]
    result={'direct':fit(frame.iloc[t],y[t],np.ones(len(t))).predict_proba(frame.iloc[q])[:,1]}
    for k in range(3):
        model=fit(xtrain,y[it],weights)
        p=model.predict_proba(xquery)[:,1]
        if k in (0,2):
            total=np.bincount(iq,weights=prior[query]*p,minlength=len(frame))
            result['initial' if k==0 else 'mixture']=total[q]
        if k<2:
            training=model.predict_proba(xtrain)[:,1].clip(.002,.998)
            weights=prior[train]*np.where(y[it],training,1-training)
            weights/=np.bincount(it,weights=weights,minlength=len(frame))[it]
            np.testing.assert_allclose(np.bincount(it,weights=weights,minlength=len(frame))[t],1.)
    return result,dict(train_instances=len(it),training_people=len(t),scored_instances=len(iq))


if __name__=='__main__':
    with threadpool_limits(2):
        evaluate('service_bag_trees',run_fold,REPORT)
