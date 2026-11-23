"""Estimate outcome probabilities for pairs of services and compare matched other passengers."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT
from broad_service_bag_trees_n23 import expand
from exaone_validation_data import CATEGORICAL

REPORT=ROOT/'reports/nov23_passenger_service_pairs'


def partner_bank(frame,groups,t,ids,k_donors):
    money=frame[BILLS]
    x=np.column_stack([np.log1p(money.sum(axis=1))/7,money.gt(0).sum(axis=1),money.isna().sum(axis=1),
                       frame.CryoSleep.eq('True').astype(int),frame.CryoSleep.eq('__MISSING__').astype(int)])
    pool=np.random.default_rng(328).permutation(t)
    k=np.unique(groups[t],return_counts=True)[1].max()+2+k_donors
    _,near=cKDTree(x[pool]).query(x[ids],k=k)
    available=groups[pool[near]]!=groups[ids,None]
    chosen=available & (available.cumsum(1)<=k_donors)
    assert (chosen.sum(1)==k_donors).all()
    donor=pool[near[chosen]].reshape(len(ids),k_donors)
    assert np.isin(donor,t).all() and (groups[donor]!=groups[ids,None]).all()
    return donor


def pair_table(owner,rows,prior,ids,donor):
    # expand() orders instances by their passenger; preserve native service weights
    # for each draw, including pairs of identical services.
    counts=np.bincount(owner)
    starts=np.r_[0,counts.cumsum()[:-1]]
    first,second,people=[],[],[]
    for a,b in zip(ids,donor):
        n,m=counts[a],counts[b]
        first.append(starts[a]+np.repeat(np.arange(n),m))
        second.append(starts[b]+np.tile(np.arange(m),n))
        people.append(np.full(n*m,a))
    left,right=np.concatenate(first),np.concatenate(second)
    people=np.concatenate(people)
    weight=prior[left]*prior[right]
    np.testing.assert_allclose(np.bincount(people,weights=weight,minlength=counts.size)[ids],1.)
    data=rows.iloc[left].reset_index(drop=True)
    data['SecondService']=rows.Service.to_numpy()[right]
    data['SecondFee']=rows.Fee.to_numpy()[right]
    data['SecondShare']=rows.Share.to_numpy()[right]
    return people,data,weight


def fit_pair(x,y,weight):
    fit=CatBoostClassifier(iterations=250,depth=5,learning_rate=.065,l2_leaf_reg=5,one_hot_max_size=32,
                           thread_count=2,random_seed=693,verbose=False,allow_writing_files=False)
    fit.fit(x,y,sample_weight=weight,cat_features=[c for c in [*CATEGORICAL,'Service','SecondService'] if c in x.columns])
    return fit


def run_fold(frame,groups,y,t,q):
    owner,rows,prior=expand(frame)
    answers={}
    notes={}
    ids=np.r_[t,q]
    for key,donors in [('same_passenger',ids),('matched_passenger',partner_bank(frame,groups,t,ids,1)[:,0])]:
        people,data,weight=pair_table(owner,rows,prior,ids,donors)
        train=np.isin(people,t)
        for variant in (['both','amount_only','first_only'] if key=='same_passenger' else ['both']):
            columns=list(data.columns)
            if variant!='both':
                columns.remove('SecondService')
            if variant=='first_only':
                columns.remove('SecondFee')
                columns.remove('SecondShare')
            x=data[columns]
            fit=fit_pair(x[train],y[people[train]],weight[train])
            p=fit.predict_proba(x[~train])[:,1]
            answers[key+'_'+variant]=np.bincount(people[~train],weights=weight[~train]*p,minlength=len(frame))[q]
        notes[key]=dict(training_pairs=int(train.sum()),scored_pairs=int((~train).sum()))
    return answers,notes


if __name__=='__main__':
    with threadpool_limits(2):
        evaluate('service_pairs',run_fold,REPORT)
