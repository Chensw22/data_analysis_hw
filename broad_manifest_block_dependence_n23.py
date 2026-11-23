"""Estimate class-specific information in the alignment of registration, route, and billing blocks."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_service_bag_trees_n23 import fit
from broad_unexpected_bill_n23 import REPORT
from exaone_validation_data import CATEGORICAL


def donors(train,groups,y,rng):
    # Permuting within each outcome preserves each block's empirical conditional
    # distribution exactly. Reject same-booking links, including the source row.
    result=np.empty(len(train),int)
    for label in (False,True):
        src=np.flatnonzero(y[train]==label)
        for _ in range(200):
            perm=rng.permutation(src)
            if not np.any(groups[train[perm]]==groups[train[src]]):
                break
        else:
            order=src[np.argsort(groups[train[src]],kind='stable')]
            longest=np.unique(groups[train[order]],return_counts=True)[1].max()
            assert longest<len(src)//2
            perm=np.empty_like(src)
            perm[np.searchsorted(src,order)]=np.roll(order,longest)
        result[src]=train[perm]
    np.testing.assert_array_equal(np.sort(result),np.sort(train))
    assert np.all(y[result]==y[train]) and np.all(groups[train]!=groups[result])
    return result


def run_fold(frame,groups,y,t,q):
    assert not np.intersect1d(groups[t],groups[q]).size
    rng=np.random.default_rng(702)
    route=['HomePlanet','Destination','Deck','Side','CabinNumber','BookingSize']
    personal=['Age','CryoSleep','VIP']
    assert set(route+personal+BILLS)==set(frame)
    # Retain the route; replace personal records and the complete five-service vector separately.
    people=donors(t,groups,y,rng)
    accounts=donors(t,groups,y,rng)
    synthetic=frame.iloc[t].copy()
    synthetic[personal]=frame.iloc[people][personal].to_numpy()
    synthetic[BILLS]=frame.iloc[accounts][BILLS].to_numpy()
    x=pd.concat([frame.iloc[t],synthetic],ignore_index=True)
    outcome=np.tile(y[t].astype(int),2)
    genuine=np.repeat([1,0],len(t))
    shuffle=rng.integers(0,2,len(t))
    target=frame.iloc[q]
    result={'direct':fit(frame.iloc[t],y[t],np.ones(len(t))).predict_proba(target)[:,1]}
    for name,tags in [('authentic',genuine),('sham',genuine^np.tile(shuffle,2))]:
        model=CatBoostClassifier(iterations=225,depth=5,learning_rate=.09,l2_leaf_reg=5,one_hot_max_size=32,
                                 thread_count=2,random_seed=731,verbose=False,allow_writing_files=False)
        model.fit(x,2*outcome+tags,cat_features=CATEGORICAL)
        np.testing.assert_array_equal(model.classes_,[0,1,2,3])
        joint=model.predict_proba(target).reshape(len(q),2,2)
        real=joint[:,1,1]/joint[:,:,1].sum(1)
        synthetic_score=joint[:,1,0]/joint[:,:,0].sum(1)
        result[name+'_real']=real
        result[name+'_synthetic']=synthetic_score
        result[name+'_marginal']=joint[:,1].sum(1)
        result[name+'_correction']=expit(logit(result['direct'].clip(.002,.998))
                                         +logit(real.clip(.002,.998))-logit(synthetic_score.clip(.002,.998)))
    return result,dict(train=len(t),query=len(q),same_donor_booking=int((groups[people]==groups[accounts]).sum()))


if __name__=='__main__':
    with threadpool_limits(2):
        evaluate('manifest_block_dependence',run_fold,REPORT)
