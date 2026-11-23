"""Estimate the observable zero-versus-paid contrast using out-of-booking nuisance predictions."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_service_pairs_n23 import REPORT
from exaone_validation_data import CATEGORICAL


def describe(row,groups):
    full,g,size,_=load_roster()
    np.testing.assert_array_equal(groups,g[row])
    fees=full[BILLS].fillna(0)
    total=fees.sum(axis=1)
    positive=total.gt(0)
    peers=total.groupby(g).transform('sum')-total
    npositive=positive.groupby(g).transform('sum')-positive.astype(int)
    x=full.drop(columns=BILLS).copy()
    x['BookingSize']=size
    x['PeerSpend']=np.log1p(peers)
    x['PayingPeers']=npositive
    x['MissingFees']=full[BILLS].isna().sum(axis=1)
    return x.iloc[row].reset_index(drop=True),~positive.to_numpy()[row]


def model(x,y,t,q):
    fit=CatBoostClassifier(iterations=220,depth=4,learning_rate=.065,l2_leaf_reg=6,
                           thread_count=2,random_seed=612,verbose=False,allow_writing_files=False,one_hot_max_size=32)
    fit.fit(x.iloc[t],y[t],cat_features=CATEGORICAL)
    return fit.predict_proba(x.iloc[q])[:,1]


def contrast(x,g,y,zero,t,q):
    m,e=np.empty(len(t)),np.empty(len(t))
    for a,b in GroupKFold(3,shuffle=True,random_state=827).split(t,groups=g[t]):
        assert not np.intersect1d(g[t[a]],g[t[b]]).size
        m[b]=model(x,y,t[a],t[b])
        e[b]=model(x,zero,t[a],t[b])
    mtest=model(x,y,t,q)
    etest=model(x,zero,t,q)
    residual=zero[t].astype(float)-e.clip(.01,.99)
    delta_y=y[t].astype(float)-m
    params=dict(iterations=220,depth=4,learning_rate=.065,l2_leaf_reg=6,thread_count=2,random_seed=612,
                verbose=False,allow_writing_files=False,one_hot_max_size=32)
    reg=CatBoostRegressor(**params)
    weight=residual**2
    reg.fit(x.iloc[t],delta_y/residual,sample_weight=weight/weight.mean(),cat_features=CATEGORICAL)
    tau=reg.predict(x.iloc[q]).clip(-1,1)
    average=float(residual@delta_y/(residual@residual))
    indicator=x.assign(NoPositive=zero.astype(int))
    return dict(pooled=mtest,with_indicator=model(indicator,y,t,q),zero_only=model(x,y,t[zero[t]],q),
                constant=(mtest+(1-etest)*average).clip(0,1),
                conditional=(mtest+(1-etest)*tau).clip(0,1))


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    frame,zero=describe(row,g)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        votes={'reference':ref}
        for f in np.unique(fold):
            complete=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[complete]))
            q=complete[zero[complete]&(ref[complete]==(base[complete]>=.5))]
            assert not np.intersect1d(g[t],g[q]).size
            for k,p in contrast(frame,g,y,zero,t,q).items():
                votes.setdefault(k,ref.copy())[q]=p>=.5
                votes.setdefault('quarter_'+k,ref.copy())[q]=.75*base[q]+.25*p>=.5
        hit={k:v==y for k,v in votes.items()}
        plans[seed]={k:paired(v,ref==y) for k,v in hit.items()}
        if seed!='v131':
            for k,v in hit.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:m['fixed']-m['harmed'] for k,m in plans[seed].items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'zero_contrast.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
