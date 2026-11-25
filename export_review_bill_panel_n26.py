"""Refit the five bill evidence columns and reproduce their original outer fold before release."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, data
from latent_bill_rounding_n26 import BILLS, OUT as BUDGET

DEST = ROOT/'artifacts/nov26_review_release'
NAMES = ['expected_bill','bill_template','rounding','charge_tree','charge_tabicl']


def splits():
    train,y,groups,fold=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=train.columns.tolist())
    both=pd.concat([train,test],ignore_index=True)
    for tag,raw,t,q in [('replay',train,np.flatnonzero(fold!=0),np.flatnonzero(fold==0)),
                        ('release',both,np.arange(len(y)),np.arange(len(y),len(both)))]:
        g=raw.PassengerId.str[:4].to_numpy()
        np.testing.assert_array_equal(g[:len(groups)],groups)
        assert not np.isin(g[t],g[q]).any()
        yield tag,raw,y,g,t,q


def receipt(raw,y,g,t,q):
    from catboost import CatBoostClassifier
    from latent_receipt_channel_n26 import fits,mixture_weights
    frame=compact_features(raw)
    meta=frame.drop(columns=BILLS)
    x=pd.get_dummies(meta,dtype=float).fillna(-1).to_numpy()
    channel=np.zeros(len(q))
    for a,b in GroupKFold(2,shuffle=True,random_state=4372).split(t,groups=g[t]):
        bank,cal=t[a],t[b]
        values,_=fits(frame,x,bank,np.r_[cal,q])
        term=[]
        for state in (False,True):
            weights=mixture_weights(values[:len(cal)][y[cal]==state])
            term.append(logsumexp(values[len(cal):]+np.log(weights),axis=1))
        channel+=(term[1]-term[0])/2
    prior=CatBoostClassifier(iterations=330,depth=5,verbose=False,thread_count=2,
                            random_seed=612,allow_writing_files=False)
    prior.fit(meta.iloc[t],y[t],cat_features=CATEGORICAL)
    return expit(logit(prior.predict_proba(meta.iloc[q])[:,1].clip(.002,.998))+channel)


def trees(raw,y,g,t,q):
    from catboost import CatBoostClassifier
    from conditional_bill_deviations_n26 import design,fit_fold
    from classifying_reconstructed_bills_n26 import load_views
    deviation=expit(fit_fold(*design(raw),y,g,t,q,4329)['deviation'])
    template=receipt(raw,y,g,t,q)
    budget=compact_features(raw)
    with np.load(BUDGET/'features.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(raw)],raw.PassengerId)
        budget['sum_exact']=z['total'][:len(raw)]
        features=z['floor'][:len(raw)]
    for j,col in enumerate(['absent_units','posterior_peak','local_log_density','budget_peak']):
        budget[col]=features[:,j]
    budget['budget_code']=features[:,4].astype(int).astype(str)
    clf=CatBoostClassifier(iterations=430,depth=6,learning_rate=.06,l2_leaf_reg=5,random_seed=4287,
                           thread_count=2,verbose=False,allow_writing_files=False)
    clf.fit(budget.iloc[t],y[t],cat_features=CATEGORICAL+['budget_code'])
    rounding=expit(clf.predict(budget.iloc[q],prediction_type='RawFormulaVal'))
    _,views,_,_=load_views(raw)
    clf=CatBoostClassifier(iterations=430,depth=6,learning_rate=.06,random_seed=9841,
                           thread_count=2,verbose=False,allow_writing_files=False)
    clf.fit(views['spectrum'].iloc[t],y[t],cat_features=CATEGORICAL)
    charge=clf.predict_proba(views['spectrum'].iloc[q])[:,1].clip(.002,.998)
    return np.column_stack([deviation,template,rounding,charge])


def neural(raw,y,g,t,q):
    import torch
    from tabicl_reconstructed_charge_n26 import build
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    _,extra,_,_=build(raw)
    p=infer(extra,y,t,q,891)
    # Archive converts clipped probabilities to float32 log-odds before float64 expit.
    return expit(logit(p.clip(.002,.998)).astype(float))[:,None]


def run(name):
    DEST.mkdir(exist_ok=True)
    for tag,raw,y,g,t,q in splits():
        p=(trees if name=='trees' else neural)(raw,y,g,t,q)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        file=DEST/f'bills_{name}_{tag}.npz'
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,p=p,t=t,q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str))
        file.with_suffix('.tmp').replace(file)
        print(name,tag,len(q),flush=True)


def bundle():
    from receipt_expert_transfer_n26 import read
    raw,y,g,f=data()
    _,odds=read('full',np.arange(len(y)),f)
    fold=expit(odds[f==0])
    evidence={}
    for tag,raw,y,g,t,q in splits():
        cols=[]
        for name in ('trees','neural'):
            with np.load(DEST/f'bills_{name}_{tag}.npz') as z:
                for key,val in [('t',t),('q',q),('passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],val)
                cols.append(z['p'])
        p=np.column_stack(cols)
        if tag=='replay':
            errors=np.max(abs(p-fold),axis=0)
            flips=((p>=.5)!=(fold>=.5)).sum(0)
            print(errors,flips,flush=True)
            assert np.all(errors<2e-5) and not flips.any()
            evidence.update(replay_fold=0,replay_rows=len(q),max_probability_errors=errors.tolist(),hard_label_differences=flips.tolist())
        else:
            path=DEST/'bills.npz'
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,names=NAMES,p=p,passenger=raw.PassengerId.iloc[q].to_numpy(str))
            path.with_suffix('.tmp').replace(path)
            evidence.update(training_rows=len(t),test_rows=len(q),names=NAMES,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/'reports/nov23_matched_learning_curve/review_release_bills.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['trees','neural','bundle'])
    action=parser.parse_args().action
    with threadpool_limits(2):
        bundle() if action=='bundle' else run(action)
