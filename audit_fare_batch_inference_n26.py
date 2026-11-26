"""Replay frozen fare test models and change only their unlabeled query companions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from full_roster_booking_validation_n23 import ROOT, data
from nori_booking_regression_n26 import CACHE, design, packet_dir
from nori_fare_party_n26 import ART as N, SHA, features, predictor
from price_implied_deck_n26 import OUT as F, build_features

OUT=ROOT/'artifacts/nov26_fare_inference_audit'
ART=ROOT/'artifacts'
FARE=ART/'proposal_n26_fare_deck/scores.npz'
FIRST=ART/'proposal_n26_champion_fare_six'
SECOND=ART/'proposal_n26_joint_fare_foundations'
REPORT=ROOT/'reports/nov23_matched_learning_curve/fare_inference_audit.json'


def roster():
    a,y,g,f=data()
    b=pd.read_csv(ROOT/'data/raw/test.csv',usecols=a.columns.tolist())
    allrows=pd.concat([a,b],ignore_index=True)
    orders=allrows.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(g,orders[len(y):]).any()
    np.testing.assert_array_equal(pd.Series(orders).groupby(orders).transform('size')[:len(a)],
                                  pd.Series(g).groupby(g).transform('size'))
    return a,allrows,y,orders,f


def compare(a,b):
    d=np.abs(a-b)
    return dict(rows=len(d),mean_absolute=float(d.mean()),q95_absolute=float(np.quantile(d,.95)),
                max_absolute=float(d.max()),sign_changes=int(((a>=.5)!=(b>=.5)).sum()))


def infer(family,variant,mode):
    a,frame,y,g,f=roster()
    n=len(y)
    ti=np.arange(n,len(frame));test=mode=='test';support=mode.startswith('support')
    k=int(mode[-1]) if support else 0
    t=np.arange(n) if test else np.flatnonzero(f!=k)
    q=ti if test else np.flatnonzero(f==0)
    # Fixed selection by booking; all train indices/features stay unchanged in a batch comparison.
    sub=next(GroupKFold(4,shuffle=True,random_state=525625).split(ti,groups=g[ti]))[1]
    part=ti[sub] if test or support else np.r_[q,ti[:len(ti)-len(q)]]
    if family=='tabicl':
        import torch
        from tabicl import TabICLClassifier
        from tabicl_routed_context_n23 import DIGEST,WEIGHT
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
        torch.set_num_threads(2)
        x=build_features(frame,g)[variant]
        pd.testing.assert_frame_equal(build_features(a,g[:n])[variant],x.iloc[:n])
        model=TabICLClassifier(n_estimators=2,model_path=WEIGHT,allow_auto_download=False,
            device='cpu',batch_size=1,n_jobs=2,random_state=5583,kv_cache=False,use_amp=False,use_fa3=False)
        model.fit(x.iloc[t],y[t])
        np.testing.assert_array_equal(model.classes_,[False,True])
        def pred(rows):return model.predict_proba(x.iloc[rows])[:,1]
        file=FARE if test or support else F/'neural_full_0.npz'
        with np.load(file) as z: expected=z[variant]
    else:
        torch,regressor=predictor()
        base=design(frame,g,t)
        np.testing.assert_array_equal(design(a,g[:n],t),base[:n])
        if variant=='fine':
            # local features() expects t + q to cover the entire input. Use the
            # original fold roster for training/query, then reconstruct foreign
            # rows from the same training pool for the expanded query.
            if not test:
                full_q=np.r_[np.flatnonzero(f==k),ti]
            else:
                full_q=q
            own=features(frame,g,t,full_q)[0]
            if not test:
                np.testing.assert_array_equal(own[:n],features(a,g[:n],t,np.flatnonzero(f==k))[0])
            base=np.column_stack([base,own]).astype(np.float32)
        torch.manual_seed(6117);np.random.seed(6117)
        model=regressor(model_path=str(CACHE/'nori.pt'),model='nori-6m',device='cpu',
            inference_config=str(CACHE/'one_member.json'),categorical_columns=[],augmentations=())
        model.fit(base[t],y[t].astype(float))
        def pred(rows):return np.asarray(model.predict(base[rows]),float).ravel()
        file=(N/('own_test.npz' if variant=='fine' else 'raw_test.npz')) if test or support else (
            N/'own0.npz' if variant=='fine' else packet_dir(0)/'fold0.npz')
        with np.load(file) as z: expected=z['p']
    start=monotonic()
    if support:
        # Both score vectors are for exactly the same test rows. Test-batch dependence
        # is separately controlled by the full-support split-query experiment.
        full=expected;rows=part;paired=expected[sub];changed=pred(part)
        output=OUT/f'{family}_{variant}_{mode}.npz'
        np.savez_compressed(output,rows=rows,full=full,reference=expected,paired=paired,changed=changed,
                            full_ids=frame.PassengerId.iloc[ti].to_numpy(str),elapsed=monotonic()-start)
        print(family,variant,mode,'training support comparison',compare(paired,changed),flush=True)
        return
    full=pred(q)
    np.testing.assert_allclose(full,expected,rtol=0,atol=3e-5)
    np.testing.assert_array_equal(full>=.5,expected>=.5)
    print(family,variant,mode,'full replay',compare(full,expected),flush=True)
    reduced=pred(part)
    if test:
        rows=part; paired=full[sub]; changed=reduced
    else:
        rows=q; paired=full; changed=reduced[:len(q)]
    output=OUT/f'{family}_{variant}_{mode}.npz'
    np.savez_compressed(output,rows=rows,full=full,reference=expected,paired=paired,changed=changed,
                        full_ids=frame.PassengerId.iloc[q].to_numpy(str),elapsed=monotonic()-start)
    print(family,variant,mode,'batch comparison',compare(paired,changed),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('family',choices=['nori','tabicl']);p.add_argument('variant',choices=['raw','fine'])
    p.add_argument('mode',choices=['test','fold0','support0','support1']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    infer(**vars(args))
