"""Order-held Nori ablation of own fare-derived deck evidence and leave-one-member-out evidence."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

from nori_booking_regression_n26 import CACHE, FEES, ROOT, SHA, WHL, data, design, packet_dir

ART = ROOT/'artifacts/nov26_nori_fare_party'
REPORT = ROOT/'reports/nov23_matched_learning_curve/nori_fare_party.json'
DECKS = list('ABCDEFGT')


def features(raw,g,t,q):
    fees=raw[FEES]
    positive=fees.gt(0).sum(axis=1).to_numpy()
    present=(positive>0)&fees.notna().all(axis=1).to_numpy()
    costs=2*fees.fillna(0).sum(axis=1).to_numpy(int)+positive
    size=int(costs.max()+136)
    decks=raw.Cabin.str.split('/').str[0].fillna('?').to_numpy(str)
    origin=raw.HomePlanet.fillna('?').to_numpy(str)
    posterior=np.full((len(raw),8),np.nan)
    def infer(bank,rows):
        assert not np.isin(g[bank],g[rows]).any()
        eligible=bank[present[bank]&(decks[bank]!='?')]
        for planet in np.unique(origin[rows]):
            ask=rows[(origin[rows]==planet)&present[rows]]
            if not len(ask):continue
            ref=eligible[origin[eligible]==planet]
            if len(ref)<25:ref=eligible
            density=[]
            for deck in DECKS:
                parts=ref[decks[ref]==deck]
                h=np.bincount(costs[parts],minlength=size).astype(float)
                broad=gaussian_filter1d(h,42,mode='constant')+.00002*(1+len(parts))
                fine=gaussian_filter1d(h,2.4,mode='constant')
                density.append((.78*fine+.22*broad)[costs[ask]])
            x=np.column_stack(density)
            posterior[ask]=x/x.sum(1,keepdims=True)
    for a,b in GroupKFold(5,shuffle=True,random_state=9664).split(t,groups=g[t]):
        infer(t[a],t[b])
    infer(t,q)
    np.testing.assert_allclose(posterior[present].sum(1),1)
    lookup=pd.Index(DECKS).get_indexer(decks)
    registered=np.where(lookup>=0,posterior[np.arange(len(raw)),lookup.clip(0)],np.nan)
    own=np.column_stack([posterior,registered])
    counts=pd.Series(present.astype(int)).groupby(g).transform('sum').to_numpy()-present
    totals=pd.DataFrame(np.nan_to_num(posterior)).groupby(g).transform('sum').to_numpy()-np.nan_to_num(posterior)
    peers=np.full_like(posterior,np.nan)
    np.divide(totals,counts[:,None],out=peers,where=counts[:,None]>0)
    np.testing.assert_allclose(peers[counts>0].sum(1),1)
    peer=np.column_stack([counts,peers])
    return own,peer,counts,present


def predictor():
    os.environ['MPLCONFIGDIR']=str(CACHE/'mpl')
    os.environ['HF_HUB_OFFLINE']='1'
    sys.path.insert(0,str(CACHE/'lib'))
    import torch
    from synthefy_nori import NoriRegressor
    torch.set_num_threads(3)
    assert hashlib.sha256((CACHE/'nori.pt').read_bytes()).hexdigest()==SHA
    wheel=CACHE/'synthefy_nori-0.24.0-py3-none-any.whl'
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==WHL
    return torch,NoriRegressor


def fit(raw,y,g,t,q,mode,seed,resources):
    torch,regressor=resources
    own,peer,counts,present=features(raw,g,t,q)
    base=design(raw,g,t)
    x=np.column_stack([base,own] if mode=='own' else [base,own,peer]).astype(np.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    reg=regressor(model_path=str(CACHE/'nori.pt'),model='nori-6m',device='cpu',inference_config=str(CACHE/'one_member.json'),
                  categorical_columns=[],augmentations=())
    reg.fit(x[t],y[t].astype(float))
    p=np.asarray(reg.predict(x[q]),float).ravel()
    assert np.isfinite(p).all() and len(p)==len(q)
    return dict(t=t,q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str),labels=y[t],p=p,peer_count=counts[q],
                own_evidence=present[q],sha=SHA)


def run(mode,folds):
    ART.mkdir(exist_ok=True)
    raw,y,g,f=data()
    resources=predictor()
    for k in folds:
        start=time.monotonic()
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        path=ART/f'{mode}{k}.npz'
        if path.exists():continue
        out=fit(raw,y,g,t,q,mode,6117+k,resources)
        np.savez_compressed(path,**out,seconds=time.monotonic()-start)
        print(mode,k,float(np.mean((out['p']>=.5)==y[q])),int(time.monotonic()-start),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['own','peer'])
    ap.add_argument('--folds',nargs='+',type=int,default=list(range(8)))
    args=ap.parse_args()
    run(args.mode,args.folds)
