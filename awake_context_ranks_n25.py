"""Fit TabICL to awake passengers with leave-booking-out service percentiles."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data

BILLS = ('RoomService','FoodCourt','ShoppingMall','Spa','VRDeck')
OUT = ROOT/'artifacts/nov25_awake_context_ranks'
MODES = ('personal','conditional_ranks','shuffled_ranks')


def reference(values, cells, bookings):
    # Empirical CDF mid-ranks and positive incidence among passengers outside the whole query order.
    table = pd.DataFrame(dict(v=values, cell=cells, booking=bookings))
    totals = np.zeros((len(table),3))
    for keys, sign in [(['cell'],1),(['cell','booking'],-1)]:
        for idx in table.groupby(keys,sort=False).indices.values():
            a = np.sort(values[idx][np.isfinite(values[idx])])
            if not len(a):
                continue
            v = values[idx]
            midpoint = .5*(np.searchsorted(a,v,side='left')+np.searchsorted(a,v,side='right'))
            totals[idx] += sign*np.column_stack([np.full(len(idx),len(a)),midpoint,np.full(len(idx),(a>0).sum())])
    return totals


def contextual(raw):
    fees = raw[list(BILLS)].to_numpy(float)
    group = raw.PassengerId.str.split('_').str[0].to_numpy()
    planet = raw.HomePlanet.fillna('?').astype(str).to_numpy()
    awake = raw.CryoSleep.astype('string').fillna('?').to_numpy(str)
    ages = np.digitize(raw.Age.fillna(-1), [0,13,21,40])
    result = {}
    for j,name in enumerate(BILLS):
        rest = np.delete(fees,j,axis=1)
        cost = np.digitize(np.nansum(rest,axis=1),[1,300,900,2000,5000])
        active = (rest>0).sum(1).clip(0,2)
        pooled = np.full((len(raw),2),.5)
        cells = pd.DataFrame(dict(awake=awake,planet=planet,cost=cost,active=active,age=ages))
        for keys in [('awake',),('awake','planet'),('awake','planet','cost'),tuple(cells)]:
            ids = cells.groupby(list(keys),sort=False).ngroup().to_numpy()
            observations = reference(fees[:,j],ids,group)
            pooled = (observations[:,1:]+20*pooled)/(observations[:,0,None]+20)
        result[name+'_rank']=np.where(np.isfinite(fees[:,j]),pooled[:,0],np.nan)
        result[name+'_positive_rate']=pooled[:,1]
    features = pd.DataFrame(result)
    perm = np.arange(len(raw))
    rng = np.random.default_rng(713)
    bands = pd.DataFrame(dict(awake=awake,planet=planet,n=(fees>0).sum(1)))
    for idx in bands.groupby(list(bands),sort=False).indices.values():
        perm[idx] = rng.permutation(idx)
    personal = compact_features(raw)
    return (personal,pd.concat([personal,features],axis=1),
            pd.concat([personal,features.iloc[perm].reset_index(drop=True)],axis=1))


def run(selected):
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,booking,fold=data()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    roster = pd.concat([raw,test],ignore_index=True)
    views = contextual(roster)
    awake = raw.CryoSleep.eq(False).to_numpy()
    OUT.mkdir(exist_ok=True)
    for k in selected:
        t,q=np.flatnonzero((fold!=k)&awake),np.flatnonzero((fold==k)&awake)
        assert not np.isin(booking[t],booking[q]).any()
        for name,x in zip(MODES,views,strict=True):
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                p=infer(x,y,t,q,891+k)
                assert len(p)==len(q) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                                        checkpoint=DIGEST,fields=np.asarray(x.columns,dtype=str),p=p)
                path.with_suffix('.tmp').replace(path)
            print(k,name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',type=int,choices=range(8),default=list(range(8)))
    run(parser.parse_args().folds)
