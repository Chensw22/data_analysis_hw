"""Replace cross-fitted deck densities with exact leave-one-booking-out kernels on the unlabelled roster."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from price_implied_deck_n26 import ROOT,DECKS,FEES,OUT as ORIGINAL,build_features,data

OUT=ROOT/'artifacts/nov27_leave_booking_out_fare'
REPORT=ROOT/'reports/nov23_matched_learning_curve/leave_booking_out_fare.json'


def roster():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    frame=pd.concat([raw,test],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    return frame,y,groups,f


def kernel(sigma):
    radius=int(4*sigma+.5)
    x=np.arange(-radius,radius+1,dtype=float)
    p=np.exp(-x*x/(2*sigma*sigma))
    return p/p.sum(),radius


def weight(delta,p,r):
    return np.where(abs(delta)<=r,p[(delta+r).clip(0,2*r)],0.)


def prepare():
    raw,y,group,f=roster()
    deck=pd.Index(DECKS).get_indexer(raw.Cabin.str.split('/').str[0])
    origin=raw.HomePlanet.fillna('?').to_numpy()
    fee=raw[FEES]
    qty=fee.gt(0).sum(axis=1).to_numpy()
    cost=2*fee.fillna(0).sum(axis=1).to_numpy(int)+qty
    valid=fee.notna().all(axis=1).to_numpy()&(qty>0)
    usable=valid&(deck>=0)
    length=cost.max()+136
    members={g:np.asarray(ix) for g,ix in raw.groupby(group,sort=False).indices.items()}
    spec=[kernel(s) for s in (2.4,42.)]
    posterior=np.full((len(raw),len(DECKS)),1/len(DECKS))
    for planet in np.unique(origin):
        same=usable&(origin==planet)
        asks=np.flatnonzero(valid&(origin==planet))
        backup=np.array([same.sum()-same[members[group[i]]].sum()<25 for i in asks],bool)
        for m,bank in [(~backup,same),(backup,usable)]:
            q=asks[m]
            if not len(q):continue
            after=np.empty((len(q),len(DECKS)))
            for d in range(len(DECKS)):
                donor=bank&(deck==d)
                h=np.bincount(cost[donor],minlength=length).astype(float)
                sharp=gaussian_filter1d(h,2.4,mode='constant')[cost[q]]
                wide=gaussian_filter1d(h,42.,mode='constant')[cost[q]]
                mass=.78*sharp+.22*(wide+.00002*(1+donor.sum()))
                # Every own-booking term in the kernel mass and class prior gets subtracted.
                queries=[];others=[]
                for j,i in enumerate(q):
                    b=members[group[i]]
                    src=b[donor[b]]
                    queries.extend([j]*len(src));others.extend(src)
                if queries:
                    qi=np.asarray(queries)
                    diff=cost[q[qi]]-cost[np.asarray(others)]
                    contribution=.78*weight(diff,*spec[0])+.22*(weight(diff,*spec[1])+.00002)
                    mass-=np.bincount(qi,weights=contribution,minlength=len(q))
                assert np.all(mass>0)
                after[:,d]=mass
            posterior[q]=after/after.sum(1,keepdims=True)
    rng=np.random.default_rng(10752)
    # Verify direct exclusion on both singleton and shared bookings, each represented origin, and both rosters.
    checked=set()
    size=raw.groupby(group).PassengerId.transform('size').to_numpy()
    for planet in np.unique(origin):
        for multi in [False,True]:
            for test in [False,True]:
                rows=np.flatnonzero(valid&(origin==planet)&((size>1)==multi)&((np.arange(len(raw))>=len(y))==test))
                if len(rows):checked.add(int(rng.choice(rows)))
    errors=[]
    for i in sorted(checked):
        survivors=usable&(group!=group[i])
        mask=survivors&(origin==origin[i])
        if mask.sum()<25:mask=survivors
        values=[]
        for d in range(len(DECKS)):
            selected=mask&(deck==d)
            h=np.bincount(cost[selected],minlength=length).astype(float)
            sharp=gaussian_filter1d(h,2.4,mode='constant')[cost[i]]
            wide=gaussian_filter1d(h,42,mode='constant')[cost[i]]+.00002*(1+selected.sum())
            values.append(.78*sharp+.22*wide)
        check=np.array(values)/sum(values)
        np.testing.assert_allclose(check,posterior[i],rtol=1e-10,atol=2e-12)
        errors.append(float(np.max(abs(check-posterior[i]))))
    with np.load(ORIGINAL/'deck_estimates.npz') as old:
        for k,a in [('passenger',raw.PassengerId),('label',deck),('eligible',valid),('cost',cost)]:
            np.testing.assert_array_equal(old[k],a)
        known=usable
        print('deck',(old['fine'][known].argmax(1)==deck[known]).sum(),(posterior[known].argmax(1)==deck[known]).sum(),known.sum(),flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'posterior.npz',ids=raw.PassengerId.to_numpy(str),p=posterior,checked=np.array(sorted(checked)),errors=errors,
                        eligible=valid,deck=deck)


def train():
    import torch
    from hashlib import sha256
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,f=roster()
    frame=build_features(raw,g)['fine']
    with np.load(OUT/'posterior.npz') as z:
        np.testing.assert_array_equal(raw.PassengerId,z['ids'])
        p,deck,m=z['p'],z['deck'],z['eligible']
    names=['fare_deck_'+d for d in DECKS]+['registered_probability','maximum_deck_probability']
    assert frame.columns[-len(names):].tolist()==names
    values=np.column_stack([p,np.where(deck>=0,p[np.arange(len(deck)),deck.clip(0)],np.nan),p.max(1)])
    values[~m]=np.nan
    frame[names]=values
    n=len(y)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(raw))
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'fold{k}.npz'
        if not path.exists():
            prob=infer(frame,y,t,q,5583+(k if k<8 else 0))
            temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,t=t,q=q,y=y[t],ids=raw.PassengerId.iloc[q].to_numpy(str),fine=prob,sha=DIGEST)
            temp.replace(path)
        with np.load(path) as z:
            for key,a in [('t',t),('q',q),('y',y[t]),('ids',raw.PassengerId.iloc[q])]:np.testing.assert_array_equal(z[key],a)
            assert str(z['sha'])==DIGEST and np.isfinite(z['fine']).all()
        print('fold',k,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','train'])
    if parser.parse_args().mode=='prepare':prepare()
    else:train()
