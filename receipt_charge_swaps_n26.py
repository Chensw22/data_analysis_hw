"""Test whether a real service charge preserves the exact spending pattern better than a matched replacement."""
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

from fare_route_inference_n26 import DECK, FEES, ROOT, build_features, data
from solo_selection_likelihood_n26 import fit_model

OUT = ROOT/'artifacts/nov26_disjoint_charge_swaps'


def rank(observed, alternatives):
    return np.mean((observed > alternatives) + .5 * (observed == alternatives))


def prepare():
    raw, y, _, _ = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns.tolist())
    frame = pd.concat([raw,test], ignore_index=True)
    groups = frame.PassengerId.str[:4].to_numpy()
    planet = frame.HomePlanet.fillna('?').to_numpy()
    deck = frame.Cabin.str.split('/').str[0].fillna('?').to_numpy()
    charges = frame[FEES].to_numpy(float)
    total = np.nansum(charges,axis=1).astype(int)
    count = (charges > 0).sum(1)
    eligible = (count > 0) & np.isfinite(charges).all(1)
    price = 2 * total + count
    output = {name: np.full((len(frame),len(FEES)),np.nan) for name in ['exact','broad','displaced']}
    fold = np.zeros(len(frame),int)
    for k,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=9664).split(total,groups=groups)):
        assert not np.isin(groups[t],groups[q]).any()
        fold[q]=k
        support=t[eligible[t]]
        order_ids=np.unique(groups[support])
        rng=np.random.default_rng(4391+k)
        density_orders=rng.permutation(order_ids)[::2]
        density_source=np.isin(groups,density_orders)
        for p,d in sorted(set(zip(planet[q],deck[q]))):
            bank=support[(planet[support]==p)&(deck[support]==d)]
            targets=q[eligible[q]&(planet[q]==p)&(deck[q]==d)]
            density=bank[density_source[bank]]
            suppliers=bank[~density_source[bank]]
            assert not np.isin(groups[density],groups[suppliers]).any()
            if min(len(density),len(suppliers))<12 or not len(targets):
                continue
            upper=int(4*total.max()+300)
            hist=np.bincount(price[density],minlength=upper).astype(float)
            wide=gaussian_filter1d(hist,42,mode='constant') + .00005
            fine=.78*gaussian_filter1d(hist,2.4,mode='constant')+.22*wide
            def lookup(sums,active):
                values=2*sums+active
                shifted=128*(sums//64)+active+2*((sums+23)%64)
                return {'broad':wide[values], 'exact':fine[values],
                        'displaced':wide[values]*fine[shifted]/wide[shifted]}
            for j in range(len(FEES)):
                # All replacement donors and the price spectrum exclude the target booking.
                donors=suppliers[charges[suppliers,j]>0]
                donors=donors[np.argsort(charges[donors,j],kind='stable')]
                values=charges[donors,j]
                for i in targets[charges[targets,j]>0]:
                    actual=charges[i,j]
                    radius=max(32.,.15*actual)
                    l,r=np.searchsorted(values,[actual-radius,actual+radius],side='left')
                    candidates=donors[l:r]
                    candidates=candidates[charges[candidates,j]!=actual]
                    if len(candidates)<3:
                        continue
                    _,unique=np.unique(groups[candidates],return_index=True)
                    candidates=candidates[np.sort(unique)]
                    if len(candidates)<3:
                        continue
                    if len(candidates)>24:
                        candidates=candidates[np.linspace(0,len(candidates)-1,24).astype(int)]
                    amounts=np.r_[actual,charges[candidates,j]].astype(int)
                    scores=lookup(total[i]-int(actual)+amounts,count[i])
                    for name,likelihood in scores.items():
                        output[name][i,j]=rank(likelihood[0],likelihood[1:])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',passenger=frame.PassengerId.to_numpy(str),**output)
    available=np.isfinite(output['exact'])
    measures={}
    for name,v in output.items():
        measures[name]=dict(passengers=int(available.any(1).sum()),service_comparisons=int(available.sum()),
                            mean_rank=float(np.nanmean(v)),per_service=np.nanmean(v,axis=0).tolist(),
                            fold_mean_rank=[float(np.nanmean(v[fold==k])) for k in range(5)])
    (OUT/'diagnostic.json').write_text(json.dumps(measures,indent=2)+'\n')
    print(measures,flush=True)


def train():
    raw,y,g,_=data()
    x=build_features(raw,g)['fine']
    with np.load(OUT/'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(y)],raw.PassengerId)
        variations={key:pd.concat([x,pd.DataFrame(z[key][:len(y)],columns=['swapped_'+f for f in FEES])],axis=1)
                    for key in ('exact','broad','displaced')}
    for path in sorted(DECK.glob('tree_*.npz')):
        with np.load(path) as z:
            t,q=z['t'],z['q']
            np.testing.assert_array_equal(z['y'],y[t])
        assert not np.isin(g[t],g[q]).any()
        pred={name:fit_model(frame.iloc[t],y[t]).predict_proba(frame.iloc[q])[:,1]
              for name,frame in variations.items()}
        np.savez_compressed(OUT/path.name,t=t,q=q,**pred)


if __name__=='__main__':
    prepare()
    train()
