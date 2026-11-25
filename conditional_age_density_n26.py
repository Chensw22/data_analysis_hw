"""Learn the distribution of recorded age under each outcome and evaluate within-band evidence."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, data

OUT = ROOT/'artifacts/nov26_conditional_age_density'
FEES = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
AGES = np.arange(100)
BAND = np.digitize(AGES, [13,18,25,40,60])


def density_estimates(matrix, age, t, q, labels, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.tensor(matrix,dtype=torch.float32)
    # Smooth neighboring years while retaining the abrupt transitions into adolescence and adulthood.
    basis = np.column_stack([np.exp(-.5*((AGES-c)/w)**2)
        for w, centers in [(4,range(0,61,4)),(10,range(10,100,10))] for c in centers]
        + [AGES==0,AGES<13,AGES<18,AGES/100]).astype(np.float32)
    b = torch.tensor(basis.T)
    net = nn.Sequential(nn.Linear(matrix.shape[1]+1,64),nn.SiLU(),nn.Linear(64,len(b)))
    year_bias = nn.Parameter(torch.zeros(len(AGES)))
    opt = torch.optim.AdamW([*net.parameters(),year_bias],lr=.003,weight_decay=.025)
    tt = t[age[t]>=0]
    target = torch.tensor(age[tt],dtype=torch.long)
    encoded = torch.tensor(labels[age[t]>=0],dtype=torch.float32)
    def logits(rows, outcome):
        return net(torch.cat([x[rows],outcome[:,None]],dim=1))@b + year_bias
    snapshots = []
    for epoch in range(22):
        for batch in np.array_split(rng.permutation(len(tt)),int(np.ceil(len(tt)/256))):
            out = logits(tt[batch],encoded[batch])
            loss = nn.functional.cross_entropy(out,target[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (13,21):
            with torch.no_grad():
                snapshots.append(np.stack([logits(q,torch.full((len(q),),float(v))).softmax(1).numpy()
                                           for v in (0,1)],axis=1))
    return np.mean(snapshots,axis=0)


def train():
    import torch
    from catboost import CatBoostClassifier
    torch.set_num_threads(2)
    raw,y,g,f=data()
    frame=compact_features(raw)
    size=raw.PassengerId.groupby(g).transform('size').to_numpy()
    frame['party_size']=size
    recorded=frame.Age.notna().to_numpy()
    age=frame.Age.fillna(-1).to_numpy(int)
    assert np.array_equal(age[recorded],frame.Age[recorded]) and age.max()<len(AGES)
    # -1 is a separate missing-value code, and never enters the year likelihood.
    band=np.where(recorded,BAND[age.clip(0)],-1)
    masked=frame.drop(columns='Age')
    banded=masked.assign(age_band=band)
    x=masked.copy()
    for fee in FEES:
        x[fee+'_missing']=x[fee].isna()
        x[fee+'_used']=x[fee]>0
        x[fee]=np.log1p(x[fee])/6
    x['CabinNumber']=x.CabinNumber/1000
    x['party_size']=np.log1p(size)
    floats=pd.get_dummies(x,columns=CATEGORICAL).fillna(-1).to_numpy(np.float32)
    strata=(frame.HomePlanet+'/'+frame.CryoSleep+'/'+(size==1).astype(str)+'/'+raw[FEES].gt(0).any(axis=1).astype(str)).to_numpy()
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    OUT.mkdir(exist_ok=True)
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            q=ids[fold==k]
            other=ids[fold!=k]
            t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[t],g[q]).any()
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                seed=4270+int(k)
                def direct(inputs):
                    clf=CatBoostClassifier(iterations=380,depth=5,learning_rate=.055,l2_leaf_reg=6,
                        random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False,cat_features=CATEGORICAL)
                    return clf.fit(inputs.iloc[t],y[t]).predict_proba(inputs.iloc[q])[:,1]
                scores=dict(prior=direct(banded),direct=direct(frame))
                rng=np.random.default_rng(seed)
                fake=y[t].copy()
                for level in np.unique(strata[t]):
                    ix=np.flatnonzero((strata[t]==level)&recorded[t])
                    fake[ix]=rng.permutation(fake[ix])
                for name,labels in [('true',y[t]),('shuffled',fake)]:
                    pmf=density_estimates(floats,age,t,q,labels,seed)
                    np.testing.assert_allclose(pmf.sum(2),1,atol=3e-6)
                    safe_year=age[q].clip(0)
                    mass=(pmf*(BAND==BAND[safe_year,None])[:,None,:]).sum(2)
                    exact=pmf[np.arange(len(q)),:,safe_year]
                    scores[name+'_year']=exact/mass.clip(1e-10)
                    assert np.isfinite(scores[name+'_year']).all()
                with path.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,truth=y[t],age=age[q],**scores)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,val in [('t',t),('q',q),('truth',y[t]),('age',age[q])]:
                    np.testing.assert_array_equal(z[name],val)
            print(tag,k,flush=True)


def score():
    from disjoint_tree_fields_n25 import DEST, stats
    from sklearn.metrics import log_loss
    from solo_selection_likelihood_n26 import regimes
    raw,y,g,views=regimes()
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    report={}
    for tag,ids,fold,baseline,view in views:
        fresh=~np.isin(g[ids],g[views[0][1]])
        prob={}
        for k in np.unique(fold):
            loc=fold==k
            if tag=='historical':
                fresh[loc]=~np.isin(g[ids[loc]],g[ids[~loc]])
            with np.load(OUT/f'{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],ids[loc])
                for key in ('prior','direct','true_year','shuffled_year'):
                    shape=z[key].shape
                    prob.setdefault(key,np.empty((len(ids),*shape[1:])))[loc]=z[key]
        valid=raw.Age.iloc[ids].notna().to_numpy()
        alone=solo[ids]
        truth=y[ids]
        anchor=view['anchor']
        known=view['known']
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        masks={'all':known,'solo':known&alone,'common':common,'solo_common':common&alone,
               'fresh':known&fresh,'fresh_solo':known&fresh&alone}
        metrics={}
        actions={}
        for key in ('true','shuffled'):
            density=prob[key+'_year'].clip(1e-9,1)
            step=np.log(density[:,1]/density[:,0])
            updated=logit(prob['prior'])+step
            metrics[key+'_within_band_age_nll']=-np.log(density[np.where(valid)[0],truth[valid].astype(int)]).mean()
            metrics[key+'_transport_nll']=log_loss(truth[valid],expit(updated[valid]))
            changed=(updated>=0)!=anchor
            actions[key+'_conditional']=changed
            actions[key+'_induced_vs_direct']=changed&((prob['direct']>=.5)==anchor)
            effective=updated-logit(prob['direct'])
            actions[key+'_half_increment']=((logit(baseline.clip(.002,.998))+.5*effective>=0)!=anchor)&((baseline>=.5)==anchor)
        metrics['band_transport_nll']=log_loss(truth[valid],prob['prior'][valid])
        metrics['direct_transport_nll']=log_loss(truth[valid],prob['direct'][valid])
        table={name:{c:stats(valid*act*np.where(anchor!=truth,1,-1),mask,fold,g[ids])
                     for c,mask in masks.items()} for name,act in actions.items()}
        report[tag]=dict(losses=metrics,known_ages=int(valid.sum()),actions=table)
        print(tag,metrics,[(n,v['all']['fixed'],v['all']['harmed']) for n,v in table.items()],flush=True)
    DEST.with_name('conditional_age_density.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','score'])
    (train if parser.parse_args().action=='train' else score)()
