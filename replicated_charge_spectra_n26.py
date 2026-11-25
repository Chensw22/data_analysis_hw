"""Learn charge periodicities that distinguish decks in two disjoint unlabeled booking samples."""
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d,maximum_filter1d
from scipy.special import softmax
from sklearn.model_selection import GroupKFold

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES,DECKS
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker

OUT=ROOT/'artifacts/nov26_replicated_charge_spectra'


def deck_spectrum(x,deck,first,second,q):
    if not len(q):
        return np.empty((0,3,len(DECKS)))
    arrays=[]
    grid=1<<max(12,int(x.max()).bit_length())
    for subset in (first,second):
        hist=np.zeros((len(DECKS),grid))
        np.add.at(hist,(deck[subset],x[subset]),1)
        arrays.append(hist)
    h0,h1=arrays
    combined=h0+h1
    def centered(h):
        complex_sum=np.fft.rfft(h,axis=1)
        n=h.sum(1)
        prior=complex_sum.sum(0)/(1+n.sum())
        return (complex_sum-n[:,None]*prior)/(n[:,None]+24)
    a,b=centered(h0),centered(h1)
    c=centered(combined)
    mass=np.sqrt(h0.sum(1)*h1.sum(1))
    replication=(mass[:,None]*(a*b.conj()).real).sum(0)
    # Retain separated positive peaks with periods between six and 256 amount units.
    freq=np.arange(len(replication))
    keep=(freq>=grid//256)&(freq<=grid//6)&(replication>0)&(replication==maximum_filter1d(replication,max(3,grid//1024)))
    choices=np.flatnonzero(keep)
    basis=choices[np.argsort(-replication[choices],kind='stable')[:8]]
    coarse=gaussian_filter1d(combined,40,axis=1,mode='constant')+.0002*(1+combined.sum(1,keepdims=True))
    base=np.log(coarse[:,x[q]].T)
    def shift(amount):
        phase=np.exp(2j*np.pi*amount[:,None]*basis/grid)
        return 2*np.real(phase@c[:,basis].T)
    # Within-band displacement keeps the exact same broad density, spectral power and selected frequencies.
    moved=(x[q]//128)*128+(x[q]+43)%128
    return np.stack([softmax(base,axis=1),softmax(base+shift(x[q]),axis=1),
                     softmax(base+shift(moved),axis=1)],axis=1)


def prepare(people,g):
    fees=people[FEES]
    total=(2*fees.sum(axis=1,min_count=5)+fees.gt(0).sum(axis=1)).to_numpy()
    values=np.column_stack([fees,total])
    assert np.nanmin(values)>=0 and np.all(values[np.isfinite(values)]%1==0)
    x=np.nan_to_num(values,nan=0).astype(int)
    origins=people.HomePlanet.fillna('?').to_numpy()
    deck=pd.Index(DECKS).get_indexer(people.Cabin.str.split('/').str[0])
    valid=(deck>=0)
    feats=np.full((len(g),3,3*values.shape[1]),np.nan)
    diagnostic={}
    for k,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=7315).split(x,groups=g)):
        assert not np.isin(g[t],g[q]).any()
        rng=np.random.default_rng(6192+k)
        side=np.isin(g,rng.permutation(np.unique(g[t]))[::2])
        for j,col in enumerate(x.T):
            for origin in np.unique(origins[q]):
                train=t[(origins[t]==origin)&valid[t]&(col[t]>0)]
                if min(side[train].sum(),(~side[train]).sum())<30:
                    train=t[valid[t]&(col[t]>0)]
                ask=q[(origins[q]==origin)&(col[q]>0)]
                if not len(ask):continue
                scores=deck_spectrum(col,deck,train[side[train]],train[~side[train]],ask)
                known=deck[ask]>=0
                for h in range(3):
                    p=scores[:,h]
                    reg=p[np.arange(len(ask)),np.maximum(0,deck[ask])]
                    feats[ask,h,3*j:3*j+3]=np.column_stack([np.where(known,np.log(reg),np.nan),
                        p.max(1),-np.sum(p*np.log(p),axis=1)])
                    key=(j,h)
                    diagnostic.setdefault(key,[]).append(np.column_stack([p[known].argmax(1)==deck[ask][known],-np.log(reg[known])]))
    return feats,{f'{j}_{h}':dict(rows=len(v),deck_accuracy=float(v[:,0].mean()),deck_loss=float(v[:,1].mean()))
                  for (j,h),lst in diagnostic.items() if len(v:=np.concatenate(lst))}


def run():
    raw,y,g,fold,views,votes=projection(include_full=True)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people=pd.concat([raw,test],ignore_index=True)
    group=people.PassengerId.str[:4].to_numpy()
    dry=~raw[FEES].gt(0).any(axis=1).to_numpy()
    OUT.mkdir(exist_ok=True)
    features=OUT/'features.npz'
    if not features.exists():
        matrix,diag=prepare(people,group)
        with features.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle,ids=people.PassengerId.to_numpy(str),matrix=matrix,diag=json.dumps(diag))
        features.with_suffix('.tmp').replace(features)
    with np.load(features) as z:
        np.testing.assert_array_equal(z['ids'],people.PassengerId)
        features=z['matrix']
        diag=json.loads(str(z['diag']))
    names=['coarse','replicated','displaced']
    plain=compact_features(raw)
    plain['party_size']=raw.PassengerId.groupby(g).transform('size')
    scores=np.full((len(y),3),np.nan)
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'{k}.npz'
        if not path.exists():
            pred=[]
            for j in range(3):
                view=pd.concat([plain,pd.DataFrame(features[:len(y),j],columns=[f'deck_evidence_{n}' for n in range(features.shape[2])])],axis=1)
                pred.append(fit_model(view.iloc[t],y[t]).predict_proba(view.iloc[q])[:,1])
            with path.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle,t=t,q=q,y=y[t],p=np.column_stack(pred))
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for name,value in [('t',t),('q',q),('y',y[t])]:
                np.testing.assert_array_equal(z[name],value)
            scores[q]=z['p']
        print('fold',k,flush=True)
    report={'deck_diagnostic':diag}
    for tag in ['full','bookingmix','historical_to_new']:
        mask,parent=views[tag]
        wrong=parent!=y
        strict=(votes==parent[:,None]).all(1)
        domain=np.where(dry,(votes[:,:-2]==parent[:,None]).all(1),strict)
        flip=(scores>=.5)!=parent[:,None]
        actions={**dict(zip(names,flip.T,strict=True)), 'replicated_only':flip[:,1]&~flip[:,0],
                 'displaced_only':flip[:,2]&~flip[:,0], 'exclusive':flip[:,1]&~flip[:,0]&~flip[:,2]}
        subsets={'all':mask,'paid':mask&~dry,'original_common':mask&strict,'added_common':mask&domain&~strict}
        report[tag]={n:{k:ticker(m,wrong,action) for k,m in subsets.items()} for n,action in actions.items()}
        print(tag,[(n,v['all']['fixed'],v['all']['harmed']) for n,v in report[tag].items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/replicated_charge_spectra.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
