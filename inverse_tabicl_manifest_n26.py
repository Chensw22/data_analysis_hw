"""Score observed passenger attributes under both transport hypotheses using grouped in-context classifiers."""
import argparse
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit,logit

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data
from tabicl_routed_context_n23 import BILLS,DIGEST,WEIGHT,infer

OUT=ROOT/'artifacts/nov26_inverse_tabicl_manifest'
VIEWS=('dominant_service','cabin_side','home_planet')


def masked_view(x,name):
    if name!='dominant_service':
        column={'cabin_side':'Side','home_planet':'HomePlanet'}[name]
        return x.drop(columns=column),x[column].astype(str).to_numpy()
    amounts=x[BILLS].to_numpy(float)
    filled=np.nan_to_num(amounts,nan=-1.)
    sorting=np.sort(filled,axis=1)
    largest=sorting[:,-1]
    # Match an observed service to the full sorted basket. All tied positive maxima
    # share a separate symbol so the order of the original columns cannot break a tie.
    label=np.asarray(BILLS)[filled.argmax(1)].astype(object)
    label[largest<=0]='no_positive_charge'
    label[(largest>0)&(sorting[:,-2]==largest)]='tied_services'
    masked=x.drop(columns=BILLS).copy()
    for j,amount in enumerate(sorting.T):
        masked[f'ordered_bill_{j}']=amount
    return masked,label.astype(str)


def conditional_observation(x,attribute,y,t,q,seed,placebo):
    from tabicl import TabICLClassifier
    levels=np.unique(attribute[t])
    index=pd.Index(levels).get_indexer(attribute[q])
    labels=pd.Index(levels).get_indexer(attribute[t])
    train=x.iloc[t].copy()
    hypothesis=y[t].astype(int)
    if placebo:
        hypothesis=np.random.default_rng(seed+1101).permutation(hypothesis)
    train['hypothetical_transport']=hypothesis
    model=TabICLClassifier(n_estimators=2,model_path=WEIGHT,allow_auto_download=False,device='cpu',
              batch_size=1,n_jobs=2,random_state=seed,kv_cache=False,use_amp=False,use_fa3=False)
    model.fit(train,labels)
    np.testing.assert_array_equal(model.classes_,np.arange(len(levels)))
    query=x.iloc[q]
    probabilities=[]
    for side in (0,1):
        result=model.predict_proba(query.assign(hypothetical_transport=side))
        valid=index>=0
        observed=np.ones(len(q))
        observed[valid]=result[np.flatnonzero(valid),index[valid]]
        probabilities.append(observed)
    return np.column_stack(probabilities),levels


def run(folds,views):
    import torch
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,f=data()
    x=compact_features(raw)
    OUT.mkdir(exist_ok=True)
    for k in folds:
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        for name in views:
            masked,attribute=masked_view(x,name)
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                seed=891+k
                prior=infer(masked,y,t,q,seed)
                actual,levels=conditional_observation(masked,attribute,y,t,q,seed,False)
                shuffled,_=conditional_observation(masked,attribute,y,t,q,seed,True)
                def update(p):
                    return expit(logit(prior.clip(.001,.999))+np.log(p[:,1].clip(.001,1)/p[:,0].clip(.001,1)))
                both=update(actual)
                sham=update(shuffled)
                np.testing.assert_allclose(update(np.ones_like(actual)),prior.clip(.001,.999),atol=2e-7)
                assert np.isfinite(np.r_[both,sham]).all()
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as buf:
                    np.savez_compressed(buf,prior=prior,observed=actual,shuffled=shuffled,restored=both,placebo=sham,
                        q=q,t=t,train_y=y[t],levels=levels,fields=masked.columns.to_numpy(str),
                        passenger=raw.PassengerId.iloc[q].to_numpy(str),checkpoint=DIGEST)
                tmp.replace(path)
            with np.load(path) as z:
                for key,expected in [('t',t),('q',q),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q]),('checkpoint',DIGEST)]:
                    np.testing.assert_array_equal(z[key],expected)
            print('reconstructed',k,name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folds',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--views',nargs='+',choices=VIEWS,default=list(VIEWS))
    args=p.parse_args()
    run(args.folds,args.views)
