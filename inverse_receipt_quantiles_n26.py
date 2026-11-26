"""Identify a two-service subtotal from Nori quantiles conditional on each transport hypothesis."""
import argparse
import json
from itertools import combinations

import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier

from matched_state_context_n26 import inputs,one_per_order,ROOT
from nori_booking_regression_n26 import CACHE,SHA,design
from nori_fare_party_n26 import FEES,predictor

OUT=ROOT/'artifacts/nov26_inverse_receipt_quantiles'
REPORT=ROOT/'reports/nov23_matched_learning_curve/inverse_receipt_quantiles.json'
COLUMNS=[FEES.index('FoodCourt'),FEES.index('ShoppingMall')]


def domains(bills,query):
    lo=np.full(len(query),-np.inf)
    hi=np.full(len(query),np.inf)
    distinct=np.empty(len(query),int)
    for j,i in enumerate(query):
        v=bills[i]
        attainable=np.unique([v[a]+v[b] for a,b in combinations(range(5),2)])
        actual=v[COLUMNS].sum()
        p=np.searchsorted(attainable,actual)
        assert attainable[p]==actual and v.sum()>0
        distinct[j]=len(attainable)
        if p:lo[j]=(actual+attainable[p-1])/(2*v.sum())
        if p+1<len(attainable):hi[j]=(actual+attainable[p+1])/(2*v.sum())
    return lo,hi,distinct


def mass(quantiles,taus,lo,hi,distinct):
    assert np.isfinite(quantiles).all() and np.all(np.diff(quantiles,axis=1)>=0)
    cdf=lambda x,r:np.interp(x,r,taus,left=0.,right=1.)
    p=np.fromiter((cdf(b,r)-cdf(a,r) for a,b,r in zip(lo,hi,quantiles,strict=True)),float,len(lo))
    assert np.all((p>=0)&(p<=1))
    return .97*p+.03/distinct


def train():
    from threadpoolctl import threadpool_limits
    frame,y,g,f,_=inputs()
    bills=frame[FEES].to_numpy()
    eligible=np.isfinite(bills).all(axis=1)&(np.nansum(bills,axis=1)>0)
    # Target and quantile-cell boundaries are formed from the passenger's observed bill.
    ratio=np.nansum(bills[:,COLUMNS],axis=1)/np.maximum(1,np.nansum(bills,axis=1))
    ordered=np.log1p(np.sort(bills,axis=1))
    torch,regressor=predictor()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    n=len(y)
    for k in range(9):
        bank=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        rows=np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
        t,q=bank[eligible[bank]],rows[eligible[rows]]
        path=OUT/f'fold{k}.npz'
        if path.exists():
            with np.load(path) as z:
                for key,v in [('t',t),('q',q),('labels',y[t]),('passenger',frame.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],v)
                assert str(z['sha'])==SHA
            continue
        assert not np.isin(g[t],g[q]).any()
        # Last five original columns contain the named charge amounts; retain all other fields.
        reg=design(frame,g,bank)
        np.testing.assert_allclose(reg[:,-5:],np.log1p(bills),rtol=1e-6,atol=1e-6,equal_nan=True)
        masked=np.column_stack([reg[:,:-5],ordered]).astype(np.float32)
        seed=6128+k
        rng=np.random.default_rng(seed)
        s=one_per_order(t,g,rng)[:768]
        assert len(s)==len(set(g[s]))==768
        estimator=lambda:HistGradientBoostingClassifier(max_iter=170,max_leaf_nodes=13,l2_regularization=4,
                          min_samples_leaf=22,learning_rate=.065,early_stopping=False,random_state=17)
        with threadpool_limits(1):
            prior=estimator().fit(masked[t],y[t]).predict_proba(masked[q])[:,1]
            direct=estimator().fit(reg[t],y[t]).predict_proba(reg[q])[:,1]
        lo,hi,nd=domains(bills,q)
        scores={}
        fits=y[s].astype(float)
        queries=np.vstack([np.column_stack([masked[q],np.full(len(q),v)]) for v in (0.,1.)]).astype(np.float32)
        for name,hypothesis in [('actual',fits),('shuffled',rng.permutation(fits))]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model=regressor(model_path=str(CACHE/'nori.pt'),model='nori-6m',device='cpu',
                            inference_config=str(CACHE/'one_member.json'),categorical_columns=[],augmentations=())
            model.fit(np.column_stack([masked[s],hypothesis]).astype(np.float32),ratio[s])
            prediction=model.predict(queries,output_type='full')
            quantiles=prediction['quantiles'].reshape(2,len(q),-1)
            observed=np.column_stack([mass(v,prediction['taus'],lo,hi,nd) for v in quantiles])
            scores[name]=expit(logit(prior.clip(.001,.999))+np.log(observed[:,1]/observed[:,0]))
            scores[name+'_mass']=observed
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,t=t,q=q,s=s,labels=y[t],passenger=frame.PassengerId.iloc[q].to_numpy(str),
                            prior=prior,direct=direct,sha=SHA,alternatives=nd,**scores)
        tmp.replace(path)
        print(k,len(q),flush=True)


def report():
    from exhaustive_family_quorums_n26 import summary
    from latent_passenger_difficulty_n26 import OUT as PANEL
    frame,y,g,f,masks=inputs()
    n=len(y)
    bills=frame[FEES].to_numpy()
    eligible=np.isfinite(bills).all(axis=1)&(np.nansum(bills,axis=1)>0)
    scores={k:np.full(len(g),np.nan) for k in ['actual','shuffled','prior','direct']}
    ll={}
    for k in range(9):
        with np.load(OUT/f'fold{k}.npz') as z:
            q=z['q']
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId.iloc[q])
            np.testing.assert_array_equal(z['labels'],y[z['t']])
            assert not np.isin(g[z['s']],g[q]).any()
            for name,arr in scores.items(): arr[q]=z[name]
            if k<8:
                for name in ('actual','shuffled'):
                    ll.setdefault(name,[]).extend(-np.log(z[name+'_mass'][np.arange(len(q)),y[q].astype(int)]))
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        base,common,known,fresh=[z[k] for k in ('anchor','common','known','fresh')]
    assert all(np.array_equal(eligible,np.isfinite(v)) for v in scores.values())
    votes={k:eligible&((v>=.5)!=base) for k,v in scores.items()}
    actions=dict(**votes, inverse_only=votes['actual']&~votes['prior']&~votes['shuffled'],confirmed=votes['actual']&votes['direct'])
    scopes=dict(all=known,common=common,fresh=known&fresh,hard_awake=common&masks['awake_paid'][:n]&~base[:n])
    metrics={k:{s:summary(a[:n],base[:n]!=y,m,f) for s,m in scopes.items()} for k,a in actions.items()}
    m=eligible[:n]
    result=dict(eligible=int(m.sum()),test_eligible=int(eligible[n:].sum()),
                accuracy={k:float(((v[:n][m]>=.5)==y[m]).mean()) for k,v in scores.items()},
                observed_subtotal_logloss={k:float(np.mean(v)) for k,v in ll.items()},
                against_161=metrics,test_changes={k:int(a[n:].sum()) for k,a in actions.items()})
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**scores)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['accuracy'],result['observed_subtotal_logloss'],[(k,v['all']['net']) for k,v in metrics.items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','report'])
    if parser.parse_args().action=='train': train()
    else: report()
