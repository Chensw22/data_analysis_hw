"""Fixed regularized activity-pattern intercepts on strictly nested raw-Mitra logits."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import relational_error_corrector as relational
import error_overlap_diagnostic as overlap

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/nov18_alternative'
PLAN=REPORT/'protocol.json'
CACHE=ROOT/'artifacts/nov18_alternative_oof.npz'
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
LAMBDA=20.
sha,write=relational.sha,relational.write


def patterns(x):
    values=x[SPEND].to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(values<0):raise ValueError('Invalid spending.')
    states=np.where(np.isnan(values),0,np.where(values==0,1,2))
    codes=states@(3**np.arange(5))
    # A categorical joint pattern: its integer order is never a numerical predictor.
    return np.array([json.dumps([str(h),str(c),int(k)],separators=(',',':'))
        for h,c,k in zip(x.HomePlanet,x.CryoSleep,codes)],dtype=str)


def objective(theta,offset,y,index):
    eta=offset+theta[0]
    if len(theta)>1:eta=eta+theta[1:][index]
    r=expit(eta)-y
    value=float(np.sum(np.logaddexp(0,eta)-y*eta)+.5*LAMBDA*(theta[1:]@theta[1:]))
    gradient=np.r_[r.sum(),np.bincount(index,weights=r,minlength=len(theta)-1)+LAMBDA*theta[1:]] if len(theta)>1 else np.array([r.sum()])
    return value,gradient


def fit(p,y,keys,constant=False):
    vocab=sorted(set(keys)) if not constant else []
    lookup={k:i for i,k in enumerate(vocab)}
    index=np.array([lookup[k] for k in keys],dtype=int) if vocab else np.zeros(len(y),int)
    result=minimize(objective,np.zeros(len(vocab)+1),args=(relational.logit(p),y,index),jac=True,
        method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-6))
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('Fixed optimizer failed: '+str(result.message))
    return dict(intercept=float(result.x[0]),pattern_offsets=dict(zip(vocab,result.x[1:].tolist())),
        objective=float(result.fun),iterations=int(result.nit),gradient_max=float(abs(result.jac).max()),lambda_=LAMBDA)


def predict(pack,p,keys):
    delta=np.array([pack['pattern_offsets'].get(k,0.) for k in keys])
    return expit(relational.logit(p)+pack['intercept']+delta)


def check():
    x=pd.DataFrame({'HomePlanet':['Earth']*3,'CryoSleep':['False']*3,**{s:[0.,0.,1.] for s in SPEND}})
    x.loc[1,'Spa']=np.nan;k=patterns(x);assert len(set(k))==3
    off=np.array([-.2,.4,1.]);y=np.array([0,1,0]);idx=np.array([0,1,0]);t=np.array([.2,-.1,.4])
    _,g=objective(t,off,y,idx);finite=[]
    for i in range(3):
        d=np.zeros(3);d[i]=1e-6;finite.append((objective(t+d,off,y,idx)[0]-objective(t-d,off,y,idx)[0])/2e-6)
    np.testing.assert_allclose(g,finite,atol=1e-7,rtol=1e-6)
    pack=dict(intercept=.2,pattern_offsets={k[0]:.1})
    a=predict(pack,np.array([.2,.3,.4]),k);b=predict(json.loads(json.dumps(pack)),np.array([.2,.3,.4]),k)
    np.testing.assert_array_equal(a,b)
    assert predict(pack,np.array([.5]),['unseen'])[0]==expit(.2)
    print('Synthetic jointstates/gradient/intercept/unknown/reload checks passed.')


def run():
    if (REPORT/'result.json').exists() or CACHE.exists():raise RuntimeError('One experiment only; nooverwrite.')
    plan=json.loads(PLAN.read_text());started=time.monotonic()
    for name,digest in plan['source_and_prior_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen source/prior changed: '+name)
    def timeout(*_):raise TimeoutError('CPU120second budget exceeded.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra']
        keys=patterns(inputs['features'].loc[dev])
        frozen=relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        relational.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
        inners=[relational.load_inner(i,a,b,dev,y,evidence) for i,(a,b) in enumerate(splits)]
        evidence.update(plan['source_and_prior_sha256'])
        write(REPORT/'evidence.json',evidence) # All nine packets authenticated before any fitting.
        pred=np.full(len(y),np.nan);constant=pred.copy();records=[]
        for i,((a,b),inner) in enumerate(zip(splits,inners)):
            model=fit(inner,y[a],keys[a]);control=fit(inner,y[a],keys[a],constant=True)
            pred[b]=predict(model,raw[b],keys[b]);constant[b]=predict(control,raw[b],keys[b])
            path=REPORT/f'fold{i}_pack.json';write(path,dict(model=model,control=control))
            restored=json.loads(path.read_text())
            np.testing.assert_array_equal(pred[b],predict(restored['model'],raw[b],keys[b]))
            np.testing.assert_array_equal(constant[b],predict(restored['control'],raw[b],keys[b]))
            records.append(dict(fold=i,training_patterns=len(model['pattern_offsets']),
                unseen_query_patterns=int(sum(k not in model['pattern_offsets'] for k in keys[b])),
                pack_sha256=sha(path),reload_max_difference=0.,
                metrics={n:relational.score(v,y[b]) for n,v in [('activity',pred[b]),('constant',constant[b]),('raw',raw[b])]},
                versus_raw=relational.pair(pred[b],raw[b],y[b]),versus_constant=relational.pair(pred[b],constant[b],y[b])))
        # No outcome-defined mask has been read/constructed before fitting completes.
        catalog_path=ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv'
        rows=pd.read_csv(catalog_path,dtype={'PassengerId':str})
        for c,want in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[c],want)
        x=inputs['features'].loc[dev]
        masks=dict(all=np.ones(len(y),bool),hard546=rows.all_recipes_still_wrong.to_numpy(bool),
            hard616=rows.all_primary_still_wrong.to_numpy(bool),hard914=rows.family_wrong_count.to_numpy()==5,
            commoncorrect5263=rows.family_wrong_count.to_numpy()==0,
            Earth_awake=(x.HomePlanet.eq('Earth')&x.CryoSleep.eq('False')).to_numpy(),
            Earth_sleep=(x.HomePlanet.eq('Earth')&x.CryoSleep.eq('True')).to_numpy())
        assert masks['hard546'].sum()==546 and masks['hard914'].sum()==914
        subset={}
        for name,m in masks.items():
            subset[name]=dict(versus_raw=relational.pair(pred,raw,y,m),versus_constant=relational.pair(pred,constant,y,m),
                correct={n:int((((v>=.5)==y)&m).sum()) for n,v in [('activity',pred),('constant',constant),('raw',raw)]},
                confusion={n:dict(tn=int((m&(y==0)&(v<.5)).sum()),fp=int((m&(y==0)&(v>=.5)).sum()),
                    fn=int((m&(y==1)&(v<.5)).sum()),tp=int((m&(y==1)&(v>=.5)).sum()))
                    for n,v in [('activity',pred),('constant',constant),('raw',raw)]})
        np.savez_compressed(CACHE,prob=pred,constant=constant,raw=raw,dev_ids=dev,fold_id=folds,
            passenger_ids=np.asarray(inputs['passengers'],dtype=str),**masks)
        for name,digest in evidence.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Evidence changed: '+name)
        result=dict(status='complete_fixed_adaptive_CPU_experiment',folds=records,
            metrics={n:relational.score(v,y) for n,v in [('activity',pred),('constant',constant),('raw',raw)]},
            subsets=subset,seconds=time.monotonic()-started,protocol_sha256=sha(PLAN),
            predictions_path=str(CACHE.relative_to(ROOT)),predictions_sha256=sha(CACHE),
            limitations=plan['limitations'],no_new_base_fit=True,no_upload=True)
        write(REPORT/'result.json',result)
        print(json.dumps(dict(metrics=result['metrics'],paired=subset['all'],hard546=subset['hard546'],seconds=result['seconds']),indent=2))
    except BaseException as error:
        write(REPORT/'failure.json',dict(error_type=type(error).__name__,error=str(error)));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check() if args.check else run()
