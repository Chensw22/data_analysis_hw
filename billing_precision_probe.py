"""Fixed billing-digit residual test with magnitude controls and shuffled negative control."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/billing_precision'
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
L2=10.

def amounts(x):
    a=x[SPEND].to_numpy(float)
    if np.isinf(a).any() or np.any(a<0):raise ValueError('Invalid amount')
    total=np.where(np.isfinite(a).all(1),np.nansum(a,axis=1),np.nan)
    return np.column_stack([a,total])

def design(x):
    a=amounts(x);positive=np.isfinite(a)&(a>0);integer=positive&(a==np.floor(a))
    z=np.log1p(np.nan_to_num(a,nan=0.));state=np.column_stack([positive,np.isnan(a)]).astype(float)
    control=np.column_stack([z,z*z,state])
    blocks=[]
    for j in range(6):
        digit=np.where(integer[:,j],np.nan_to_num(a[:,j])%10,-1)
        blocks.append(np.column_stack([digit[:,None]==np.arange(10),(integer[:,j]&(np.nan_to_num(a[:,j])%100==0))]).astype(float))
    return control,blocks,a

def shuffled(blocks,a,x,seed):
    # Independently shuffle within each amount's positive magnitude decade and
    # Planet/Cryo stratum. No labels, no cross train/query transfer.
    rng=np.random.default_rng(seed);out=[];movable=[]
    group=(x.HomePlanet.astype(str)+'|'+x.CryoSleep.astype(str)).to_numpy()
    for j,block in enumerate(blocks):
        positive=np.isfinite(a[:,j])&(a[:,j]>0)
        decade=np.full(len(a),-1,int);decade[positive]=np.floor(np.log10(a[positive,j])).astype(int)
        keys=np.array([f'{g}|{d}' for g,d in zip(group,decade)])
        copy=block.copy();n=0
        for k in np.unique(keys[positive]):
            idx=np.flatnonzero(positive&(keys==k));copy[idx]=block[rng.permutation(idx)]
            if len(idx)>1:n+=len(idx)
        out.append(copy);movable.append(n)
    return np.column_stack(out),movable

def fit(p,y,x):
    offset=r.logit(p)
    def obj(w):
        eta=offset+w[0]+x@w[1:];err=expit(eta)-y
        return float(np.sum(np.logaddexp(0,eta)-y*eta)+.5*L2*(w[1:]@w[1:])),np.r_[err.sum(),x.T@err+L2*w[1:]]
    opt=minimize(obj,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-6))
    if not opt.success or not np.isfinite(opt.x).all():raise RuntimeError(str(opt.message))
    return opt.x,dict(iterations=int(opt.nit),objective=float(opt.fun),gradient_max=float(np.abs(opt.jac).max()))

def check():
    x=pd.DataFrame({s:[0.,np.nan,120.,123.] for s in SPEND});x['HomePlanet']='Earth';x['CryoSleep']='False'
    c,b,a=design(x);d=np.column_stack(b)
    assert d.shape==(4,66) and not d[:2].any() and np.isnan(a[1,-1])
    assert b[0][2,0]==1 and b[0][3,3]==1 and b[-1][2,10]==1
    shuffled(b,a,x,1)

def run():
    check()
    if OUT.exists():raise FileExistsError('Single fixed protocol, no overwrite.')
    OUT.mkdir(parents=True);start=time.monotonic()
    names=['billing_precision_probe.py','error_overlap_diagnostic.py','relational_error_corrector.py','planet_mixture_experiment.py','exaone_validation_data.py','artifacts/error_rescue_20261117/catalog_rows.csv']
    sources={n:r.sha(ROOT/n) for n in names}
    plan=dict(status='frozen_before_new_fit',hypothesis='Positive billing last digits and round hundreds may identify different generation/price regimes beyond amount scale, zero and missingness; no new information beyond amounts.',
        amount_fields=SPEND+['complete_total'],digits='For each positive integer amount: ten one-hot decimal lastdigits and indicator divisible100. Zero/missing/noninteger: all0. Total only if all5 observed.',
        controls='Six log1p amounts and squares (training mean/std); positive/missing flags; train-vocabulary Planet×Cryo indicators. No Name.',
        negative_control='Permute the 11-column digit block within Planet×Cryo×floor(log10 positive amount), separately per bill/total and separately train/query, seeds20261118+100fold and+1. Keep amount controls unchanged. No label access in shuffle.',
        objective='sum binary NLL with fixed strictnested rawlogit offset + 5*sum(coefficient²), unpenalized intercept; all coefficients regularized equally after continuous-controls scaling, digits remain0/1.',
        models=['magnitude_control','precision','shuffled_precision'],fixed_L2=L2,threshold=.5,seed=20261112,folds=3,
        training='All outertrain rows, same cached9innerOOF; no546 weighting/selection. Outerlabels only score afterprediction.',
        limitations=['Same reused6954dev adaptive experiment.','One shuffled negative control is descriptive, not a permutation significance test.','Digit terms are mostly zero for Earthsleep: cannot discriminate allzero minority.','Total digits depend on component digits, not independent evidence.','Residual digit association could reflect unmodeled magnitude/price bands rather than a causal generation regime.'],
        budget_seconds=120,sources_sha256=sources)
    r.write(OUT/'protocol.json',plan)
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU120s')));signal.setitimer(signal.ITIMER_REAL,120)
    try:
        data,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y'];raw=bases['rawMitra'];x=data['features'].loc[dev]
        c,blocks,a=design(x);digits=np.column_stack(blocks)
        # Descriptive attributes only, no label relationship inspected before fits.
        stats={}
        for j,name in enumerate(SPEND+['complete_total']):
            vals=a[:,j];pos=vals[np.isfinite(vals)&(vals>0)];ints=pos[pos==np.floor(pos)]
            stats[name]=dict(observed=int(np.isfinite(vals).sum()),zero=int(np.sum(vals==0)),positive=len(pos),noninteger_positive=int(len(pos)-len(ints)),
                positive_lastdigit_counts=np.bincount((ints%10).astype(int),minlength=10).tolist(),positive_multiple100=int(np.sum(ints%100==0)))
        r.write(OUT/'attribute_diagnostic.json',stats)
        frozen=r.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json');r.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
        inners=[r.load_inner(i,aa,bb,dev,y,evidence) for i,(aa,bb) in enumerate(splits)]
        evidence.update(sources);r.write(OUT/'evidence.json',evidence)
        pred={name:np.full(len(y),np.nan) for name in plan['models']};pred['raw']=raw.copy();records=[]
        keys=(x.HomePlanet.astype(str)+'|'+x.CryoSleep.astype(str)).to_numpy()
        with threadpool_limits(limits=4):
            for fold,((aa,bb),inner) in enumerate(zip(splits,inners)):
                mean=c[aa,:12].mean(0);scale=c[aa,:12].std(0);scale[scale==0]=1
                ctrl=c.copy();ctrl[:,:12]=(ctrl[:,:12]-mean)/scale
                vocab=sorted(set(keys[aa]));ctrl=np.column_stack([ctrl,keys[:,None]==np.array(vocab)])
                negtrain,movtrain=shuffled([b[aa] for b in blocks],a[aa],x.iloc[aa],20261118+100*fold)
                negquery,movquery=shuffled([b[bb] for b in blocks],a[bb],x.iloc[bb],20261119+100*fold)
                designs={'magnitude_control':(ctrl[aa],ctrl[bb]),'precision':(np.column_stack([ctrl[aa],digits[aa]]),np.column_stack([ctrl[bb],digits[bb]])),
                    'shuffled_precision':(np.column_stack([ctrl[aa],negtrain]),np.column_stack([ctrl[bb],negquery]))}
                fitted={}
                for name,(train,query) in designs.items():
                    coef,meta=fit(inner,y[aa],train);pred[name][bb]=expit(r.logit(raw[bb])+coef[0]+query@coef[1:]);fitted[name]=dict(coefficients=coef.tolist(),**meta)
                pack=dict(fold=fold,mean=mean.tolist(),scale=scale.tolist(),vocabulary=vocab,fitted=fitted,
                    negative_movable_train=movtrain,negative_movable_query=movquery,
                    metrics={k:r.score(p[bb],y[bb]) for k,p in pred.items()})
                r.write(OUT/f'fold{fold}.json',pack);records.append(pack)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for key,expected in [('row_id',dev),('y',y),('fold',folds),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[key],expected)
        masks=dict(all=np.ones(len(y),bool),hard546=rows.all_recipes_still_wrong.to_numpy(bool),hard914=rows.family_wrong_count.to_numpy()==5,
            commoncorrect5263=rows.family_wrong_count.to_numpy()==0,Earth_sleep=(x.HomePlanet.eq('Earth')&x.CryoSleep.eq('True')).to_numpy(),Earth_awake=(x.HomePlanet.eq('Earth')&x.CryoSleep.eq('False')).to_numpy())
        def paired(mask):return {k:dict(correct=int((((p>=.5)==y)&mask).sum()),versus_raw=r.pair(p,raw,y,mask),versus_magnitude=r.pair(p,pred['magnitude_control'],y,mask)) for k,p in pred.items()}
        for i,rec in enumerate(records):rec['paired']=paired(folds==i);rec['hard546']=paired((folds==i)&masks['hard546']);r.write(OUT/f'fold{i}.json',rec)
        cache=ROOT/'artifacts/billing_precision_oof.npz'
        if cache.exists():raise FileExistsError('Prediction cache exists')
        np.savez_compressed(cache,**pred,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str))
        for name,digest in evidence.items():
            if r.sha(ROOT/name)!=digest:raise RuntimeError('Frozen evidence changed '+name)
        result=dict(status='complete',metrics={k:r.score(p,y) for k,p in pred.items()},subsets={k:paired(m) for k,m in masks.items()},folds=records,
            seconds=time.monotonic()-start,protocol_sha256=r.sha(OUT/'protocol.json'),cache_sha256=r.sha(cache),no_upload=True,no_new_base_fit=True,old_holdout_labels_read=False)
        r.write(OUT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:r.write(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
