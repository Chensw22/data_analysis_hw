"""One fixed cross-fitted difficulty-weighted CatBoost and matched half-mixtures."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
import iterate
import relational_error_corrector as r
import error_overlap_diagnostic as overlap
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/hard_weighted_combo'
PARAMS=dict(depth=6,iterations=700,learning_rate=.045,l2_leaf_reg=5,
    loss_function='Logloss',thread_count=4,verbose=False,allow_writing_files=False)

def weights(p,y):
    p=np.asarray(p,float);y=np.asarray(y)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():raise ValueError('Invalid difficulty inputs')
    true_probability=np.where(y==1,p,1-p)
    w=1+2*(1-true_probability)
    return w/w.mean()

def run():
    if OUT.exists():raise FileExistsError('Fixed experiment cannot overwrite/retry.')
    OUT.mkdir(parents=True);start=time.monotonic()
    names=['hard_weighted_combo.py','iterate.py','experiment.py','relational_error_corrector.py',
        'error_overlap_diagnostic.py','planet_mixture_experiment.py','exaone_validation_data.py',
        'artifacts/error_rescue_20261117/catalog_rows.csv']
    sources={n:r.sha(ROOT/n) for n in names}
    plan=dict(status='fixed_before_fitting',parameters=PARAMS,seed='20261112+outerfold',
        weight='w=(1+2*(1-p_true))/mean(1+2*(1-p_true)); p_true from matching strict nested innerOOF only. Fixed slope2; no sweep.',
        candidates=['ordinary_cat','weighted_cat','ordinary_half_raw','weighted_half_raw'],
        mixture='Each Cat probability .5 plus rawMitra .5. No selection of weights/seeds/subgroups.',
        input='engineered_no_ids 30 columns; raw Name excluded before feature construction, hence MissingCount excludes Name for BOTH Cat models.',
        boundary='All outertrain rows; no outerquery labels enter fitting or weights; 546 and other error masks reporting only after all predictions.',
        matched='Identical features, iterations, seed and Cat defaults. Only sample_weight differs; normalized mean1.',
        evaluation='Original 6954dev SKF3seed20261112; cached nine strict inner raw packets. No1739 holdout labels.',
        limitations=['Adaptive reused development set, not independent validation.','Cross-fitted difficulty may emphasize label noise or irreducible ambiguity.','Inner teacher has smaller support than outer teacher.','Four prespecified outcomes reported together, not a searched ensemble.'],
        budget_seconds=600,sources_sha256=sources)
    r.write(OUT/'protocol.json',plan)
    def stop(*_):raise TimeoutError('Fixed600secondCPU budget')
    previous=signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600)
    try:
        data,splits,fold_ids,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y'];raw=bases['rawMitra']
        frozen=r.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json');r.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
        inners=[r.load_inner(i,a,b,dev,y,evidence) for i,(a,b) in enumerate(splits)]
        attrs=pd.concat([pd.read_csv(ROOT/'data/raw'/n,usecols=lambda c:c not in ('Transported','Name')) for n in ['train.csv','test.csv']],ignore_index=True)
        x,cats=iterate.build_features(attrs,False)
        assert x.shape==(12970,30) and 'Name' not in x and 'PassengerId' not in x
        np.testing.assert_array_equal(attrs.PassengerId.iloc[dev].to_numpy(),data['passengers'])
        for f in ['train.csv','test.csv']:evidence['data/raw/'+f]=r.sha(ROOT/'data/raw'/f)
        evidence.update(sources);r.write(OUT/'evidence.json',evidence)
        pred={k:np.full(len(y),np.nan) for k in plan['candidates']};pred['raw']=raw.copy();records=[]
        with threadpool_limits(limits=4):
            for fold,((a,b),inner) in enumerate(zip(splits,inners)):
                w=weights(inner,y[a]);t=time.monotonic()
                for kind in ['ordinary','weighted']:
                    model=CatBoostClassifier(**PARAMS,random_seed=20261112+fold,cat_features=cats)
                    model.fit(x.iloc[dev[a]],y[a],sample_weight=None if kind=='ordinary' else w)
                    p=model.predict_proba(x.iloc[dev[b]])[:,1]
                    pred[kind+'_cat'][b]=p;pred[kind+'_half_raw'][b]=.5*p+.5*raw[b]
                    del model
                rec=dict(fold=fold,train_rows=len(a),query_rows=len(b),weight_min=float(w.min()),weight_max=float(w.max()),weight_mean=float(w.mean()),
                    train_ids_sha256=r.nested.ids_hash(dev[a]),query_ids_sha256=r.nested.ids_hash(dev[b]),
                    metrics={k:r.score(p[b],y[b]) for k,p in pred.items()},seconds=time.monotonic()-t)
                records.append(rec);r.write(OUT/f'fold{fold}.json',rec);print(json.dumps(rec),flush=True)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for name,expected in [('row_id',dev),('y',y),('fold',fold_ids),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[name],expected)
        masks=dict(all=np.ones(len(y),bool),hard546=rows.all_recipes_still_wrong.to_numpy(bool),hard914=rows.family_wrong_count.to_numpy()==5,
            commoncorrect5263=rows.family_wrong_count.to_numpy()==0)
        f=data['features'].loc[dev]
        masks['Earth_sleep']=(f.HomePlanet.eq('Earth')&f.CryoSleep.eq('True')).to_numpy()
        masks['Earth_awake']=(f.HomePlanet.eq('Earth')&f.CryoSleep.eq('False')).to_numpy()
        assert masks['hard546'].sum()==546
        def report(mask):
            return {k:dict(correct=int(((p>=.5)==y)[mask].sum()),versus_raw=r.pair(p,raw,y,mask),
                versus_ordinary_cat=r.pair(p,pred['ordinary_cat'],y,mask),versus_ordinary_half=r.pair(p,pred['ordinary_half_raw'],y,mask)) for k,p in pred.items()}
        subsets={name:report(mask) for name,mask in masks.items()}
        for fold,(_,b) in enumerate(splits):
            fm=fold_ids==fold;records[fold]['paired']=report(fm);records[fold]['hard546']=report(fm&masks['hard546'])
            r.write(OUT/f'fold{fold}.json',records[fold])
        cache=ROOT/'artifacts/hard_weighted_combo_oof.npz'
        if cache.exists():raise FileExistsError('Prediction cache exists')
        np.savez_compressed(cache,**pred,dev_ids=dev,fold_id=fold_ids,passenger_ids=np.asarray(data['passengers'],dtype=str))
        for name,digest in evidence.items():
            if r.sha(ROOT/name)!=digest:raise RuntimeError('Source/evidence changed '+name)
        result=dict(status='complete',metrics={k:r.score(p,y) for k,p in pred.items()},subsets=subsets,folds=records,
            seconds=time.monotonic()-start,protocol_sha256=r.sha(OUT/'protocol.json'),cache_sha256=r.sha(cache),
            cache_path=str(cache.relative_to(ROOT)),no_upload=True,old_holdout_labels_read=False)
        r.write(OUT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:
        r.write(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

if __name__=='__main__':
    # Fixed formula check only; no real-label peeking to choose the slope.
    np.testing.assert_array_equal(weights(np.array([.5,.5]),np.array([0,1])),np.ones(2))
    assert weights(np.array([.9,.1]),np.array([1,1]))[1]>weights(np.array([.9,.1]),np.array([1,1]))[0]
    run()
