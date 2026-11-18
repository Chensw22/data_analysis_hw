"""Fixed nested-teacher distillation; CPU students only, no upload."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import hashlib
import json
from pathlib import Path
import sys
import time
import signal
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import nonlinear_error_corrector as residual
import exception_error_detector_nov18 as reporting

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/teacher_distillation'
OUT=ROOT/'artifacts/teacher_distillation'
PLAN=REPORT/'protocol.json'
PARAMS=dict(loss_function='CrossEntropy',depth=6,iterations=700,learning_rate=.045,l2_leaf_reg=5,
            thread_count=4,verbose=False,allow_writing_files=False)
sha=residual.sha
write=residual.write


def freeze():
    if PLAN.exists():raise FileExistsError('Fixed plan already exists.')
    write(PLAN,dict(status='frozen_before_student_fits',parameters=PARAMS,features=residual.COLUMNS,
        teacher='0.5*strict_outertrain_innerOOF_rawMitra+0.5*strict_outertrain_innerOOF_Kumo',
        targets={'hard':'y','uniform':'0.5*y+0.25','distilled':'0.5*y+0.5*teacher'},
        predictions='Each standalone student and fixed 0.5*student+0.5*authenticated outer rawMitra',
        folds=3,outer_seed=20261112,model_seed='20261112+outer',budget_seconds=600,
        labels='Only original6954 development labels; no historical-hard-subset fitting or teacher reuse across outer folds.',
        masks='Load914/616/546/5263 only after all9student fits/predictions.',
        controls='Same CrossEntropy loss, features30 and params for all3targets; hard label control is fresh, not assumed identical to oldLogloss.',
        no_Name=True,no_base_fits=True,no_holdout_labels=True,no_upload=True,
        limitations=['Teacher task-specific probabilities add sample-specific training information, not new raw inputs.',
          'Inner teacher support is about3090 vs outer4636; teacher errors/calibration may propagate.',
          'Repeated adaptive development is not independent evidence of public85percent accuracy.'],
        source_hashes={p:sha(ROOT/p) for p in ['teacher_distillation.py','nonlinear_error_corrector.py','relational_error_corrector.py','planet_mixture_experiment.py','exception_error_detector_nov18.py','error_overlap_diagnostic.py']},
        historical_hashes={'artifacts/planet_mixture_20261117/freeze.json':sha(ROOT/'artifacts/planet_mixture_20261117/freeze.json')}))


def verify_plan():
    p=json.loads(PLAN.read_text())
    if p['parameters']!=PARAMS or p['features']!=residual.COLUMNS:raise RuntimeError('Fixed recipe changed.')
    for n,d in {**p['source_hashes'],**p['historical_hashes']}.items():
        if sha(ROOT/n)!=d:raise RuntimeError('Frozen source/evidence changed: '+n)
    return p


def kumo_inner(outer,a,b,dev,y,evidence):
    nested=residual.relational.nested;old=ROOT/'artifacts/planet_mixture_20261117';pub=ROOT/'reports/planet_mixture_20261117'
    row=nested.read(old/f'outer{outer}.json')
    assert sha(old/f'outer{outer}.json')==sha(pub/f'outer{outer}.json')
    assert row['outer_train_ids_sha256']==nested.ids_hash(dev[a])
    assert row['outer_validation_ids_sha256']==nested.ids_hash(dev[b])
    p=np.full(len(a),np.nan);coverage=np.zeros(len(a),int)
    for inner,(ia,ib) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261117+outer).split(a,y[a])):
        folder=old/'jobs'/f'outer{outer}_inner{inner}_kumo'
        for n in ['job.json','worker_result.json','metadata.json']:
            assert sha(folder/n)==sha(pub/'jobs'/folder.name/n)
            evidence[str((folder/n).relative_to(ROOT))]=sha(folder/n)
        job=nested.read(folder/'job.json');record=nested.read(folder/'worker_result.json');meta=nested.read(folder/'metadata.json')
        assert job['kind']=='kumo' and job['outer_fold']==outer and job['inner_fold']==inner
        assert job['seed']==20261117+100*outer+inner
        assert sha(folder/'rows.npz')==job['rows_sha256']
        with np.load(folder/'rows.npz',allow_pickle=False) as z:
            assert set(z.files)=={'train_ids','query_ids','outer_train_ids','outer_validation_ids','train_labels'}
            for key,want in [('train_ids',dev[a[ia]]),('query_ids',dev[a[ib]]),('outer_train_ids',dev[a]),('outer_validation_ids',dev[b]),('train_labels',y[a[ia]])]:np.testing.assert_array_equal(z[key],want)
        assert not np.intersect1d(dev[b],np.r_[dev[a[ia]],dev[a[ib]]]).size
        q=nested.verify_prediction(folder,record,job,dev[a[ia]],dev[a[ib]],y[a[ia]])
        assert meta['candidate']=='v83_Kumo_medium8' and meta['training_rows']==len(ia) and meta['feature_count']==13 and meta['seed']==job['seed']
        assert meta['settings']['n_estimators']==8
        p[ib]=q;coverage[ib]+=1
        for n in ['rows.npz','prediction.npz']:evidence[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    assert (coverage==1).all() and np.isfinite(p).all()
    cache=old/f'outer{outer}_inner_oof.npz'
    assert sha(cache)==row['inner_oof_sha256']
    with np.load(cache,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['outer_train_ids'],dev[a]);np.testing.assert_array_equal(z['kumo'],p)
    evidence[str(cache.relative_to(ROOT))]=sha(cache)
    return p


def run():
    plan=verify_plan()
    if OUT.exists() or (REPORT/'result.json').exists():raise FileExistsError('No overwrite.')
    OUT.mkdir(parents=True);started=time.monotonic()
    def timeout(*_):raise TimeoutError('Fixed600sCPU budget exceeded.')
    prior=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    try:
        inputs,splits,fold_ids,bases,evidence=residual.overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra'];x=residual.features()
        frozen=residual.relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        residual.relational.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
        probabilities={'raw':raw,**{n:np.full(len(dev),np.nan) for n in ['hard','uniform','distilled']}};folds=[]
        for outer,(a,b) in enumerate(splits):
            r=residual.relational.load_inner(outer,a,b,dev,y,evidence);k=kumo_inner(outer,a,b,dev,y,evidence);teacher=.5*(r+k)
            targets={'hard':y[a].astype(float),'uniform':.5*y[a]+.25,'distilled':.5*y[a]+.5*teacher}
            models={}
            for name,t in targets.items():
                assert np.isfinite(t).all() and t.min()>=0 and t.max()<=1
                model=CatBoostClassifier(**PARAMS,random_seed=20261112+outer,cat_features=residual.CATS)
                model.fit(x.iloc[dev[a]],t)
                p=model.predict_proba(x.iloc[dev[b]])[:,1];probabilities[name][b]=p
                path=OUT/f'{name}_fold{outer}.cbm';model.save_model(str(path))
                loaded=CatBoostClassifier();loaded.load_model(str(path));np.testing.assert_array_equal(p,loaded.predict_proba(x.iloc[dev[b]])[:,1])
                models[name]=dict(model_sha256=sha(path),reload_difference=0.,target_min=float(t.min()),target_max=float(t.max()))
                del model,loaded
            np.savez_compressed(OUT/f'fold{outer}_training.npz',train_ids=dev[a],query_ids=dev[b],raw_teacher=r,kumo_teacher=k,distilled_target=targets['distilled'])
            for name in targets:probabilities[name+'_raw_half']=.5*(probabilities[name]+raw)
            row=dict(fold=outer,models=models,metrics={n:residual.relational.score(p[b],y[b]) for n,p in probabilities.items()},paired_to_raw={n:residual.relational.pair(p[b],raw[b],y[b]) for n,p in probabilities.items() if n!='raw'})
            folds.append(row);write(REPORT/f'fold{outer}.json',row)
            print(json.dumps(dict(fold=outer,metrics=row['metrics'])),flush=True)
        if not all(np.isfinite(p).all() for p in probabilities.values()):raise RuntimeError('Incomplete predictions.')
        masks=reporting.reporting_masks(dev,fold_ids,y,inputs,bases,evidence)
        f=x.iloc[dev];masks.update(Earth_sleep=(f.HomePlanet.eq('Earth')&f.CryoSleep.eq('True')).to_numpy(),Earth_awake=(f.HomePlanet.eq('Earth')&f.CryoSleep.eq('False')).to_numpy())
        subsets={n:dict(rows=int(m.sum()),correct={key:int(((p[m]>=.5)==y[m]).sum()) for key,p in probabilities.items()},paired_to_raw={key:residual.relational.pair(p,raw,y,m) for key,p in probabilities.items() if key!='raw'},confusion={key:reporting.confusion(p,y,m) for key,p in probabilities.items()}) for n,m in masks.items()}
        verify_plan()
        for name,digest in evidence.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Evidence changed: '+name)
        np.savez_compressed(OUT/'oof.npz',dev_ids=dev,fold_id=fold_ids,passenger_ids=np.asarray(inputs['passengers'],dtype=str),**probabilities,**masks)
        result=dict(status='complete_fixed_adaptive_threefold',metrics={n:residual.relational.score(p,y) for n,p in probabilities.items()},paired_to_raw={n:residual.relational.pair(p,raw,y) for n,p in probabilities.items() if n!='raw'},distilled_vs_controls={n:residual.relational.pair(probabilities['distilled'],probabilities[n],y) for n in ['hard','uniform']},folds=folds,subsets=subsets,seconds=time.monotonic()-started,oof_path=str((OUT/'oof.npz').relative_to(ROOT)),oof_sha256=sha(OUT/'oof.npz'),protocol_sha256=sha(PLAN),limitations=plan['limitations'])
        write(REPORT/'evidence.json',evidence);write(REPORT/'result.json',result);print(json.dumps(result['metrics'],indent=2))
    except BaseException as error:
        if (OUT/'oof.npz').exists():(OUT/'oof.npz').rename(OUT/'oof.partial.npz')
        write(REPORT/'failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)

if __name__=='__main__':
    if sys.argv[1:]==['--freeze']:freeze()
    elif sys.argv[1:]:raise ValueError('Only --freeze or no arguments supported.')
    else:run()
