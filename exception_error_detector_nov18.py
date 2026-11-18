"""Fixed error-probability detector using strictly nested rawMitra training predictions.

No new base fits, threshold search, hard-case training selection, or upload.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def ensure_openmp():
    # Same existing-environment loader entry as nonlinear_error_corrector, before numeric imports.
    if any(n in sys.modules for n in ['numpy','scipy','sklearn','lightgbm','torch']):
        raise RuntimeError('Runtime entry must precede numeric imports.')
    if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):
        raise RuntimeError('Duplicate-runtime tolerance must not be enabled.')
    env={**os.environ,'KMP_DUPLICATE_LIB_OK':'FALSE'}
    if sys.platform=='darwin':
        spec=importlib.util.find_spec('sklearn');library=Path(spec.origin).parent/'.dylibs'
        if not (library/'libomp.dylib').is_file() or not library.resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError('Existing current-environment sklearn libomp required.')
        current=os.environ.get('DYLD_FALLBACK_LIBRARY_PATH','')
        if str(library) not in current.split(':'):
            env['DYLD_FALLBACK_LIBRARY_PATH']=str(library)+((':'+current) if current else '')
    if any(os.environ.get(k)!=v for k,v in env.items()):os.execve(sys.executable,[sys.executable,*sys.argv],env)


if __name__=='__main__':ensure_openmp()

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import nonlinear_error_corrector as residual

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'reports/error_targeted_20261118/exception_protocol.json'
REPORT=PLAN.parent
OUT=ROOT/'artifacts/error_targeted_20261118/exception_detector'
EXTRA=['raw_probability','raw_margin','raw_class']
COLUMNS=residual.COLUMNS+EXTRA
PARAMS={**residual.PARAMS,'boost_from_average':True}
ROUNDS=60
sha=residual.sha
write=residual.write


def check_protocol():
    plan=json.loads(PLAN.read_text())
    if (plan['parameters']!=PARAMS or plan['rounds']!=ROUNDS or plan['columns']!=COLUMNS
            or plan['threshold']!=.5 or plan['seed_policy']!='20261112+outer_fold'):
        raise RuntimeError('Fixed recipe changed.')
    for name,digest in {**plan['sources_sha256'],**plan['historical_evidence_sha256']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen source/evidence changed: '+name)
    return plan


def append_raw(frame,p):
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(len(frame),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid base probabilities.')
    out=frame.copy();out[EXTRA[0]]=p;out[EXTRA[1]]=np.abs(p-.5);out[EXTRA[2]]=(p>=.5).astype(float)
    if list(out.columns)!=COLUMNS:raise RuntimeError('Expected original30 plus fixed3.')
    return out


def correct(raw,q):
    raw=np.asarray(raw);q=np.asarray(q,dtype=np.float64)
    if q.shape!=raw.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid error probability.')
    c=raw>=.5
    p=c*(1-q)+(~c)*q
    labels=np.where(q>.5,~c,c)
    # Preserve original negative class at exact error-probability tie with the adjacent float.
    p[(q==.5)&(~c)]=np.nextafter(.5,0.)
    np.testing.assert_array_equal(p>=.5,labels)
    return p,labels


def scores(p,y):
    return dict(accuracy=float(np.mean((p>=.5)==y)),log_loss=float(log_loss(y,p,labels=[0,1])))


def comparison(p,ref,y,mask=None):
    return residual.relational.pair(p,ref,y,mask)


def confusion(p,y,mask):
    truth=y[mask].astype(bool);pred=p[mask]>=.5
    return dict(rows=int(mask.sum()),tn=int((~truth&~pred).sum()),fp=int((~truth&pred).sum()),
                fn=int((truth&~pred).sum()),tp=int((truth&pred).sum()))


def old_residual(dev,fold_ids,raw,evidence):
    path=ROOT/'reports/error_targeted_20261117/residual_runtime_fixed_result.json'
    result=json.loads(path.read_text());npz=ROOT/result['oof_path']
    if result['status']!='complete' or sha(npz)!=result['oof_sha256']:raise RuntimeError('Residual reference incomplete.')
    with np.load(npz,allow_pickle=False) as z:
        for key,expected in [('dev_ids',dev),('fold_id',fold_ids),('raw',raw)]:np.testing.assert_array_equal(z[key],expected)
        p=z['nonlinear'].copy()
    evidence.update({str(pth.relative_to(ROOT)):sha(pth) for pth in [path,npz]})
    return p


def reporting_masks(dev,folds,y,inputs,bases,evidence):
    # Called only after all three detectors and save/load checks are complete.
    path=ROOT/'reports/error_rescue_20261117/catalog.json';catalog=json.loads(path.read_text())
    if catalog['status']!='complete_descriptive_only':raise RuntimeError('Catalog incomplete.')
    for name,digest in catalog['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Catalog provenance changed: '+name)
    evidence.update(catalog['evidence_sha256'])
    rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
    for col,expected in [('row_id',dev),('fold',folds),('y',y),('PassengerId',inputs['passengers'])]:
        np.testing.assert_array_equal(rows[col].to_numpy(),expected)
    names=catalog['primary_representatives'];allcols=[c for c in rows if c.startswith('correct_')]
    for col in allcols:
        if not rows[col].isin([True,False]).all():raise RuntimeError('Nonboolean catalog correctness.')
    masks={'common_wrong914':(rows.family_wrong_count==5).to_numpy(),
           'common_correct5263':(rows.family_wrong_count==0).to_numpy(),
           'primary_common_wrong616':~rows[['correct_'+n for n in names]].to_numpy(dtype=bool).any(axis=1),
           'all_recipes_common_wrong546':~rows[allcols].to_numpy(dtype=bool).any(axis=1)}
    family_wrong=np.column_stack([(bases[n]>=.5)!=y for n in residual.overlap.FAMILIES])
    np.testing.assert_array_equal(masks['common_wrong914'],family_wrong.all(axis=1))
    np.testing.assert_array_equal(masks['common_correct5263'],(~family_wrong).all(axis=1))
    for name,count in [('common_wrong914',914),('common_correct5263',5263),('primary_common_wrong616',616),('all_recipes_common_wrong546',546)]:
        if masks[name].sum()!=count:raise RuntimeError('Historical count changed: '+name)
    return masks


def run():
    if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Explicit FALSE runtime required.')
    plan=check_protocol()
    if OUT.exists() or (REPORT/'exception_result.json').exists() or (REPORT/'exception_failure.json').exists():
        raise FileExistsError('Fresh experiment only; no overwrite/resume.')
    OUT.mkdir(parents=True);started=time.monotonic()
    try:
        import lightgbm as lgb
        inputs,splits,fold_ids,bases,evidence=residual.overlap.load_authenticated()
        dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra'];x=residual.features()
        evidence.update(plan['historical_evidence_sha256'])
        frozen=residual.relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        residual.relational.nested.verify_hashes(frozen['sources'])
        q_all=np.full(len(dev),np.nan);candidate=q_all.copy();folds=[]
        for outer,(a,b) in enumerate(splits):
            inner=residual.relational.load_inner(outer,a,b,dev,y,evidence)
            train=append_raw(x.iloc[dev[a]],inner);query=append_raw(x.iloc[dev[b]],raw[b])
            train,query,vocab=residual.categorical_frames(train,query)
            error=((inner>=.5)!=y[a]).astype(np.int32)
            # All outertrain rows; normal empirical error prior, no base init_score or balancing.
            dataset=lgb.Dataset(train,label=error,categorical_feature=residual.CATS,free_raw_data=True)
            model=lgb.train({**PARAMS,'seed':20261112+outer},dataset,num_boost_round=ROUNDS)
            q=np.asarray(model.predict(query));p,labels=correct(raw[b],q)
            booster=OUT/f'fold{outer}_booster.txt';schema=OUT/f'fold{outer}_schema.json'
            model.save_model(str(booster));write(schema,dict(columns=COLUMNS,categories=vocab))
            saved=json.loads(schema.read_text())
            if saved['columns']!=COLUMNS:raise RuntimeError('Saved schema differs.')
            _,query_reload,_=residual.categorical_frames(append_raw(x.iloc[dev[a]],inner),
                append_raw(x.iloc[dev[b]],raw[b]),saved['categories'])
            restored=lgb.Booster(model_file=str(booster));q_reload=np.asarray(restored.predict(query_reload))
            np.testing.assert_array_equal(q,q_reload);np.testing.assert_array_equal(p,correct(raw[b],q_reload)[0])
            q_all[b]=q;candidate[b]=p
            np.savez_compressed(OUT/f'fold{outer}_rows.npz',train_ids=dev[a],query_ids=dev[b],inner_raw=inner,
                error_target=error,error_probability=q,probability=p,predicted_class=labels)
            row=dict(fold=outer,training_rows=len(a),query_rows=len(b),training_error_rate=float(error.mean()),
                seed=20261112+outer,requested_rounds=ROUNDS,actual_iterations=model.current_iteration(),
                exact_q_half_count=int((q==.5).sum()),booster_sha256=sha(booster),schema_sha256=sha(schema),
                rows_sha256=sha(OUT/f'fold{outer}_rows.npz'),reload_max_abs_difference=0.)
            folds.append(row);write(REPORT/f'exception_fold{outer}.json',row)
            print(f'Completed error detector fold {outer+1}/3',flush=True)
            del model,restored,dataset
        if len(folds)!=3 or not np.isfinite(candidate).all():raise RuntimeError('Incomplete threefold OOF.')
        # No reporting-label mask, prior residual prediction or subgroup enters any fit.
        check_protocol();old=old_residual(dev,fold_ids,raw,evidence)
        masks=reporting_masks(dev,fold_ids,y,inputs,bases,evidence)
        probabilities={'raw':raw,'nonlinear_residual':old,'exception_detector':candidate}
        for row,(_,b) in zip(folds,splits):
            row.update(metrics={k:scores(p[b],y[b]) for k,p in probabilities.items()},
                paired={k:comparison(candidate[b],p[b],y[b]) for k,p in [('raw',raw),('nonlinear_residual',old)]})
            write(REPORT/f"exception_fold{row['fold']}.json",row)
        subsets={name:dict(rows=int(mask.sum()),correct={k:int(((p[mask]>=.5)==y[mask]).sum()) for k,p in probabilities.items()},
            paired={k:comparison(candidate,p,y,mask) for k,p in [('raw',raw),('nonlinear_residual',old)]}) for name,mask in masks.items()}
        attr=x.iloc[dev];earth=attr.HomePlanet.astype(str).eq('Earth').to_numpy()
        groups={state:earth&attr.CryoSleep.astype(str).eq(value).to_numpy() for state,value in [('Earth_sleep','True'),('Earth_awake','False')]}
        group_reports={name:{k:confusion(p,y,mask) for k,p in probabilities.items()} for name,mask in groups.items()}
        for row,(_,b) in zip(folds,splits):
            row['earth_confusion']={name:{k:confusion(p[b],y[b],mask[b]) for k,p in probabilities.items()} for name,mask in groups.items()}
            write(REPORT/f"exception_fold{row['fold']}.json",row)
        path=OUT/'oof.npz';np.savez_compressed(path,dev_ids=dev,fold_id=fold_ids,passenger_ids=inputs['passengers'],
            error_probability=q_all,predicted_class=candidate>=.5,**probabilities,**masks)
        check_protocol()
        for name,digest in evidence.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Input evidence changed: '+name)
        if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Runtime changed.')
        result=dict(status='complete',protocol_sha256=sha(PLAN),folds=folds,metrics={k:scores(p,y) for k,p in probabilities.items()},
            paired={k:comparison(candidate,p,y) for k,p in [('raw',raw),('nonlinear_residual',old)]},
            historical_subsets=subsets,earth_confusion=group_reports,oof_path=str(path.relative_to(ROOT)),oof_sha256=sha(path),
            seconds=time.monotonic()-started,base_models_retrained=False,old_holdout_labels_read=False,automatic_upload=False,
            disclosure=plan['limitations'])
        write(REPORT/'exception_evidence.json',evidence);write(REPORT/'exception_result.json',result)
    except BaseException as error:
        path=OUT/'oof.npz'
        if path.exists():path.rename(OUT/'oof.partial.npz')
        write(REPORT/'exception_failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    if sys.argv[1:]:raise ValueError('No CLI tuning parameters supported.')
    run()
