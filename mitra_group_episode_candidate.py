"""Fixed group-only v85 screen against authenticated cached raw-Mitra folds.

No random-screen table, tier update, holdout scoring, export, or upload is done.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss

import audit_top5_references as audit
import iterate
import mitra_group_episode as core
base=core.base
ROOT=Path(__file__).resolve().parent
NAME='v85_Mitra_group_episode50'
OUT=ROOT/'artifacts/group_episode_20261116'
REPORT=ROOT/'reports/group_episode_20261116'
REVIEW=REPORT/'review.json'
REFERENCE=ROOT/'artifacts/top5_reference_audit/mitra'
RESOURCE=ROOT/'reports/mitra_full_episode_resources'
SCREEN_SETTINGS=dict(kind='group',outer_seed=20261112,folds=3,fit_seed='20261112+zero_based_fold',
    development_rows=6954,threshold=.5,budget_seconds=14400,device='mps',threads=4,
    mps_memory_fraction=.65,query_chunk=2318,
    reference='artifacts/top5_reference_audit/mitra/group_20261112.npz',
    old_holdout_targets_allowed=False,random_tier_update=False,
    resume=False,automatic_horizon_or_seed_extension=False)
RESOURCE_REUSE=dict(max_inner_support=3709,max_inner_query=928,max_outer_support=4636,
    max_outer_query=2318,extra_inner_support_rows_over_probe=1)
OWN_SOURCES=['mitra_group_episode.py','test_mitra_group_episode.py',
    'mitra_group_episode_candidate.py','test_mitra_group_episode_candidate.py',
    'audit_top5_references.py','test_audit_top5_references.py',
    'mitra_full_episode.py','test_mitra_full_episode.py','reports/official_data_verification.json']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    audit.json_write(path,value)


def review():
    value=json.loads(REVIEW.read_text())
    if (value.get('candidate')!=NAME or value.get('recipe')!=core.SETTINGS
            or value.get('screen')!=SCREEN_SETTINGS or value.get('weekly_reserve_percent')!=30
            or value.get('automatic_export') is not False
            or value.get('adaptive_reused_development') is not True
            or value.get('resource_reuse')!=RESOURCE_REUSE):
        raise RuntimeError('Prospective fixed group-episode review differs or is incomplete.')
    return value


def verify_reference_basis():
    history=ROOT/'artifacts/run_038/protocol.json'
    historical=json.loads(history.read_text())
    protocol=json.loads((REFERENCE/'protocol.json').read_text())
    summary=json.loads((REFERENCE/'summary.json').read_text())
    public=ROOT/'reports/top5_reference_audit/mitra/summary.json'
    if (summary.get('status')!='complete' or summary.get('reference')!='mitra'
            or summary.get('version')!='v71_Mitra2_zeroshot'
            or summary.get('historical_run_unchanged') is not True
            or summary.get('no_old_holdout_targets_used') is not True
            or audit.file_hash(public)!=audit.file_hash(REFERENCE/'summary.json')
            or protocol.get('reference')!='mitra' or protocol.get('version')!='v71_Mitra2_zeroshot'
            or protocol.get('historical_protocol_sha256')!=audit.file_hash(history)
            or protocol.get('historical_sources')!=historical['sources']
            or protocol.get('checkpoint_sha256')!=base.WEIGHT_SHA
            or protocol.get('runtime',{}).get('device')!='mps'
            or protocol.get('runtime',{}).get('query_chunk')!=2318):
        raise RuntimeError('Cached reference identity, completion or historical binding changed.')
    audit.check_historical_sources(historical)
    audit.check_historical_sources(dict(sources=protocol['audit_sources']))
    env={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    if protocol['environment']!=env:
        raise RuntimeError('Use the unchanged cached-reference Mitra environment.')
    entries=[r for r in summary['audits'] if r['kind']=='group' and r['seed']==20261112]
    if len(entries)!=1:raise RuntimeError('Exactly one cached group20261112 reference required.')
    entry=entries[0]
    if (entry['oof_path']!=SCREEN_SETTINGS['reference']
            or entry['oof_sha256']!=audit.file_hash(ROOT/entry['oof_path'])):
        raise RuntimeError('Cached OOF differs from the completed reference manifest.')
    base.verify_checkpoint(base.CHECKPOINT_DIR)
    return historical,protocol,summary,entry


def verify_resource():
    frozen=json.loads((RESOURCE/'freeze.json').read_text())
    result=json.loads((RESOURCE/'result.json').read_text())
    env_sha=hashlib.sha256(base.environment_lock().encode()).hexdigest()
    if (result.get('status')!='complete' or result.get('synthetic_only') is not True
            or result.get('accuracy_computed') is not False
            or frozen.get('checkpoint_sha256')!=base.WEIGHT_SHA
            or frozen.get('recipe')!=result.get('recipe')
            or frozen.get('environment_sha256')!=env_sha
            or audit.file_hash(RESOURCE/'requirements.lock')!=env_sha):
        raise RuntimeError('Complete matching full-episode resource evidence required.')
    recipe=result['recipe']
    required=dict(support_rows=1024,query_rows=256,steps=2,validation_support=3708,
        validation_query=928,final_support=8693,final_query=2318,query_chunk=2318,
        trainable_parameters=75670026,mps_memory_fraction=.65,weights_dtype='float32',
        mps_autocast='bfloat16',native_head=10,weight_decay=.3)
    if (any(recipe.get(k)!=v for k,v in required.items())
            or not 0<=result.get('reload_max_abs_difference',float('inf'))<=2e-6
            or [r.get('optimizer_state_steps') for r in result.get('updates',[])]!=[[1],[2]]
            or any(not all(r.get('updated_blocks',{}).values()) for r in result['updates'])
            or result.get('validation',{}).get('support_rows')!=3708
            or result.get('final_inference',{}).get('support_rows')!=8693):
        raise RuntimeError('Required full-parameter updates/full-support inference are not established.')
    audit.check_historical_sources(frozen)
    return frozen


def validate_cached_arrays(cached,fold_rows,dev,groups,passengers):
    dev=np.asarray(dev,dtype=np.int64);groups=np.asarray(groups).astype(str)
    passengers=np.asarray(passengers).astype(str)
    for key,actual in [('dev_ids',dev),('group_ids',groups),('passenger_ids',passengers)]:
        stored=np.asarray(cached[key])
        if key!='dev_ids':stored=stored.astype(str)
        if not np.array_equal(stored,actual):raise RuntimeError('Cached row identity differs: '+key)
    assignment=np.asarray(cached['fold_id'])
    probability=np.asarray(cached['prob'])
    if (assignment.shape!=dev.shape or not np.issubdtype(assignment.dtype,np.integer)
            or set(assignment.tolist())!={0,1,2} or len(fold_rows)!=3
            or probability.shape!=dev.shape or not np.isfinite(probability).all()
            or not np.all((probability>=0)&(probability<=1))):
        raise RuntimeError('Invalid complete cached OOF/fold assignment.')
    covered=np.zeros(len(dev),dtype=int);folds=[]
    for fold in range(3):
        a=np.flatnonzero(assignment!=fold);b=np.flatnonzero(assignment==fold)
        if (not len(a) or not len(b) or set(groups[a])&set(groups[b])
                or not np.array_equal(fold_rows[fold]['train_ids'],dev[a])
                or not np.array_equal(fold_rows[fold]['validation_ids'],dev[b])):
            raise RuntimeError('Cached fold IDs or group boundary differ.')
        covered[b]+=1;folds.append((a,b))
    if not np.all(covered==1):raise RuntimeError('Cached folds must cover every dev row once.')
    return probability.copy(),assignment.copy(),folds


def verify_data_files():
    record=json.loads((ROOT/'reports/official_data_verification.json').read_text())
    if record.get('competition')!='spaceship-titanic':
        raise RuntimeError('Official data verification names a different competition.')
    result={}
    for name in ['train.csv','test.csv']:
        path=ROOT/'data/raw'/name;expected=record.get('files',{}).get(name,{}).get('official_sha256')
        actual=audit.file_hash(path)
        if actual!=expected:raise RuntimeError('Official raw data SHA differs: '+name)
        result[str(path.relative_to(ROOT))]=actual
    return result


def load_inputs(reference):
    verify_data_files()  # Hash bytes before any target parsing; holdout targets are never parsed.
    _,protocol,summary,entry=reference
    # This helper reads only saved dev membership and dev target rows; no
    # full-target stratification, original1739holdout labels, or test targets.
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    identity=dict(development_ids_sha256=audit.row_hash(dev),group_ids_sha256=audit.string_hash(groups),
        passenger_ids_sha256=audit.string_hash(passengers))
    if len(dev)!=6954 or identity['development_ids_sha256']!=audit.DEV_HASH:
        raise RuntimeError('Expected original6954development membership.')
    for key,value in identity.items():
        if protocol.get(key)!=value:raise RuntimeError('Reference data identity differs: '+key)
    with np.load(REFERENCE/'group_20261112.npz',allow_pickle=False) as stored:
        cached={key:stored[key].copy() for key in stored.files}
    fold_rows=[];logs=[]
    for fold in range(3):
        with np.load(REFERENCE/f'group_20261112_fold{fold}_rows.npz',allow_pickle=False) as stored:
            fold_rows.append({key:stored[key].copy() for key in stored.files})
        logs.append(json.loads((REFERENCE/f'group_20261112_fold{fold}.json').read_text()))
    probability,assignment,folds=validate_cached_arrays(cached,fold_rows,dev,groups,passengers)
    expected=audit.audit_splits('group',20261112,y,groups)
    for fold,((a,b),(aa,bb),log) in enumerate(zip(folds,expected,logs)):
        training=log.get('training_metadata',{})
        if (not np.array_equal(a,aa) or not np.array_equal(b,bb)
                or log.get('kind')!='group' or log.get('split_seed')!=20261112 or log.get('fold')!=fold
                or log.get('model_fit_seed')!=20261112+fold or log.get('group_overlap_count')!=0
                or log.get('train_rows')!=len(a) or log.get('validation_rows')!=len(b)
                or log.get('train_ids_sha256')!=audit.row_hash(dev[a])
                or log.get('validation_ids_sha256')!=audit.row_hash(dev[b])
                or training.get('training_ids_sha256')!=audit.row_hash(dev[a])
                or training.get('training_rows')!=len(a) or training.get('seed')!=20261112+fold
                or training.get('checkpoint_sha256')!=base.WEIGHT_SHA
                or training.get('query_chunk')!=2318 or not training.get('full_prediction_support')
                or abs(log['accuracy']-accuracy_score(y[b],probability[b]>=.5))>1e-12):
            raise RuntimeError('Cached fold protocol, seed or metric differs.')
    if (abs(entry['accuracy']-accuracy_score(y,probability>=.5))>1e-12
            or abs(entry['log_loss']-log_loss(y,probability,labels=[0,1]))>1e-10):
        raise RuntimeError('Cached whole-OOF metrics differ from the completed manifest.')
    preflight=[]
    for fold,(a,b) in enumerate(folds):
        inner,valid=core.inner_split(y[a],groups[a],20261112+fold)
        row=dict(fold=fold,fit_seed=20261112+fold,outer_support=len(a),outer_query=len(b),
            inner_support=len(inner),inner_query=len(valid),inner_group_overlap=0)
        if any(row[k]>RESOURCE_REUSE['max_'+k] for k in ['inner_support','inner_query','outer_support','outer_query']):
            raise RuntimeError('Actual split exceeds explicitly reviewed resource extrapolation.')
        preflight.append(row)
    identity.update(targets_sha256=audit.row_hash(y),engineered_schema=list(matrices[False][0].columns),
        engineered_values_sha256=hashlib.sha256(pd.util.hash_pandas_object(
            matrices[False][0],index=True).to_numpy().tobytes()).hexdigest())
    return matrices,dev,y,groups,passengers,probability,assignment,folds,preflight,identity


def source_and_history(historical,resource):
    sources={**historical['sources'],**resource['sources']}
    for relative in [*OWN_SOURCES,str(REVIEW.relative_to(ROOT)),
            *[str((RESOURCE/name).relative_to(ROOT)) for name in ['freeze.json','result.json','requirements.lock']]]:
        sources[relative]=audit.file_hash(ROOT/relative)
    files=[REFERENCE/name for name in ['summary.json','protocol.json','group_20261112.npz']]
    files += [REFERENCE/f'group_20261112_fold{fold}{suffix}' for fold in range(3) for suffix in ['.json','_rows.npz']]
    files += [ROOT/'artifacts/run_038/protocol.json',ROOT/'artifacts/run_002/split_indices.npz',
        ROOT/'reports/top5_reference_audit/mitra/summary.json',
        ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv']
    return sources,{str(p.relative_to(ROOT)):audit.file_hash(p) for p in files}


def require_committed(sources):
    tracked=subprocess.check_output(['git','ls-files','-z','--',*sources],cwd=ROOT).decode()
    if set(filter(None,tracked.split('\0')))!=set(sources):
        raise RuntimeError('Review and all experiment sources must be committed before training.')
    if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=ROOT).strip():
        raise RuntimeError('Reviewed sources differ from their committed versions.')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def assert_frozen(record):
    for group in ['sources','historical_files']:
        for relative,expected in record[group].items():
            if audit.file_hash(ROOT/relative)!=expected:raise RuntimeError('Frozen evidence changed: '+relative)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=record['environment_sha256']:
        raise RuntimeError('Frozen environment changed during group screening.')
    review();base.verify_checkpoint(base.CHECKPOINT_DIR)


def run():
    if OUT.exists():raise RuntimeError('Group screen output exists; no resume or overwrite.')
    for name in ['protocol.json','progress.json','summary.json','failure.json','group_metrics.csv']:
        if (REPORT/name).exists():raise RuntimeError('Group report output already exists: '+name)
    if list(REPORT.glob('fold*.json')):raise RuntimeError('Existing fold report/logs; refuse overwrite.')
    review();reference=verify_reference_basis();resource=verify_resource()
    sources,history=source_and_history(reference[0],resource);commit=require_committed(sources)
    values=load_inputs(reference)
    matrices,dev,y,groups,passengers,raw,assignment,folds,preflight,identity=values
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('Disable automatic MPS fallback.')
    if not base.torch.backends.mps.is_available():raise RuntimeError('MPS required; no CPU fallback.')
    base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    for relative in sources:
        target=OUT/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    lock=base.environment_lock();(OUT/'environment.lock.txt').write_text(lock)
    record=dict(status='frozen_before_group_fits',candidate=NAME,source_commit=commit,
        sources=sources,historical_files=history,recipe=core.SETTINGS,screen=SCREEN_SETTINGS,
        resource_reuse=RESOURCE_REUSE,preflight=preflight,data=identity,
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        reference='rawMitra_cached_same_group_fold_same_fit_seed',automatic_eligibility=False,
        disclosure='Reused development group folds; no random-OOF superiority comparison or independent confirmation.',
        resource_disclosure='Twoinner supports3709 exceed synthetic3708 by1row; explicitly accepted extrapolation, not an identical probe. Fail closed on OOM; no split/context reduction.')
    write(OUT/'protocol.json',record);write(REPORT/'protocol.json',record)
    np.savez(OUT/'folds.npz',dev_ids=dev,group_ids=groups.astype(str),passenger_ids=passengers.astype(str),fold_id=assignment)
    started=time.monotonic();base.DEADLINE=started+14400
    def timeout(*_):raise TimeoutError('Fixed14400second group-screen budget exceeded.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,14400)
    probability=np.full(len(y),np.nan);covered=np.zeros(len(y),dtype=bool);rows=[];model=None
    try:
        for fold,(a,b) in enumerate(folds):
            assert_frozen(record);base.check_deadline();tick=time.monotonic()
            model=core.fit_group_episode(matrices[False][0].iloc[dev[a]],y[a],dev[a],groups[a],
                20261112+fold,'mps',OUT/f'fold{fold}_training.json')
            predicted=model.predict_proba(matrices[False][0].iloc[dev[b]])[:,1]
            if (predicted.shape!=(len(b),) or not np.isfinite(predicted).all()
                    or not np.all((predicted>=0)&(predicted<=1))):raise RuntimeError('Invalid complete fold prediction.')
            base.check_deadline();probability[b]=predicted;covered[b]=True
            row=dict(fold=fold,fit_seed=20261112+fold,train_rows=len(a),query_rows=len(b),
                train_ids_sha256=audit.row_hash(dev[a]),query_ids_sha256=audit.row_hash(dev[b]),
                group_overlap_count=0,accuracy=float(accuracy_score(y[b],predicted>=.5)),
                reference_accuracy=float(accuracy_score(y[b],raw[b]>=.5)),
                log_loss=float(log_loss(y[b],predicted,labels=[0,1])),
                reference_log_loss=float(log_loss(y[b],raw[b],labels=[0,1])),
                selected_steps=model.metadata['selected_steps'],seconds=time.monotonic()-tick)
            row['accuracy_gain']=row['accuracy']-row['reference_accuracy'];rows.append(row)
            write(OUT/f'fold{fold}.json',row);write(REPORT/f'fold{fold}.json',row)
            model.release();model=None;base.release_memory()
            np.savez(OUT/'partial_oof.npz',prob=probability,covered=covered,dev_ids=dev,fold_id=assignment)
            progress=dict(status='incomplete',completed_folds=rows,whole_score_available=False)
            write(OUT/'progress.json',progress);write(REPORT/'progress.json',progress)
            print(json.dumps(row),flush=True)
        assert_frozen(record)
        if not np.all(covered) or not np.isfinite(probability).all() or len(rows)!=3:
            raise RuntimeError('All fixed folds and rows required before whole score.')
        result=dict(status='complete',candidate=NAME,kind='group',seed=20261112,
            development_rows=len(dev),folds=rows,accuracy=float(accuracy_score(y,probability>=.5)),
            reference_accuracy=float(accuracy_score(y,raw>=.5)),log_loss=float(log_loss(y,probability,labels=[0,1])),
            reference_log_loss=float(log_loss(y,raw,labels=[0,1])),seconds=time.monotonic()-started,
            automatic_eligibility=False,protocol_sha256=audit.file_hash(OUT/'protocol.json'),
            no_old_holdout_targets_used=True,random_tier_update=False,
            interpretation='Same-group paired development result only; no stable-superiority or upload qualification claimed.')
        result['accuracy_gain']=result['accuracy']-result['reference_accuracy']
        np.savez(OUT/'group_20261112.npz',prob=probability,reference_prob=raw,dev_ids=dev,
            fold_id=assignment,group_ids=groups.astype(str),passenger_ids=passengers.astype(str))
        result['oof_sha256']=audit.file_hash(OUT/'group_20261112.npz')
        pd.DataFrame(rows).to_csv(REPORT/'group_metrics.csv',index=False)
        write(OUT/'summary.json',result);write(REPORT/'summary.json',result)
        progress=dict(status='complete',completed_folds=rows,whole_score_available=True)
        write(OUT/'progress.json',progress);write(REPORT/'progress.json',progress)
        (OUT/'partial_oof.npz').unlink()
        print(json.dumps(result),flush=True)
        return result
    except BaseException as error:
        failure=dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),
            completed_folds=rows,whole_score_available=False,seconds=time.monotonic()-started)
        for path in [OUT/'summary.json',REPORT/'summary.json',REPORT/'group_metrics.csv']:
            path.unlink(missing_ok=True)
        write(OUT/'failure.json',failure);write(REPORT/'failure.json',failure)
        write(OUT/'progress.json',failure);write(REPORT/'progress.json',failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);base.DEADLINE=None
        if model is not None:model.release()
        model=None;base.release_memory()
        for path in OUT.glob('fold*_training.json'):
            shutil.copy2(path,REPORT/path.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['screen'])
    parser.parse_args();run()
