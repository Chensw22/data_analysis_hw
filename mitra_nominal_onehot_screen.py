"""One fixed nominal encoding screen with cached raw Mitra control, no export."""
import argparse
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import mitra_candidate as base
import exaone_validation_data as data

ROOT=Path(__file__).resolve().parent
NAME='v94_Mitra2_nominal_onehot'
REFERENCE='v71_Mitra2_zeroshot'
RUN='run_057'
PLAN='reports/top5_nominal_onehot_review.json'
PLAN_SHA='b1a02ca0aac320b8388f0e09763e58eccbe0d6341cedb31909189081493e39af'
RESOURCE='reports/mitra_nominal_onehot_resources'
ARTIFACT_RESOURCE='artifacts/probes/mitra_nominal_onehot'
BUDGET=900
OWN=['mitra_nominal_onehot_screen.py','test_mitra_nominal_onehot_screen.py','exaone_validation_data.py',
    'audit_top5_references.py','three_family_mean_screen.py','prior_adapt_mean_screen.py',data.PROOF,PLAN]
DISCLOSURE='Adaptively proposed using previously explored development data. This tests the whole fixed nominal encoding package, not a pure causal encoding comparison. Reused folds, same practical band within1pp; no independent or S evidence.'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def check_deadline(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Fixed900s nominal onehot screen deadline exhausted.')


def quota_guard(session):
    from quota_guard import latest_weekly,assess
    try:allowed=assess(latest_weekly(session),reserve=30)['can_start_more_work']
    except (OSError,ValueError,TypeError,KeyError):allowed=False
    if not allowed:raise RuntimeError('Fresh reserve30 guard did not allow this screen/fold.')


def assert_frozen(frozen):
    for name,digest in {**frozen['sources'],**frozen['evidence']}.items():
        if base.sha256(ROOT/name)!=digest:raise RuntimeError('Frozen nominal screen source/evidence changed: '+name)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=frozen['environment_sha256']:
        raise RuntimeError('Qualified Mitra environment changed.')
    base.verify_checkpoint()
    if frozen['checkpoint_sha256']!=base.WEIGHT_SHA:raise RuntimeError('Qualified checkpoint changed.')
    if 'resource_freeze' in frozen:
        importlib.import_module('mitra_nominal_onehot_probe').check_frozen(frozen['resource_freeze'])


def resource_contract():
    if base.sha256(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Fixed nominal encoding plan changed.')
    probe=importlib.import_module('mitra_nominal_onehot_probe')
    current,lock=probe.contract();saved=read(ROOT/RESOURCE/'freeze.json')
    if {k:v for k,v in current.items() if k!='source_commit'}!={k:v for k,v in saved.items() if k!='source_commit'}:
        raise RuntimeError('Completed nominal resource source/environment/recipe differs.')
    evidence={}
    for name in ['freeze.json','result.json','worker_result.json','requirements.lock']:
        public=ROOT/RESOURCE/name;local=ROOT/ARTIFACT_RESOURCE/name
        data.common.require_committed([str(public.relative_to(ROOT))])
        if base.sha256(public)!=base.sha256(local):raise RuntimeError('Nominal resource artifact/report differs.')
        data.remember(public,evidence);data.remember(local,evidence)
    result=read(ROOT/RESOURCE/'result.json');worker_result=read(ROOT/RESOURCE/'worker_result.json')
    if probe.qualify(worker_result) is not True:raise RuntimeError('Nominal resource coverage/reload qualification failed.')
    if (result.get('status')!='complete' or result.get('resource_qualification') is not True
            or result.get('worker_result_sha256')!=base.sha256(ROOT/RESOURCE/'worker_result.json')
            or result.get('cases')!=worker_result['cases'] or result.get('tiny')!=worker_result['tiny']):
        raise RuntimeError('Complete nominal-specific resource qualification required before targets.')
    probe.check_frozen(saved)
    if (ROOT/RESOURCE/'requirements.lock').read_text()!=lock:raise RuntimeError('Qualified environment lock differs.')
    for name in ['protocol.json','selection.json','screen_results.csv','requirements-mitra.lock.txt']:
        data.report_pair('run_038',name,evidence)
    protocol,files=data.common.verify_protocol(ROOT/'artifacts/run_038');evidence.update(files)
    if protocol.get('mitra_environment_sha256')!=hashlib.sha256(lock.encode()).hexdigest():
        raise RuntimeError('Reference and qualified Mitra environments differ.')
    frozen=dict(sources=dict(current['sources']),evidence=evidence,
        environment_sha256=current['environment_sha256'],checkpoint_sha256=current['checkpoint_sha256'],resource_freeze=saved)
    assert_frozen(frozen)
    return frozen,lock


def check_fit_metadata(metadata,ids,labels,seed,core):
    if (metadata.get('status')!='fit_complete' or metadata.get('candidate')!=NAME or metadata.get('settings')!=core.SETTINGS
            or metadata.get('seed')!=seed or metadata.get('training_rows')!=len(ids)
            or metadata.get('training_ids_sha256')!=base.ids_hash(ids) or metadata.get('labels_sha256')!=base.ids_hash(labels)
            or metadata.get('checkpoint_sha256')!=base.WEIGHT_SHA or metadata.get('optimizer_steps')!=0
            or metadata.get('full_prediction_support') is not True):
        raise RuntimeError('Actual fixed zero-update fit identity differs.')
    schema=metadata['schema']
    if (schema.get('original11_exact') is not True or not schema.get('columns') or not schema.get('active_columns')
            or not schema.get('vocabularies') or not schema.get('parent_by_column')
            or len(schema['active_columns'])!=schema.get('actual_retained_features')
            or len(schema.get('derived_mirror_mask',[]))!=schema.get('actual_retained_features')):
        raise RuntimeError('Persisted nominal schema/original-column/mirror evidence incomplete.')


def reference_inputs(frozen):
    inputs=data.load_data();splits,references,evidence=data.random_references(inputs)
    frozen['evidence'].update(inputs['evidence']);frozen['evidence'].update(evidence)
    # v81 and v44 authenticate original folds and provide context only, never the primary gate.
    data.common.require_committed([p for p in frozen['evidence'] if p.startswith('reports/')]);assert_frozen(frozen)
    return inputs,splits,references


def continuation(candidate,raw):
    folds=np.asarray(candidate['fold_accuracy'])+1e-12>=np.asarray(raw['fold_accuracy'])
    if folds.shape!=(3,):raise RuntimeError('Three complete fold metrics required.')
    checks=dict(pooled_accuracy_gain_at_least_001=candidate['accuracy']+1e-12>=raw['accuracy']+.001,
        pooled_logloss_not_worse=candidate['log_loss']<=raw['log_loss']+1e-12,two_nondecreasing_folds=int(folds.sum())>=2)
    return dict(passed=all(checks.values()),checks=checks,nonlower_folds=int(folds.sum()),
        S_supported=False,automatic_export=False,automatic_upload=False)


def paired(prob,reference,y,splits):
    def one(indices):
        a=prob[indices]>=.5;b=reference[indices]>=.5;t=y[indices]
        return dict(corrected=int(np.sum((a==t)&(b!=t))),regressed=int(np.sum((a!=t)&(b==t))),
            disagreements=int(np.sum(a!=b)),accuracy_delta=float(np.mean(a==t)-np.mean(b==t)))
    return dict(overall=one(np.arange(len(y))),folds=[one(b) for _,b in splits])


def snapshot():
    base.torch.mps.synchronize();return base.snapshot('mps')


def train_folds(out,report,inputs,splits,references,frozen,deadline,session):
    core=importlib.import_module('mitra_nominal_onehot')
    probe=importlib.import_module('mitra_nominal_onehot_probe')
    dev,y=inputs['dev'],inputs['y'];prob=np.full(len(dev),np.nan);fold_ids=np.full(len(dev),-1,dtype=np.int64);records=[]
    raw=references[REFERENCE]
    for fold,(a,b) in enumerate(splits):
        quota_guard(session);check_deadline(deadline);assert_frozen(frozen)
        model=None;seed=20261112+fold;tick=time.monotonic()
        try:
            model=core.fit_nominal_onehot(inputs['features'].iloc[dev[a]],y[a],dev[a],seed,device='mps')
            if len(model.support)!=len(a) or model.query_chunk!=2318 or not np.array_equal(model.labels,y[a]):
                raise RuntimeError('Complete original prediction support required.')
            predicted,forward_records=probe.predict(model,inputs['features'].iloc[dev[b]])
            predicted=np.asarray(predicted)
            if (predicted.shape!=(len(b),2) or not np.isfinite(predicted).all() or np.any((predicted<0)|(predicted>1))
                    or not np.allclose(predicted.sum(axis=1),1,rtol=0,atol=2e-6)):
                raise RuntimeError('Complete finite binary predictions required.')
            metadata=dict(model.metadata,prediction_ids_sha256=base.ids_hash(dev[b]),prediction_rows=len(b),forward_records=forward_records)
            check_fit_metadata(metadata,dev[a],y[a],seed,core)
            if (metadata.get('transformed_support_sha256')!=hashlib.sha256(model.support.tobytes()).hexdigest()
                    or model.support.shape[1]!=metadata['schema']['actual_retained_features']):
                raise RuntimeError('Actual transformed support differs from persisted schema/hash.')
            forward_resources=snapshot()
        finally:
            if model is not None:model.release()
            model=None;base.release_memory()
        check_deadline(deadline);assert_frozen(frozen)
        prob[b]=predicted[:,1];fold_ids[b]=fold
        local=[(np.array([],dtype=int),np.arange(len(b)))];metrics=data.metrics(prob[b],y[b],local)
        matched=data.metrics(raw[b],y[b],local);corrections=paired(prob[b],raw[b],y[b],local)
        record=dict(status='complete',candidate=NAME,fold=fold,seed=seed,train_rows=len(a),validation_rows=len(b),
            train_ids_sha256=base.ids_hash(dev[a]),validation_ids_sha256=base.ids_hash(dev[b]),
            metrics=metrics,primary_reference=matched,paired=corrections,metadata=metadata,
            seconds=time.monotonic()-tick,resources_after_forward=forward_resources,resources_after_release=snapshot())
        for folder in [out,report]:write(folder/f'fold{fold}.json',record)
        np.savez(out/f'fold{fold}_rows.npz',train_ids=dev[a],validation_ids=dev[b])
        np.savez(out/'partial_oof.npz',prob=prob,dev_ids=dev,fold_id=fold_ids)
        records.append(record)
        for folder in [out,report]:write(folder/'progress.json',dict(status='incomplete',completed_folds=len(records),partial_not_a_winner=True))
        print(json.dumps(dict(fold=fold,accuracy=metrics['accuracy'],log_loss=metrics['log_loss'],raw_accuracy=matched['accuracy'],
            raw_log_loss=matched['log_loss'],corrected=corrections['overall']['corrected'],regressed=corrections['overall']['regressed'],seconds=record['seconds'])),flush=True)
    if len(records)!=3 or np.any(fold_ids<0):raise RuntimeError('All3 folds required.')
    return dict(status='worker_complete',completed_folds=3,metrics=data.metrics(prob,y,splits),
        references={k:data.metrics(v,y,splits) for k,v in references.items()},folds=records)


def worker(out,report,inputs,splits,references,frozen,deadline,session):
    out,report=Path(out),Path(report);base.DEADLINE=deadline;result=dict(status='failed_inconclusive',release_complete=False)
    try:
        check_deadline(deadline);assert_frozen(frozen)
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0') or not base.torch.backends.mps.is_available():
            raise RuntimeError('MPS without implicit CPU fallback required.')
        base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)
        result=train_folds(out,report,inputs,splits,references,frozen,deadline,session)
    except BaseException as error:result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
    finally:
        try:
            base.release_memory();result['resources_after_release']=snapshot();check_deadline(deadline);result['release_complete']=True
        except BaseException as error:
            result.update(status='failed_inconclusive',cleanup_error_type=type(error).__name__,cleanup_error=str(error),release_complete=False)
            if 'error_type' not in result:result.update(error_type=type(error).__name__,error=str(error))
        for folder in [out,report]:write(folder/'worker_result.json',result)


def supervise(process,deadline):
    while process.is_alive() and time.monotonic()<deadline:process.join(timeout=min(5,max(0,deadline-time.monotonic())))
    if process.is_alive():
        process.kill();process.join(timeout=5)
        if process.is_alive():raise RuntimeError('Nominal worker could not be reaped.')
        raise TimeoutError('External supervisor killed nominal screen at fixed900s deadline.')
    if process.exitcode!=0:raise RuntimeError('Nominal worker exited abnormally: '+str(process.exitcode))


def publish(out,report,inputs,splits,references,child,frozen,deadline):
    if child.get('status')!='worker_complete' or child.get('completed_folds')!=3 or child.get('release_complete') is not True:
        raise RuntimeError('Complete3-fold released worker required. '+str(child.get('error_type',''))+': '+str(child.get('error','')))
    with np.load(out/'partial_oof.npz',allow_pickle=False) as cache:prob,dev,fold_ids=cache['prob'].copy(),cache['dev_ids'],cache['fold_id']
    if not np.array_equal(dev,inputs['dev']) or any(not np.array_equal(np.flatnonzero(fold_ids==k),b) for k,(_,b) in enumerate(splits)):
        raise RuntimeError('Completed worker OOF row/fold identity differs.')
    score=data.metrics(prob,inputs['y'],splits);ref={k:data.metrics(v,inputs['y'],splits) for k,v in references.items()}
    if score!=child['metrics'] or ref!=child['references']:raise RuntimeError('Worker cache metrics differ.')
    assessment=dict(status='complete',candidate=NAME,completed_folds=3,metrics=score,primary_reference=REFERENCE,
        references=ref,continuation=continuation(score,ref[REFERENCE]),paired=paired(prob,references[REFERENCE],inputs['y'],splits),
        disclosure=DISCLOSURE,old_holdout_targets_read=False,S_supported=False,automatic_export=False,automatic_upload=False,
        externally_supervised=True,budget_seconds=BUDGET)
    assert_frozen(frozen);check_deadline(deadline)
    cache=out/f'{NAME}_seed20261112.npz'
    np.savez(cache,prob=prob,fold_accuracy=score['fold_accuracy'],dev_ids=dev,fold_id=fold_ids,
        passenger_ids=np.asarray(inputs['passengers'],dtype=str),group_ids=np.asarray(inputs['groups'],dtype=str))
    table=pd.DataFrame([dict(version=NAME,accuracy=score['accuracy'],log_loss=score['log_loss'],fold_std=float(np.std(score['fold_accuracy'])),seconds=BUDGET-max(0,deadline-time.monotonic()))])
    for folder in [out,report]:
        table.to_csv(folder/'screen_results.csv',index=False);write(folder/'assessment.json',assessment)
    assessment['result_files_sha256']={p.name:base.sha256(p) for p in [cache,out/'screen_results.csv',out/'assessment.json',out/'worker_result.json',*[out/f'fold{k}.json' for k in range(3)],*[out/f'fold{k}_rows.npz' for k in range(3)]]}
    for folder in [out,report]:write(folder/'summary.json',assessment);write(folder/'progress.json',assessment)
    assert_frozen(frozen);check_deadline(deadline)
    return assessment


def run(session):
    out,report=ROOT/'artifacts'/RUN,ROOT/'reports'/RUN
    if out.exists() or report.exists():raise RuntimeError('No overwrite/resume of nominal screen evidence.')
    quota_guard(session);started=time.monotonic();deadline=started+BUDGET
    frozen,lock=resource_contract()  # Qualified sources/environment before any development targets.
    data.common.require_committed(OWN);frozen['sources'].update({name:base.sha256(ROOT/name) for name in OWN})
    inputs,splits,references=reference_inputs(frozen);check_deadline(deadline)
    out.mkdir(parents=True);report.mkdir(parents=True);process=None
    try:
        protocol=dict(candidate=NAME,seed=20261112,folds=3,development_rows=6954,threshold=.5,optimizer_steps=0,
            budget_seconds=BUDGET,externally_supervised=True,plan_sha256=PLAN_SHA,frozen=frozen,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            development_ids_sha256=base.ids_hash(inputs['dev']),development_labels_sha256=base.ids_hash(inputs['y']),
            fold_memberships=[dict(train_ids_sha256=base.ids_hash(inputs['dev'][a]),validation_ids_sha256=base.ids_hash(inputs['dev'][b])) for a,b in splits],
            primary_reference=REFERENCE,disclosure=DISCLOSURE,old_holdout_targets_read=False,automatic_export=False,automatic_upload=False)
        for folder in [out,report]:write(folder/'protocol.json',protocol);(folder/'requirements.lock').write_text(lock)
        for name in frozen['sources']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        process=multiprocessing.get_context('spawn').Process(target=worker,args=(str(out),str(report),inputs,splits,references,frozen,deadline,session))
        process.start();supervise(process,deadline);check_deadline(deadline)
        return publish(out,report,inputs,splits,references,read(out/'worker_result.json'),frozen,deadline)
    except BaseException as error:
        if process is not None and process.is_alive():process.kill();process.join(timeout=5)
        failed=dict(status='failed_inconclusive',candidate=NAME,error_type=type(error).__name__,error=str(error),
            seconds=time.monotonic()-started,budget_seconds=BUDGET,partial_not_a_winner=True,
            automatic_export=False,automatic_upload=False,S_supported=False)
        if (out/'worker_result.json').exists():
            try:
                child=read(out/'worker_result.json');failed.update(worker_error_type=child.get('error_type'),worker_error=child.get('error'),
                    worker_cleanup_error_type=child.get('cleanup_error_type'),worker_cleanup_error=child.get('cleanup_error'))
            except (OSError,ValueError):pass
        for folder in [out,report]:
            for name in [f'{NAME}_seed20261112.npz','screen_results.csv','assessment.json']:(folder/name).unlink(missing_ok=True)
            write(folder/'summary.json',failed);write(folder/'progress.json',failed)
        raise
    finally:
        if process is not None and not process.is_alive():process.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    args=parser.parse_args();run(args.session_file)
