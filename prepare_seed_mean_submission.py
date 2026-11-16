"""Conditional v88 export: one new full50 fit, old full50 reuse, sequential reloads; no upload."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path
import shutil
import signal
import time

import joblib
import numpy as np
import pandas as pd

import mitra_seed_repeat_candidate as candidate
import mitra_seed_mean_ensemble as ensemble
import prepare_prior_adapt_submission as prior
import prepare_group_episode_submission as group

ROOT=Path(__file__).resolve().parent
common,helpers,core,base=candidate.cached.common,prior.helpers,candidate.core,candidate.base
sha,read,write=group.sha,prior.read,prior.write
AUTH,BATCH=prior.AUTH,prior.BATCH
NAME,RUN,PROPOSAL=ensemble.NAME,'run_051','proposal_nov16_seed_mean'
MODULE='mitra_seed_mean_ensemble'
DECISION='reports/online_iteration_20261116/seed_mean_decision.json'
OLD_PROPOSAL='proposal_nov16_full_episode'
NEW_SEED=20261212
BUDGET=7200
OWN=['prepare_seed_mean_submission.py','test_prepare_seed_mean_submission.py',
     'mitra_seed_mean_ensemble.py','test_mitra_seed_mean_ensemble.py',
     'mitra_prior_adapt_ensemble.py','prepare_prior_adapt_submission.py',
     'prepare_group_episode_submission.py','prepare_exploratory_iteration.py',
     'prepare_top5_submission.py','kaggle_workflow.py','submit_iteration.py']


def required_evidence():
    files={candidate.REVIEW}
    for prefix in ['artifacts','reports']:
        files.update(f'{prefix}/{RUN}/{name}' for name in
            ['protocol.json','assessment.json','selection.json','screen_results.csv','progress.json','environment.lock.txt'])
        files.update(f'{prefix}/{RUN}/fold{fold}.json' for fold in range(3))
        files.update(str(p.relative_to(ROOT)) for p in (ROOT/prefix/RUN).glob(f'{candidate.SOLO}_seed*.json'))
    files.add(f'artifacts/{RUN}/folds.npz')
    files.update(f'artifacts/{RUN}/{name}_seed20261112.npz' for name in [candidate.OLD,candidate.SOLO,NAME])
    files.update(f'artifacts/{OLD_PROPOSAL}/{name}' for name in ['model.joblib','manifest.json',
        'authorization.json','selection.json','fit_metadata.json','reload_metadata.json',
        'preparation_protocol.json','environment.lock.txt','test_probabilities.csv'])
    files.update(f'reports/{OLD_PROPOSAL}/{name}' for name in ['manifest.json','authorization.json','selection.json'])
    return files


def require_passing_assessment(assessment):
    if (assessment.get('status')!='complete' or assessment.get('primary_candidate')!=NAME
            or assessment.get('continuation',{}).get('passed') is not True
            or assessment.get('sole_possible_future_export')!=NAME
            or assessment.get('solo_upload_allowed') is not False
            or [r.get('fold') for r in assessment.get('new_folds',[])]!=[0,1,2]):
        raise RuntimeError('All three complete folds and passing fixed v88 continuation required; no solo alternative.')
    metrics=assessment['metrics']
    actual=candidate.continuation(metrics[NAME],metrics[candidate.OLD],[0,1,2])
    if actual!=assessment['continuation'] or actual['passed'] is not True:
        raise RuntimeError('Frozen continuation does not recompute from completed evidence.')


def verify_screen():
    folder=ROOT/'artifacts'/RUN
    if (folder/'screen_failure.json').exists():raise RuntimeError('Failed screen cannot authorize export.')
    for name in ['protocol.json','assessment.json','selection.json','screen_results.csv','progress.json','environment.lock.txt']:
        if sha(folder/name)!=sha(ROOT/'reports'/RUN/name):raise RuntimeError('Screen reports differ: '+name)
    protocol,assessment=read(folder/'protocol.json'),read(folder/'assessment.json')
    require_passing_assessment(assessment)
    candidate.verify_review();candidate.assert_frozen(protocol)
    if (protocol.get('candidate')!=NAME or protocol.get('settings')!=candidate.SETTINGS
            or protocol.get('seed')!=20261112 or protocol.get('threshold')!=.5
            or protocol.get('development_rows')!=6954 or protocol.get('folds')!=3
            or protocol.get('configs',{}).get(NAME,{}).get('weights')!=[.5,.5]):
        raise RuntimeError('Frozen screen identity or two-seed recipe differs.')
    with np.load(folder/'folds.npz',allow_pickle=False) as saved:
        dev,fold_ids=saved['dev_ids'].copy(),saved['fold_id'].copy()
        if (int(saved['outer_split_seed'])!=20261112
                or not np.array_equal(saved['actual_training_seeds'],candidate.TRAINING_SEEDS)):
            raise RuntimeError('Frozen outer/training seeds changed.')
    if (dev.shape!=(6954,) or candidate.audit.row_hash(dev)!=candidate.audit.DEV_HASH
            or fold_ids.shape!=dev.shape or not np.array_equal(np.bincount(fold_ids,minlength=3),[2318]*3)):
        raise RuntimeError('Original development fold assignments required.')
    probabilities={}
    for name in [candidate.OLD,candidate.SOLO,NAME]:
        with np.load(folder/f'{name}_seed20261112.npz',allow_pickle=False) as saved:
            expected_seeds=list(range(20261112,20261115)) if name==candidate.OLD else candidate.TRAINING_SEEDS
            p=prior.common.probability_vector(saved['prob'],6954)
            if (not np.array_equal(saved['dev_ids'],dev) or not np.array_equal(saved['fold_id'],fold_ids)
                    or not np.array_equal(saved['actual_training_seeds'],expected_seeds)
                    or not np.allclose(saved['fold_accuracy'],assessment['metrics'][name]['fold_accuracy'],rtol=0,atol=1e-12)):
                raise RuntimeError('Complete OOF IDs, seeds or metrics differ.')
            probabilities[name]=p
    with np.load(ROOT/'artifacts/run_046/v81_Mitra_full_episode50_seed20261112.npz',allow_pickle=False) as old:
        if not np.array_equal(probabilities[candidate.OLD],old['prob']):raise RuntimeError('Original cached v81 changed.')
    if not np.array_equal(probabilities[NAME],.5*probabilities[candidate.OLD]+.5*probabilities[candidate.SOLO]):
        raise RuntimeError('Only the frozen equal mean may be exported.')
    for fold,row in enumerate(assessment['new_folds']):
        a,b=dev[fold_ids!=fold],dev[fold_ids==fold]
        old_log=read(ROOT/f'artifacts/run_046/{candidate.OLD}_seed{20261112+fold}_{candidate.audit.row_hash(a)[:12]}.json')
        if (old_log.get('training_ids_sha256')!=candidate.audit.row_hash(a)
                or old_log.get('prediction_ids_sha256')!=candidate.audit.row_hash(b)):
            raise RuntimeError('New fold membership differs from the original v81 fold.')
        filename=f'{candidate.SOLO}_seed{candidate.TRAINING_SEEDS[fold]}_{candidate.audit.row_hash(a)[:12]}.json'
        log=read(folder/filename)
        if (row!=read(folder/f'fold{fold}.json') or row!=read(ROOT/'reports'/RUN/f'fold{fold}.json')
                or sha(folder/filename)!=sha(ROOT/'reports'/RUN/filename)
                or row.get('actual_training_seed')!=candidate.TRAINING_SEEDS[fold]
                or row.get('train_ids_sha256')!=candidate.audit.row_hash(a)
                or row.get('query_ids_sha256')!=candidate.audit.row_hash(b)
                or log.get('prediction_ids_sha256')!=candidate.audit.row_hash(b)
                or log.get('prediction_rows')!=2318):
            raise RuntimeError('Complete fold log identity differs.')
        group.check_training_log(log,4636,candidate.TRAINING_SEEDS[fold],core.SETTINGS,candidate.audit.row_hash(a))
    return protocol,assessment


def decision_evidence():
    decision,auth=read(ROOT/DECISION),read(ROOT/AUTH)  # Missing root decision always stops first.
    expected=dict(batch_id=BATCH,batch_authorization_sha256=sha(ROOT/AUTH),module=MODULE,version=NAME,
        proposal=PROPOSAL,run=RUN,exploratory=True,threshold=.5,maximum_attempts=1,
        prior_A_eligible=None,prior_S_supported=False,new_fullfit_seed=NEW_SEED,weights=[.5,.5])
    if any(decision.get(k)!=v for k,v in expected.items()) or not decision.get('rationale'):
        raise RuntimeError('Exact separately committed v88 exploratory decision required.')
    if (auth.get('batch_id')!=BATCH or auth.get('competition')!='spaceship-titanic'
            or auth.get('maximum_upload_attempts')!=10 or auth.get('weekly_reserve_percent')!=30
            or auth.get('no_duplicate_or_blind_retry_upload') is not True):raise RuntimeError('Batch scope changed.')
    protocol_path=f'artifacts/{RUN}/protocol.json'
    if decision.get('source_protocol_sha256')!=sha(ROOT/protocol_path):raise RuntimeError('Decision protocol binding changed.')
    protocol=read(ROOT/protocol_path);evidence=decision.get('evidence_files',{})
    required=required_evidence()|set(protocol['sources'])|set(protocol['evidence_files'])
    if not isinstance(evidence,dict) or not required.issubset(evidence):raise RuntimeError('Incomplete screen/full-artifact provenance.')
    prior.verify_files(evidence);prior.verify_files(protocol['sources']);prior.verify_files(protocol['evidence_files'])
    sources=dict(protocol['sources']);sources.update({p:sha(ROOT/p) for p in OWN+[AUTH,DECISION]})
    committed=helpers.require_committed(set(sources)|{p for p in evidence if p.startswith('reports/')})
    _,assessment=verify_screen()
    if (not group.valid_accuracy(decision.get('screen_accuracy'))
            or abs(decision['screen_accuracy']-assessment['metrics'][NAME]['accuracy'])>1e-12):
        raise RuntimeError('Decision accuracy differs from fixed mean screen.')
    table=pd.read_csv(ROOT/'artifacts'/RUN/'screen_results.csv')
    row=table.loc[table.version.eq(NAME),'accuracy']
    if len(row)!=1 or abs(float(row.iloc[0])-decision['screen_accuracy'])>1e-12:raise RuntimeError('Screen CSV differs.')
    resource=group.verify_v81_resource(candidate)
    import submit_iteration
    if submit_iteration.count_attempts(ROOT/'reports',BATCH)>=10:raise RuntimeError('Authorized batch exhausted.')
    return dict(decision=decision,decision_path=DECISION,decision_sha256=sha(ROOT/DECISION),
        authorization_sha256=sha(ROOT/AUTH),source_protocol=protocol,sources=sources,
        evidence_files=evidence,committed_at=committed,old_fullfit_resource=resource)


def configure_runtime(deadline):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):raise RuntimeError('Disable MPS fallback.')
    _,lock=candidate.verify_runtime()
    prior.configure_runtime(deadline)  # Original environment, empty MPS allocator, threads4/cap.65.
    return lock


def read_full_data():
    # Called only after decision_evidence and a frozen preparation protocol.
    official=read(ROOT/'reports/official_data_verification.json')['files']
    for name in ['train.csv','test.csv']:
        if sha(ROOT/'data/raw'/name)!=official[name]['official_sha256']:raise RuntimeError('Official data changed.')
    train,test,frame=prior.load_attributes()
    y=base.binary_labels(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['Transported']).Transported.to_numpy(),8693)
    return train,test,frame,y


def fit_new_component(frame,y,path):
    return core.fit_full_episode(frame.iloc[:8693],y,np.arange(8693,dtype=np.int64),NEW_SEED,
        device='mps',log_path=path,max_steps=50)


def check_component(model,frame,y,seed):
    record=prior.check_component(model,frame,candidate.OLD)
    if model.metadata.get('seed')!=seed or not np.array_equal(model.labels,y):
        raise RuntimeError('Full component training seed or labels differ.')
    group.check_training_log(model.metadata,8693,seed,core.SETTINGS,base.ids_hash(np.arange(8693)))
    history=model.metadata['inner_history']
    if (not all(np.isfinite(r['validation_log_loss']) for r in history)
            or model.metadata['selected_steps']!=min(history,key=lambda r:(r['validation_log_loss'],r['step']))['step']):
        raise RuntimeError('Full fit did not select earliest minimum inner logloss.')
    return dict(**record,seed=seed,selected_steps=model.metadata['selected_steps'])


def release(model):
    if model is not None:model.release()
    base.release_memory()


def prepare(session_file):
    candidate.check_quota(session_file)  # Ephemeral path only; never persisted.
    evidence=decision_evidence()  # Before full labels, model load or any new training.
    out,report=ROOT/'artifacts'/PROPOSAL,ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():raise RuntimeError('No overwrite or implicit resume.')
    out.mkdir(parents=True);report.mkdir(parents=True)
    started=time.monotonic();models=[];model=restored=combined=None;handler=None
    try:
        def timeout(*_):raise TimeoutError('Fixed7200second seed-mean fit/export/reload budget exceeded.')
        handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
        environment=configure_runtime(started+BUDGET);(out/'environment.lock.txt').write_text(environment)
        env_sha=sha(out/'environment.lock.txt')
        for name in evidence['sources']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        write(out/'preparation_protocol.json',dict(**evidence,budget_seconds=BUDGET,environment_sha256=env_sha,
            sources_frozen_before_full_labels=True,full_labels_allowed_only_for_this_final_fit=True,
            new_fullfit_seed=NEW_SEED,weights=[.5,.5],old_component_retrained=False))
        _,test,frame,y=read_full_data();query=frame.iloc[8693:];predictions=[];logs=[]
        # Old artifact first: verify cache and release its MPS engine before the sole new fit.
        old=ROOT/'artifacts'/OLD_PROPOSAL
        cache=pd.read_csv(old/'test_probabilities.csv',dtype={'PassengerId':str})
        if len(cache)!=4277 or not np.array_equal(cache.PassengerId.astype(str),test.PassengerId.astype(str)):
            raise RuntimeError('Original cached test IDs differ.')
        expected=prior.common.probability_vector(cache.probability.to_numpy(),4277)
        model=joblib.load(old/'model.joblib');record=check_component(model,frame.iloc[:8693],y,20261112)
        model.device='mps'
        try:
            old_probability=ensemble.probabilities(model.predict_proba(query),4277)[:,1]
            difference=prior.common.check_reload(expected,old_probability)
        finally:release(model)
        record.update(version=candidate.OLD,source_model_sha256=sha(old/'model.joblib'),reload_rows=4277,
            reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,new_fit_performed=False)
        logs.append(record);predictions.append(old_probability.copy());model.device='cpu';models.append(model);model=None
        base.check_deadline()
        model=fit_new_component(frame,y,out/'new_fit_training.json')
        record=check_component(model,frame.iloc[:8693],y,NEW_SEED)
        first_new=ensemble.probabilities(model.predict_proba(query),4277)[:,1]
        model.metadata.update(prediction_ids_sha256=base.ids_hash(np.arange(8693,12970)),prediction_rows=4277,
            final_prediction_path='original_MitraPredictor')
        if any(t.device.type!='cpu' for t in model.state.values()):raise RuntimeError('New portable state must be CPU.')
        joblib.dump(model,out/'new_component.joblib');write(out/'new_fit_metadata.json',model.metadata)
        release(model);model=None;gc.collect();base.release_memory()
        restored=joblib.load(out/'new_component.joblib')
        if restored.device!='cpu' or restored._engine is not None:raise RuntimeError('New component reload must start CPU.')
        check_component(restored,frame.iloc[:8693],y,NEW_SEED);restored.device='mps'
        try:
            new_probability=ensemble.probabilities(restored.predict_proba(query),4277)[:,1]
            difference=prior.common.check_reload(first_new,new_probability)
        finally:release(restored)
        record.update(version=candidate.SOLO,model_sha256=sha(out/'new_component.joblib'),reload_rows=4277,
            reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,new_fit_performed=True)
        logs.append(record);predictions.append(new_probability.copy());restored.device='cpu';models.append(restored);restored=None
        write(out/'component_reload_log.json',logs)
        first=.5*predictions[0]+.5*predictions[1]
        combined=ensemble.SeedMeanPredictor(models,metadata=dict(version=NAME,components=logs,weights=[.5,.5],
            new_fullfit_seed=NEW_SEED,training_ids_sha256=base.ids_hash(np.arange(8693))))
        joblib.dump(combined,out/'model.joblib');release(combined);combined=None;models=[];gc.collect();base.release_memory()
        restored=joblib.load(out/'model.joblib')
        if type(restored) is not ensemble.SeedMeanPredictor or restored.device!='cpu':raise RuntimeError('Combined portable reload differs.')
        restored.device='mps';result=ensemble.probabilities(restored.predict_proba(query),4277)[:,1]
        difference=prior.common.check_reload(first,result);release(restored);restored=None;gc.collect();base.release_memory()
        base.check_deadline()
        pending=out/'pending';pending.mkdir()
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=first>=.5)).to_csv(pending/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=first,reload_probability=result)).to_csv(pending/'test_probabilities.csv',index=False)
        check=helpers.validate_csv(pending/'submission.csv')
        if helpers.duplicate_attempts(check['sha256']):raise RuntimeError('Identical CSV already attempted.')
        after=decision_evidence()
        if {k:v for k,v in after.items() if k!='committed_at'}!={k:v for k,v in evidence.items() if k!='committed_at'}:
            raise RuntimeError('Bound sources/evidence changed during export.')
        if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=env_sha:raise RuntimeError('Environment changed.')
        base.check_deadline();decision=evidence['decision']
        selection=dict(version=NAME,source_run=RUN,components=list(ensemble.COMPONENTS),weights=[.5,.5],
            stage='explicit_autonomous_exploration',A_eligible=None,S_supported=False,
            screen_accuracy=decision['screen_accuracy'],threshold=.5)
        bound=dict(status='bound_single_upload_authorization',competition='spaceship-titanic',batch_id=BATCH,
            source_authorization_sha256=evidence['authorization_sha256'],proposal=PROPOSAL,version=NAME,
            file_sha256=check['sha256'],model_sha256=sha(out/'model.joblib'),maximum_upload_attempts_for_this_file=1,
            desktop_reminder_required=True,decision_path=DECISION,decision_sha256=evidence['decision_sha256'],
            source_protocol_sha256=decision['source_protocol_sha256'],prepared_from_commit=evidence['committed_at'],
            no_duplicate_or_blind_retry_upload=True)
        manifest=dict(**check,status='complete',batch_id=BATCH,selection=selection,fit_rows=8693,
            new_fit_performed=True,new_fullfit_count=1,new_fullfit_seed=NEW_SEED,old_component_retrained=False,
            threshold=.5,export_reload_inference_verified=True,reload_rows=4277,
            reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,
            component_reload_checks=logs,model_sha256=bound['model_sha256'],authorization=bound,
            original_failed_gates_unchanged=True,source_hashes=evidence['sources'],environment_sha256=env_sha,
            provenance=dict(source_hashes=evidence['sources'],decision_path=DECISION,decision_sha256=evidence['decision_sha256'],
                evidence_files=evidence['evidence_files'],source_protocol_sha256=decision['source_protocol_sha256'],
                prepared_from_commit=evidence['committed_at']),seconds=time.monotonic()-started,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        for name in ['submission.csv','test_probabilities.csv']:(pending/name).replace(out/name)
        pending.rmdir()
        for name,value in [('selection.json',selection),('authorization.json',bound),('manifest.json',manifest)]:
            write(out/name,value);write(report/name,value)
        return manifest
    except BaseException as error:
        for directory in (out,report):
            for name in ['manifest.json','authorization.json','selection.json','submission.csv','test_probabilities.csv']:
                (directory/name).unlink(missing_ok=True)
            shutil.rmtree(directory/'pending',ignore_errors=True)
            write(directory/'failure.json',dict(status='failed_incomplete_not_upload_eligible',candidate=NAME,
                error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started))
        raise
    finally:
        for item in [model,restored,combined,*models]:
            if item is not None:item.release()
        base.release_memory();base.DEADLINE=None
        if handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-file',type=Path,required=True)
    prepare(parser.parse_args().session_file)
