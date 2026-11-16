"""Revalidate two existing full-data artifacts and export their fixed equal mean; never fit/upload."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import shutil
import signal
import time

import joblib
import numpy as np
import pandas as pd

import prepare_exploratory_iteration as helpers
import prepare_top5_submission as common
import mitra_prior_adapt_ensemble as ensemble

ROOT = Path(__file__).resolve().parent
AUTH, BATCH = helpers.AUTH, helpers.BATCH
NAME, RUN, PROPOSAL = ensemble.NAME, 'run_050', 'proposal_nov16_prior_adapt'
DECISION = 'reports/online_iteration_20261116/prior_adapt_decision.json'
PLAN = 'reports/online_iteration_20261116/fixed_prior_adapt_plan.json'
COMPONENTS = [('v71_Mitra2_zeroshot','run_038','proposal_batch_mitra2'),
              ('v81_Mitra_full_episode50','run_046','proposal_nov16_full_episode')]
OWN = ['prepare_prior_adapt_submission.py','test_prepare_prior_adapt_submission.py',
       'mitra_prior_adapt_ensemble.py','test_mitra_prior_adapt_ensemble.py',
       'prepare_exploratory_iteration.py','prepare_top5_submission.py','kaggle_workflow.py',
       'prior_adapt_mean_screen.py','test_prior_adapt_mean_screen.py','three_family_mean_screen.py',
       'audit_top5_references.py',PLAN]
sha, read, write = common.sha, common.read, common.write


def verify_files(mapping):
    for name, digest in mapping.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not (ROOT/path).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe evidence path.')
        if sha(ROOT/path) != digest:
            raise RuntimeError('Bound source/evidence changed: '+name)


def required_evidence():
    result = {PLAN}
    for directory in ['artifacts','reports']:
        result.update(f'{directory}/{RUN}/{name}' for name in ['protocol.json','screen_results.csv','assessment.json','selection.json'])
    for _, run, proposal in COMPONENTS:
        result.add(f'artifacts/{run}/protocol.json')
        directory = ROOT/'artifacts'/proposal
        result.update(str((directory/name).relative_to(ROOT)) for name in ['manifest.json','model.joblib','test_probabilities.csv','selection.json'])
        result.update(str(path.relative_to(ROOT)) for path in directory.iterdir()
                      if path.name in ['fit_metadata.json','reload_metadata.json','environment.lock.txt','preparation_protocol.json']
                      or path.name.endswith('_training.json'))
    result.update(['artifacts/run_038/requirements-mitra.lock.txt', 'artifacts/run_046/full_episode_environment.lock.txt'])
    return result


def decision_evidence():
    decision, auth = read(ROOT/DECISION), read(ROOT/AUTH)
    expected = dict(batch_id=BATCH, batch_authorization_sha256=sha(ROOT/AUTH), module='mitra_prior_adapt_ensemble',
        version=NAME, proposal=PROPOSAL, run=RUN, exploratory=True, threshold=.5,
        maximum_attempts=1, prior_A_eligible=None, prior_S_supported=False)
    if any(decision.get(k) != v for k,v in expected.items()) or not decision.get('rationale'):
        raise RuntimeError('Exact fixed exploratory decision required.')
    if (auth.get('batch_id')!=BATCH or auth.get('competition')!='spaceship-titanic'
            or auth.get('maximum_upload_attempts')!=10 or auth.get('weekly_reserve_percent')!=30
            or auth.get('no_duplicate_or_blind_retry_upload') is not True):
        raise RuntimeError('Batch authorization differs.')
    protocol_path = f'artifacts/{RUN}/protocol.json'
    if decision.get('source_protocol_sha256') != sha(ROOT/protocol_path):
        raise RuntimeError('Screen protocol differs.')
    protocol = read(ROOT/protocol_path)
    if (protocol.get('candidate')!=NAME or protocol.get('threshold')!=.5
            or [(r['version'],r['weight']) for r in protocol.get('components',[])]!=[(v,.5) for v,_,_ in COMPONENTS]):
        raise RuntimeError('Screen identity or fixed weights differ.')
    evidence = decision.get('evidence_files', {})
    required = required_evidence() | set(protocol['source_sha256'])
    if not isinstance(evidence,dict) or not required.issubset(evidence):
        raise RuntimeError('Decision lacks full screen/component provenance.')
    verify_files(evidence); verify_files(protocol['source_sha256'])
    for name in ['protocol.json','screen_results.csv','assessment.json','selection.json']:
        if sha(ROOT/f'artifacts/{RUN}/{name}') != sha(ROOT/f'reports/{RUN}/{name}'):
            raise RuntimeError('Screen artifacts differ from committed reports.')
    assessment = read(ROOT/f'artifacts/{RUN}/assessment.json')
    if assessment.get('status')!='complete' or assessment.get('exploratory_continuation',{}).get('passed') is not True:
        raise RuntimeError('Completed passing exploratory screen required.')
    table = pd.read_csv(ROOT/f'artifacts/{RUN}/screen_results.csv')
    row = table.loc[table.version.eq(NAME),'accuracy']
    accuracy = decision.get('screen_accuracy')
    if (type(accuracy) not in [int,float] or not np.isfinite(accuracy) or not 0<=accuracy<=1
            or len(row)!=1 or abs(float(row.iloc[0])-accuracy)>1e-12):
        raise RuntimeError('Decision accuracy differs from completed screen.')
    sources, components = {}, []
    for version,run,proposal in COMPONENTS:
        original = read(ROOT/f'artifacts/{run}/protocol.json')
        verify_files(original['sources'])
        for name,digest in original['sources'].items():
            if name in sources and sources[name]!=digest:
                raise RuntimeError('Component source versions conflict.')
            sources[name]=digest
        manifest = read(ROOT/f'artifacts/{proposal}/manifest.json')
        if (manifest.get('fit_rows')!=8693 or manifest.get('rows')!=4277 or manifest.get('threshold')!=.5
                or manifest.get('selection',{}).get('version')!=version
                or manifest.get('model_sha256')!=sha(ROOT/f'artifacts/{proposal}/model.joblib')
                or manifest.get('source_protocol_sha256',manifest.get('provenance',{}).get('source_protocol_sha256'))!=sha(ROOT/f'artifacts/{run}/protocol.json')):
            raise RuntimeError('Component manifest/model identity differs.')
        components.append(dict(version=version,run=run,proposal=proposal,
            historical_export_reload_inference_verified=manifest.get('export_reload_inference_verified'),
            model_sha256=manifest['model_sha256']))
    sources.update({path:sha(ROOT/path) for path in OWN+[AUTH,DECISION]})
    committed_reports = [p for p in evidence if p.startswith('reports/')]
    commit = helpers.require_committed(set(sources)|set(committed_reports))
    return dict(decision=decision, decision_path=DECISION, decision_sha256=sha(ROOT/DECISION),
        authorization_sha256=sha(ROOT/AUTH), source_protocol=protocol, sources=sources,
        evidence_files=evidence, components=components, committed_at=commit)


def configure_runtime(deadline):
    core = ensemble.mitra
    reference = read(ROOT/'artifacts/run_038/protocol.json')['mitra_environment_sha256']
    adapted = read(ROOT/'artifacts/run_046/protocol.json')['full_episode_policy']['environment_sha256']
    environment = core.environment_lock()
    if hashlib.sha256(environment.encode()).hexdigest()!=reference or reference!=adapted:
        raise RuntimeError('Original and adapted frozen environments must match runtime.')
    core.verify_checkpoint()
    if not core.torch.backends.mps.is_available():
        raise RuntimeError('Fresh MPS process required; no CPU fallback.')
    if core.torch.mps.current_allocated_memory()!=0:
        raise RuntimeError('No previous MPS model may remain resident.')
    core.torch.set_num_threads(4)
    core.torch.mps.set_per_process_memory_fraction(.65)
    core.DEADLINE=deadline
    return environment


def load_attributes():
    import iterate
    train = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda col: col!='Transported')
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=lambda col: col!='Transported')
    if (len(train),len(test))!=(8693,4277):
        raise RuntimeError('Official roster row count differs.')
    roster=pd.concat([train,test],ignore_index=True)
    frame=iterate.build_features(roster,False)[0]
    if len(frame)!=12970 or list(frame.index)!=list(range(12970)):
        raise RuntimeError('Feature row alignment changed.')
    return train,test,frame


def check_component(model, train_frame, version):
    core=ensemble.mitra
    if type(model) is not core.MitraPredictor:
        raise RuntimeError('Expected original portable MitraPredictor.')
    labels=core.binary_labels(model.labels,8693)
    if (model.query_chunk!=2318 or model.config!=core.MODEL_CONFIG
            or model.preprocessor.columns!=core.COMPACT or len(model.support)!=8693
            or model.metadata.get('training_rows')!=8693
            or model.metadata.get('training_ids_sha256')!=core.ids_hash(np.arange(8693))
            or model.metadata.get('checkpoint_sha256')!=core.WEIGHT_SHA):
        raise RuntimeError('Full-context model recipe or training IDs differ.')
    metadata=model.metadata
    if version==COMPONENTS[0][0]:
        if metadata.get('recipe')!='zero-shot' or metadata.get('selected_steps')!=0 or metadata.get('refit_optimizer_steps')!=0:
            raise RuntimeError('Original component is not the zero-shot recipe.')
    elif version==COMPONENTS[1][0]:
        settings=read(ROOT/'artifacts/run_046/protocol.json')['full_episode_policy']['settings']
        recipe=metadata.get('recipe',{})
        if (not recipe or recipe.get('max_steps')!=50 or any(settings.get(k)!=v for k,v in recipe.items())
                or metadata.get('inner_optimizer_steps')!=50
                or metadata.get('fresh_refit_from_original_checkpoint') is not True
                or type(metadata.get('selected_steps')) is not int or not 0<=metadata['selected_steps']<=50
                or metadata.get('refit_optimizer_steps')!=metadata['selected_steps']):
            raise RuntimeError('Adapted component is not the fixed full50 recipe.')
    else:
        raise ValueError('Unknown component version.')
    if not np.array_equal(model.support,model.preprocessor.transform(train_frame)):
        raise RuntimeError('Persisted support is not the current full training attribute matrix.')
    return dict(training_ids_sha256=core.ids_hash(np.arange(8693)),
        support_labels_sha256=core.ids_hash(labels), support_rows=8693, query_chunk=2318,
        checkpoint_sha256=core.WEIGHT_SHA)


def prepare():
    evidence=decision_evidence()  # Before attribute read, model deserialization, or inference.
    out,report=ROOT/'artifacts'/PROPOSAL,ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():
        raise RuntimeError('No overwrite or implicit resume.')
    out.mkdir(parents=True);report.mkdir(parents=True)
    started=time.monotonic();models=[];component=combined=restored=None
    previous=None
    try:
        def timeout(*_): raise TimeoutError('Complete sequential export exceeded3600s; no partial result.')
        previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,3600)
        environment=configure_runtime(started+3600)
        (out/'environment.lock.txt').write_text(environment)
        env_sha=sha(out/'environment.lock.txt')
        for name in evidence['sources']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        write(out/'preparation_protocol.json',dict(**evidence,environment_sha256=env_sha,
            budget_seconds=3600,no_fit=True,no_csv_targets_read=True,weights=[.5,.5]))
        train,test,frame=load_attributes();query=frame.iloc[8693:]
        predictions=[];expected_predictions=[];logs=[]
        for version,run,proposal in COMPONENTS:
            tick=time.monotonic();directory=ROOT/'artifacts'/proposal
            cache=pd.read_csv(directory/'test_probabilities.csv',dtype={'PassengerId':str})
            if len(cache)!=4277 or not np.array_equal(cache.PassengerId.astype(str),test.PassengerId.astype(str)):
                raise RuntimeError('Component cached PassengerIds differ.')
            expected=common.probability_vector(cache.probability.to_numpy(),4277)
            component=joblib.load(directory/'model.joblib')
            record=check_component(component,frame.iloc[:8693],version)
            component.device='mps'
            try:
                actual=ensemble.probabilities(component.predict_proba(query),4277)[:,1]
                difference=common.check_reload(expected,actual)
            finally:
                component.release();ensemble.mitra.release_memory()
            record.update(version=version,proposal=proposal,reload_rows=4277,
                reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,
                source_model_sha256=sha(directory/'model.joblib'),seconds=time.monotonic()-tick,
                historical_export_reload_inference_verified=evidence['components'][len(logs)]['historical_export_reload_inference_verified'])
            predictions.append(actual.copy());expected_predictions.append(expected.copy());logs.append(record)
            component.device='cpu';models.append(component);component=None
            write(out/'component_reload_log.json',logs)
        if not np.array_equal(models[0].labels,models[1].labels):
            raise RuntimeError('Full training labels differ between components.')
        first=.5*predictions[0]+.5*predictions[1]
        cached_mean_difference=common.check_reload(.5*expected_predictions[0]+.5*expected_predictions[1],first)
        combined=ensemble.PriorAdaptMeanPredictor(models,metadata=dict(components=logs,weights=[.5,.5],
            version=NAME,no_fit=True,training_ids_sha256=logs[0]['training_ids_sha256'],
            support_labels_sha256=logs[0]['support_labels_sha256']))
        joblib.dump(combined,out/'model.joblib')
        combined.release();combined=None;models=[];gc.collect();ensemble.mitra.release_memory()
        restored=joblib.load(out/'model.joblib')
        if type(restored) is not ensemble.PriorAdaptMeanPredictor or restored.device!='cpu':
            raise RuntimeError('Combined portable reload type/device differs.')
        restored.device='mps'
        result=ensemble.probabilities(restored.predict_proba(query),4277)[:,1]
        difference=common.check_reload(first,result)
        restored.release();restored=None;gc.collect();ensemble.mitra.release_memory()
        pending=out/'pending';pending.mkdir()
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=first>=.5)).to_csv(pending/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=first,reload_probability=result)).to_csv(pending/'test_probabilities.csv',index=False)
        check=helpers.validate_csv(pending/'submission.csv')
        if helpers.duplicate_attempts(check['sha256']): raise RuntimeError('Identical CSV already attempted.')
        after=decision_evidence()
        if {k:v for k,v in after.items() if k!='committed_at'}!={k:v for k,v in evidence.items() if k!='committed_at'}:
            raise RuntimeError('Sources or evidence changed during export.')
        if hashlib.sha256(ensemble.mitra.environment_lock().encode()).hexdigest()!=env_sha:
            raise RuntimeError('Environment changed during export.')
        decision=evidence['decision']
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
            new_fit_performed=False,no_csv_targets_read=True,threshold=.5,export_reload_inference_verified=True,
            reload_rows=4277,reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,
            component_reload_checks=logs,cached_mean_max_absolute_difference=cached_mean_difference,
            cached_mean_threshold_decisions_equal=True,model_sha256=bound['model_sha256'],authorization=bound,
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
            if (directory/'pending').exists():shutil.rmtree(directory/'pending')
            write(directory/'failure.json',dict(status='failed_incomplete_not_upload_eligible',candidate=NAME,
                error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started))
        raise
    finally:
        for model in [component,combined,restored,*models]:
            if model is not None:model.release()
        ensemble.mitra.release_memory();ensemble.mitra.DEADLINE=None
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
