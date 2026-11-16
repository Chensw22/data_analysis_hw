"""Prepare one explicitly reviewed v85 group-screen proposal; never upload.

Full labels are read only after the exact committed decision and complete group
screen/resource evidence pass. No random-screen identity or stability gate is invented.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

import prepare_top5_submission as common

ROOT=Path(__file__).resolve().parent
BATCH='online_iteration_20261116'
AUTH=f'reports/{BATCH}/authorization.json'
DECISION=f'reports/{BATCH}/group_episode_decision.json'
MODULE='mitra_group_episode_candidate'
NAME='v85_Mitra_group_episode50'
PHASE='group_episode_20261116'
PROPOSAL='proposal_nov16_group_episode'
GROUP=f'artifacts/{PHASE}'
GROUP_REPORT=f'reports/{PHASE}'
RESOURCE='artifacts/proposal_nov16_full_episode'
RESOURCE_REPORT='reports/proposal_nov16_full_episode'
RESOURCE_ENVELOPE=dict(max_inner_support=7000,max_inner_query=1800,max_full_support=8693,max_query_chunk=2318)
BUDGET=10800
OWN_SOURCES=['prepare_group_episode_submission.py','test_prepare_group_episode_submission.py',
    'prepare_top5_submission.py','submit_iteration.py','kaggle_workflow.py']
read,write=common.read,common.write


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def relative_file(value):
    path=Path(value)
    if (path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file()
            or not (ROOT/path).resolve().is_relative_to(ROOT.resolve())):
        raise ValueError('Existing repository-relative evidence file required.')
    return path.as_posix()


def required_evidence_paths():
    files=[f'{GROUP}/{name}' for name in ['protocol.json','summary.json','progress.json','folds.npz','group_20261112.npz']]
    files += [f'{GROUP_REPORT}/{name}' for name in ['protocol.json','summary.json','progress.json','review.json','group_metrics.csv']]
    for fold in range(3):
        for prefix in [GROUP,GROUP_REPORT]:
            files += [f'{prefix}/fold{fold}.json',f'{prefix}/fold{fold}_training.json']
    files += [f'{RESOURCE}/{name}' for name in ['manifest.json','authorization.json','selection.json',
        'fit_metadata.json','reload_metadata.json','preparation_protocol.json','environment.lock.txt','model.joblib']]
    files += [f'{RESOURCE_REPORT}/{name}' for name in ['manifest.json','authorization.json','selection.json']]
    return files


def require_committed(paths):
    paths=sorted(paths)
    tracked=subprocess.check_output(['git','ls-files','-z','--',*paths],cwd=ROOT).decode()
    if set(filter(None,tracked.split('\0')))!=set(paths):
        raise RuntimeError('Commit the decision, public evidence and every preparation source first.')
    if subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT).strip():
        raise RuntimeError('Reviewed preparation files differ from committed versions.')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def valid_accuracy(value):
    return type(value) in (float,int) and np.isfinite(value) and 0<=value<=1


def check_training_log(log,rows,seed,recipe,ids_sha):
    selected=log.get('selected_steps')
    if (log.get('status')!='fit_complete' or log.get('recipe')!=recipe
            or log.get('training_rows')!=rows or log.get('seed')!=seed
            or log.get('training_ids_sha256')!=ids_sha
            or type(selected) is not int or not 0<=selected<=50
            or log.get('inner_optimizer_steps')!=50 or log.get('refit_optimizer_steps')!=selected
            or log.get('fresh_refit_from_original_checkpoint') is not True
            or log.get('full_prediction_support') is not True or log.get('query_chunk')!=2318
            or log.get('parameters')!=75670026
            or [r.get('step') for r in log.get('inner_history',[])]!=list(range(51))
            or [r.get('step') for r in log.get('refit_history',[])]!=list(range(1,selected+1))):
        raise RuntimeError('Incomplete or mismatched full-parameter training evidence.')


def verify_group_evidence(candidate,decision):
    """Inspect completed artifacts and identifiers only; do not parse target labels."""
    root=ROOT/GROUP;report=ROOT/GROUP_REPORT
    for folder in [root,report]:
        if (folder/'failure.json').exists():raise RuntimeError('Failed group phase cannot authorize full fitting.')
    for name in ['protocol.json','summary.json','progress.json']:
        if read(root/name)!=read(report/name):raise RuntimeError('Public and local group evidence differ.')
    protocol,summary,progress=[read(root/name) for name in ['protocol.json','summary.json','progress.json']]
    candidate.assert_frozen(protocol)
    if (protocol.get('candidate')!=NAME or protocol.get('recipe')!=candidate.core.SETTINGS
            or protocol.get('screen')!=candidate.SCREEN_SETTINGS
            or summary.get('status')!='complete' or summary.get('candidate')!=NAME
            or summary.get('kind')!='group' or summary.get('seed')!=20261112
            or summary.get('development_rows')!=6954 or summary.get('no_old_holdout_targets_used') is not True
            or summary.get('automatic_eligibility') is not False or summary.get('random_tier_update') is not False
            or summary.get('protocol_sha256')!=sha(root/'protocol.json')
            or summary.get('oof_sha256')!=sha(root/'group_20261112.npz')
            or progress.get('status')!='complete' or progress.get('whole_score_available') is not True
            or progress.get('completed_folds')!=summary.get('folds')
            or [r.get('fold') for r in summary.get('folds',[])]!=[0,1,2]
            or not valid_accuracy(summary.get('accuracy'))
            or abs(summary['accuracy']-decision['group_accuracy'])>1e-12):
        raise RuntimeError('Complete fixed three-fold group evidence and exact score are required.')
    with np.load(root/'group_20261112.npz',allow_pickle=False) as saved:
        oof={k:saved[k].copy() for k in saved.files}
    with np.load(root/'folds.npz',allow_pickle=False) as saved:
        for key in ['dev_ids','group_ids','passenger_ids','fold_id']:
            if not np.array_equal(oof[key],saved[key]):raise RuntimeError('OOF identity differs from frozen folds.')
    dev=oof['dev_ids'];groups=oof['group_ids'].astype(str);passengers=oof['passenger_ids'].astype(str)
    if len(dev)!=6954 or candidate.audit.row_hash(dev)!=candidate.audit.DEV_HASH:
        raise RuntimeError('Original6954development membership required.')
    reference=candidate.REFERENCE
    with np.load(reference/'group_20261112.npz',allow_pickle=False) as saved:
        for key in ['dev_ids','group_ids','passenger_ids','fold_id']:
            if not np.array_equal(oof[key],saved[key]):raise RuntimeError('Candidate uses different reference fold IDs.')
        if not np.array_equal(oof['reference_prob'],saved['prob']):raise RuntimeError('Cached reference probabilities changed.')
    row_files=[]
    for fold in range(3):
        with np.load(reference/f'group_20261112_fold{fold}_rows.npz',allow_pickle=False) as saved:
            row_files.append({k:saved[k].copy() for k in saved.files})
    _,_,folds=candidate.validate_cached_arrays(oof,row_files,dev,groups,passengers)
    for fold,(a,b) in enumerate(folds):
        row=summary['folds'][fold]
        log=read(root/f'fold{fold}_training.json')
        for name in [f'fold{fold}.json',f'fold{fold}_training.json']:
            if read(root/name)!=read(report/name):raise RuntimeError('Public fold evidence differs.')
        if (row!=read(root/f'fold{fold}.json') or row.get('fit_seed')!=20261112+fold
                or row.get('train_rows')!=len(a) or row.get('query_rows')!=len(b)
                or row.get('train_ids_sha256')!=candidate.audit.row_hash(dev[a])
                or row.get('query_ids_sha256')!=candidate.audit.row_hash(dev[b])
                or row.get('group_overlap_count')!=0 or not valid_accuracy(row.get('accuracy'))
                or not valid_accuracy(row.get('reference_accuracy'))
                or row.get('selected_steps')!=log.get('selected_steps')
                or log.get('inner_group_overlap_count')!=0
                or log.get('training_group_ids_sha256')!=candidate.core.group_hash(groups[a])):
            raise RuntimeError('Incomplete fixed fold identity, seed, group boundary or result.')
        check_training_log(log,len(a),20261112+fold,candidate.core.SETTINGS,candidate.audit.row_hash(dev[a]))
    weighted=sum(r['accuracy']*r['query_rows'] for r in summary['folds'])/6954
    if abs(weighted-summary['accuracy'])>1e-12:raise RuntimeError('Whole group accuracy differs from completed folds.')
    return protocol,summary


def verify_v81_resource(candidate):
    """Use prior completed full-fit metadata/portable-file hashes, never load its labels."""
    folder=ROOT/RESOURCE;report=ROOT/RESOURCE_REPORT
    for name in ['manifest.json','authorization.json','selection.json']:
        if read(folder/name)!=read(report/name):raise RuntimeError('Original v81 resource report differs.')
    manifest=read(folder/'manifest.json');fit=read(folder/'fit_metadata.json');reload=read(folder/'reload_metadata.json')
    preparation=read(folder/'preparation_protocol.json')
    env_sha=hashlib.sha256(candidate.base.environment_lock().encode()).hexdigest()
    if (manifest.get('status')!='complete' or manifest.get('selection',{}).get('version')!='v81_Mitra_full_episode50'
            or manifest.get('fit_rows')!=8693 or manifest.get('rows')!=4277 or manifest.get('reload_rows')!=4277
            or manifest.get('export_reload_inference_verified') is not True
            or manifest.get('reload_threshold_decisions_equal') is not True
            or not 0<=manifest.get('reload_max_absolute_difference',float('inf'))<=2e-6
            or manifest.get('model_sha256')!=sha(folder/'model.joblib')
            or read(folder/'authorization.json').get('model_sha256')!=manifest['model_sha256']
            or manifest.get('environment_sha256')!=env_sha or sha(folder/'environment.lock.txt')!=env_sha
            or preparation.get('environment_sha256')!=env_sha
            or preparation.get('sources_frozen_before_full_labels') is not True
            or fit!=reload or fit.get('inner_training_rows')!=6954 or fit.get('inner_validation_rows')!=1739
            or fit.get('checkpoint_sha256')!=candidate.base.WEIGHT_SHA
            or fit.get('precision')!='MPS bf16 autocast with FP32 weights'
            or fit.get('prediction_rows')!=4277):
        raise RuntimeError('Matching completed v81 full fit/reload resource evidence required.')
    expected=dict(max_steps=50,episode_support=1024,episode_query=256,query_chunk=2318,
        trainable_parameters=75670026,checkpoint_sha256=candidate.base.WEIGHT_SHA,
        learning_rate=3e-6,weight_decay=.3,fresh_full_refit=True)
    if any(fit.get('recipe',{}).get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Prior resource recipe does not establish this full-parameter backend.')
    check_training_log(fit,8693,20261112,fit['recipe'],candidate.audit.row_hash(np.arange(8693)))
    if fit['selected_steps']<1:raise RuntimeError('Prior full fit must include actual optimizer refit steps.')
    # Old orchestration can evolve: authenticate its frozen snapshots rather
    # than pretending every old exporter remains the current implementation.
    for relative,digest in manifest['provenance']['source_hashes'].items():
        if sha(folder/'source'/relative)!=digest:raise RuntimeError('Prior v81 source snapshot changed: '+relative)
    for relative in ['mitra_candidate.py','mitra_full_episode.py','mitra_full_episode_probe.py']:
        if manifest['provenance']['source_hashes'].get(relative)!=sha(ROOT/relative):
            raise RuntimeError('Prior v81 numerical/resource backend differs.')
    return dict(proposal='proposal_nov16_full_episode',inner_support=6954,inner_query=1739,
        selected_steps=fit['selected_steps'],full_support=8693,full_queries=4277,
        manifest_sha256=sha(folder/'manifest.json'),fit_metadata_sha256=sha(folder/'fit_metadata.json'),
        reload_metadata_sha256=sha(folder/'reload_metadata.json'))


def decision_evidence():
    """Exact committed decision gate; never reads complete training targets."""
    decision=read(ROOT/DECISION);auth=read(ROOT/AUTH)
    if (auth.get('batch_id')!=BATCH or auth.get('competition')!='spaceship-titanic'
            or auth.get('maximum_upload_attempts')!=10 or auth.get('weekly_reserve_percent')!=30
            or auth.get('no_duplicate_or_blind_retry_upload') is not True):
        raise RuntimeError('Current bounded batch authorization changed.')
    if (decision.get('batch_id')!=BATCH or decision.get('batch_authorization_sha256')!=sha(ROOT/AUTH)
            or decision.get('module')!=MODULE or decision.get('version')!=NAME
            or decision.get('source_phase')!=PHASE or decision.get('proposal')!=PROPOSAL
            or decision.get('validation_kind')!='group' or not valid_accuracy(decision.get('group_accuracy'))
            or 'screen_accuracy' in decision or 'run' in decision
            or decision.get('exploratory') is not True or decision.get('prior_A_eligible') is not None
            or decision.get('prior_S_supported') is not False or decision.get('threshold')!=.5
            or decision.get('maximum_attempts')!=1 or decision.get('resource_envelope')!=RESOURCE_ENVELOPE
            or not isinstance(decision.get('rationale'),str) or not decision['rationale'].strip()
            or not isinstance(decision.get('resource_extrapolation_disclosure'),str)
            or not decision['resource_extrapolation_disclosure'].strip()):
        raise RuntimeError('Decision identity, group evidence or exploratory scope differs.')
    protocol_path=f'{GROUP}/protocol.json'
    if decision.get('source_protocol_sha256')!=sha(ROOT/protocol_path):
        raise RuntimeError('Decision group protocol binding changed.')
    protocol=read(ROOT/protocol_path);sources=dict(protocol['sources'])
    for filename in ['mitra_group_episode.py',MODULE+'.py']:
        if filename not in sources:raise RuntimeError('Group implementation was not frozen before screening.')
    evidence=decision.get('evidence_files')
    if not isinstance(evidence,dict) or not set(required_evidence_paths())<=set(evidence):
        raise RuntimeError('Decision must bind complete group and v81 full-resource evidence.')
    for mapping in [sources,evidence,protocol['historical_files']]:
        for filename,digest in mapping.items():
            if sha(ROOT/relative_file(filename))!=digest:raise RuntimeError('Bound source/evidence changed: '+filename)
    sources.update({filename:sha(ROOT/filename) for filename in OWN_SOURCES+[AUTH,DECISION]})
    sources.update({filename:digest for filename,digest in evidence.items() if filename.startswith('reports/')})
    commit=require_committed(sources)
    import submit_iteration
    if submit_iteration.count_attempts(ROOT/'reports',BATCH)>=10:
        raise RuntimeError('The ten authorized attempts are already consumed.')
    candidate=importlib.import_module(MODULE)
    if candidate.NAME!=NAME:raise RuntimeError('Candidate module identity differs.')
    _,summary=verify_group_evidence(candidate,decision)
    resource=verify_v81_resource(candidate)
    return dict(decision=decision,decision_path=DECISION,decision_sha256=sha(ROOT/DECISION),
        authorization_sha256=sha(ROOT/AUTH),source_protocol=protocol,sources=sources,
        evidence_files=evidence,committed_at=commit,group_summary=summary,fullfit_resource=resource)


def read_full_fit_data(candidate):
    """Only prepare(), after committed decision_evidence(), may call this reader."""
    candidate.verify_data_files()
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    if len(train)!=8693 or len(test)!=4277:raise RuntimeError('Complete8693/4277 row counts required.')
    passengers=train.PassengerId.astype(str);test_ids=test.PassengerId.astype(str)
    if (passengers.duplicated().any() or test_ids.duplicated().any() or set(passengers)&set(test_ids)):
        raise RuntimeError('Unique disjoint training/test PassengerIds required.')
    y=candidate.base.binary_labels(train.Transported.to_numpy(),8693)
    groups=passengers.str.split('_').str[0].to_numpy()
    roster=pd.concat([train.drop(columns='Transported'),test.drop(columns='Transported',errors='ignore')],ignore_index=True)
    features,_=candidate.iterate.build_features(roster,False)
    return train,test,features,y,groups


def fullfit_preflight(candidate,y,groups):
    inner,valid=candidate.core.inner_split(y,groups,20261112)
    if (len(y)!=8693 or len(inner)>RESOURCE_ENVELOPE['max_inner_support']
            or len(valid)>RESOURCE_ENVELOPE['max_inner_query'] or len(inner)+len(valid)!=8693
            or set(groups[inner])&set(groups[valid])):
        raise RuntimeError('Full group inner split exceeds the reviewed resource envelope; do not change split/context.')
    return dict(inner_support=len(inner),inner_query=len(valid),full_support=8693,
        inner_group_overlap=0,resource_envelope=RESOURCE_ENVELOPE,
        disclosure='Small extrapolation from successful v81 inner6954/1739 full fit; not an identical resource probe.')


def release(candidate,model):
    if model is not None:model.release()
    candidate.base.release_memory()


def validate_csv(path):
    from kaggle_workflow import validate_submission
    return validate_submission(path,ROOT/'data/raw/test.csv',expected_rows=4277)


def duplicate_attempts(digest):
    return [str(p.relative_to(ROOT)) for p in (ROOT/'reports').rglob('online_attempts/*.json')
        if read(p).get('file_sha256')==digest]


def prepare():
    out=ROOT/'artifacts'/PROPOSAL;report=ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():raise RuntimeError('Proposal already exists; no overwrite or implicit resume.')
    evidence=decision_evidence()  # This includes committed root decision before full labels.
    decision=evidence['decision'];candidate=importlib.import_module(MODULE)
    out.mkdir(parents=True);report.mkdir(parents=True)
    model=restored=None;old=None;started=time.monotonic()
    try:
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
            raise RuntimeError('Disable automatic MPS fallback.')
        if not candidate.base.torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
        candidate.base.torch.set_num_threads(4);candidate.base.torch.mps.set_per_process_memory_fraction(.65)
        candidate.base.DEADLINE=started+BUDGET
        def timeout(*_):raise TimeoutError('Fixed10800second full group fit/export/reload budget exceeded.')
        old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
        environment=candidate.base.environment_lock();(out/'environment.lock.txt').write_text(environment)
        env_sha=sha(out/'environment.lock.txt')
        for filename in evidence['sources']:
            target=out/'source'/filename;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/filename,target)
        write(out/'preparation_protocol.json',dict(**evidence,environment_sha256=env_sha,
            budget_seconds=BUDGET,sources_frozen_before_full_labels=True,validation_kind='group',
            full_labels_allowed_only_for_this_final_fit=True,original_failed_gates_unchanged=True))
        _,test,x,y,groups=read_full_fit_data(candidate)
        preflight=fullfit_preflight(candidate,y,groups);write(out/'fullfit_preflight.json',preflight)
        ids=np.arange(8693,dtype=np.int64);query_ids=np.arange(8693,12970,dtype=np.int64)
        model=candidate.core.fit_group_episode(x.iloc[ids],y,ids,groups,20261112,
            device='mps',log_path=out/'fit_training.json',max_steps=50)
        if (type(model) is not candidate.base.MitraPredictor or len(model.labels)!=8693
                or len(model.support)!=8693 or not np.array_equal(model.labels,y)
                or model.query_chunk!=2318 or model.metadata.get('inner_training_rows')!=preflight['inner_support']
                or model.metadata.get('inner_validation_rows')!=preflight['inner_query']):
            raise RuntimeError('Original full-context Mitra predictor or fitted split differs.')
        check_training_log(model.metadata,8693,20261112,candidate.core.SETTINGS,candidate.audit.row_hash(ids))
        if (model.metadata.get('inner_group_overlap_count')!=0
                or model.metadata.get('training_group_ids_sha256')!=candidate.core.group_hash(groups)):
            raise RuntimeError('Full-fit group context or inner boundary differs.')
        result=np.asarray(model.predict_proba(x.iloc[query_ids]))
        if result.shape!=(4277,2):raise RuntimeError('Full binary prediction shape differs.')
        first=common.probability_vector(result[:,1],4277)
        candidate.base.check_deadline()
        if any(t.device.type!='cpu' for t in model.state.values()):raise RuntimeError('Portable model state must be CPU.')
        joblib.dump(model,out/'model.joblib');write(out/'fit_metadata.json',model.metadata)
        release(candidate,model);model=None;gc.collect();candidate.base.release_memory()
        restored=joblib.load(out/'model.joblib')
        if (type(restored) is not candidate.base.MitraPredictor or restored.device!='cpu'
                or restored._engine is not None or len(restored.labels)!=8693 or len(restored.support)!=8693):
            raise RuntimeError('Complete portable CPU artifact required before reload inference.')
        restored.device='mps'
        reloaded=np.asarray(restored.predict_proba(x.iloc[query_ids]))
        if reloaded.shape!=(4277,2):raise RuntimeError('Full reload shape differs.')
        difference=common.check_reload(first,reloaded[:,1]);candidate.base.check_deadline()
        write(out/'reload_metadata.json',restored.metadata)
        release(candidate,restored);restored=None;gc.collect();candidate.base.release_memory()
        pending=out/'pending';pending.mkdir()
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=first>=.5)).to_csv(pending/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=first,reload_probability=reloaded[:,1])).to_csv(pending/'test_probabilities.csv',index=False)
        check=validate_csv(pending/'submission.csv')
        if duplicate_attempts(check['sha256']):raise RuntimeError('Identical CSV already has an online attempt.')
        after=decision_evidence()
        if {k:v for k,v in after.items() if k!='committed_at'}!={k:v for k,v in evidence.items() if k!='committed_at'}:
            raise RuntimeError('Decision, group evidence or resource provenance changed during preparation.')
        if hashlib.sha256(candidate.base.environment_lock().encode()).hexdigest()!=env_sha:
            raise RuntimeError('Frozen export environment changed.')
        candidate.base.check_deadline()
        selection=dict(version=NAME,source_phase=PHASE,validation_kind='group',components=[NAME],
            stage='explicit_autonomous_group_exploration',A_eligible=None,S_supported=False,
            group_accuracy=decision['group_accuracy'],threshold=.5)
        bound=dict(status='bound_single_upload_authorization',competition='spaceship-titanic',batch_id=BATCH,
            source_authorization_sha256=evidence['authorization_sha256'],proposal=PROPOSAL,version=NAME,
            file_sha256=check['sha256'],model_sha256=sha(out/'model.joblib'),maximum_upload_attempts_for_this_file=1,
            desktop_reminder_required=True,decision_sha256=evidence['decision_sha256'],decision_path=DECISION,
            source_protocol_sha256=decision['source_protocol_sha256'],prepared_from_commit=evidence['committed_at'],
            no_duplicate_or_blind_retry_upload=True)
        manifest=dict(**check,status='complete',batch_id=BATCH,selection=selection,validation_kind='group',
            fit_rows=8693,threshold=.5,export_reload_inference_verified=True,reload_rows=4277,
            reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,
            model_sha256=bound['model_sha256'],authorization=bound,original_failed_gates_unchanged=True,
            source_hashes=evidence['sources'],environment_sha256=env_sha,fullfit_preflight=preflight,
            provenance=dict(source_hashes=evidence['sources'],decision_path=DECISION,
                decision_sha256=evidence['decision_sha256'],evidence_files=evidence['evidence_files'],
                source_protocol_sha256=decision['source_protocol_sha256'],prepared_from_commit=evidence['committed_at']),
            seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat())
        for name in ['submission.csv','test_probabilities.csv']:(pending/name).replace(out/name)
        pending.rmdir()
        for name,value in [('selection.json',selection),('authorization.json',bound),('manifest.json',manifest)]:
            write(out/name,value);write(report/name,value)
        print(f'Prepared {PROPOSAL}:4277rows; group evidence; full reload difference={difference}.',flush=True)
        return manifest
    except BaseException as error:
        shutil.rmtree(out/'pending',ignore_errors=True)
        for folder in [out,report]:
            for name in ['manifest.json','authorization.json','selection.json','submission.csv','test_probabilities.csv']:
                (folder/name).unlink(missing_ok=True)
            write(folder/'failure.json',dict(status='failed_incomplete_not_upload_eligible',candidate=NAME,
                error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started))
        raise
    finally:
        if old is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        release(candidate,model);release(candidate,restored);candidate.base.DEADLINE=None


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();prepare()
