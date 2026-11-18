"""Full zero-shot composition19 export; no CV, adaptation, or automatic upload."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import joblib
import numpy as np
import pandas as pd
import torch

import mitra_spend_composition_nov18 as composition
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import error_rescue_feature_probe as previous
import exaone_validation_data as data
import prepare_v84_online_first as common

ROOT=Path(__file__).resolve().parent
NAME='v103_Mitra_spend_composition19_zeroshot'
PROPOSAL='proposal_nov18_composition19'
POLICY='reports/error_followup_20261118/composition_export_policy.json'
DIAGNOSTIC='artifacts/error_followup_20261118/composition_fold0'
RESULT='reports/error_followup_20261118/composition_result.json'
AUDIT='reports/error_followup_20261118/composition_root_audit.json'
SEED,BUDGET=20261112,1800
sha,read,write=common.sha,common.read,common.write


def export():
    out,report=ROOT/'artifacts'/PROPOSAL,ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():raise RuntimeError('No overwrite/resume.')
    policy=read(ROOT/POLICY)
    if (policy['version']!=NAME or policy['allow_full_export'] is not True or policy['allow_direct_submit'] is not False
            or policy['seed']!=SEED or policy['budget_seconds']!=BUDGET or policy['columns']!=composition.COLUMNS):
        raise RuntimeError('Exact zero-shot export-only policy required.')
    frozen=read(ROOT/DIAGNOSTIC/'freeze.json');result=read(ROOT/RESULT);audit=read(ROOT/AUDIT)
    if (sha(ROOT/DIAGNOSTIC/'freeze.json')!=result['freeze_sha256']
            or result['status']!='complete_single_fold_adaptive_diagnostic'
            or audit['status']!='independently_verified'
            or audit['prediction_sha256']!=result['prediction_sha256']
            or sha(ROOT/DIAGNOSTIC/'predictions.npz')!=result['prediction_sha256']):
        raise RuntimeError('Diagnostic/audit identity differs.')
    for path,digest in policy['evidence_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Export evidence changed: '+path)
    previous.verify_sources(frozen)
    _,lock=counts.verify_reference()
    names=[Path(__file__).name,POLICY,RESULT,AUDIT,'kaggle_workflow.py']
    names+= [name for name in frozen['sources'] if name.endswith('.py') and not name.startswith('artifacts/')]
    commit=previous.committed(sorted(set(names)))
    sources={**frozen['sources'],**policy['evidence_sha256'],**{name:sha(ROOT/name) for name in names}}
    for name in ['train','test','sample_submission']:
        path=f'data/raw/{name}.csv';sources[path]=sha(ROOT/path)
    out.mkdir(parents=True);report.mkdir(parents=True)
    started=time.monotonic();base.DEADLINE=started+BUDGET
    def timeout(*_):raise TimeoutError('Complete composition19 export exceeded1800seconds.')
    handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    model=predictor=restored=None
    try:
        protocol=dict(version=NAME,proposal=PROPOSAL,source_commit=commit,source_hashes=sources,
            policy=policy,checkpoint_sha256=base.WEIGHT_SHA,environment_sha256=frozen['environment_sha256'],
            budget_seconds=BUDGET,optimizer_steps=0,original13_prefix_exact=True)
        for folder in (out,report):write(folder/'protocol.json',protocol)
        (out/'environment.lock.txt').write_text(lock)
        counts.require_false()
        if not torch.backends.mps.is_available():raise RuntimeError('Original MPS required; no fallback.')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        # Full labels are parsed only after committed policy/source and frozen evidence checks.
        train=pd.read_csv(ROOT/'data/raw/train.csv')
        test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda name:name!='Transported')
        if len(train)!=8693 or len(test)!=4277:raise RuntimeError('Complete8693/4277 required.')
        roster=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
        frame=data.compact_features(roster);x,q=frame.iloc[:8693],frame.iloc[8693:]
        y=base.binary_labels(train.Transported.astype(np.int64).to_numpy(),8693)
        ids=np.arange(8693,dtype=np.int64);test_ids=test.PassengerId.astype(str).to_numpy()
        if len(np.unique(test_ids))!=4277 or not np.array_equal(roster.PassengerId.iloc[8693:].astype(str),test_ids):
            raise RuntimeError('Test PassengerId alignment differs.')
        base.seed_everything(SEED,'mps')
        prep=composition.CompositionPreprocessor().fit(x,y,SEED);support=prep.transform(x)
        if support.shape!=(8693,19) or prep.active_columns!=composition.COLUMNS:
            raise RuntimeError('All19 model input columns required.')
        model,config=base.load_initial_model(base.CHECKPOINT_DIR,'mps')
        meta=dict(version=NAME,recipe='zero-shot',seed=SEED,selected_steps=0,refit_optimizer_steps=0,
            training_rows=8693,query_rows=4277,training_ids_sha256=base.ids_hash(ids),labels_sha256=base.ids_hash(y),
            prediction_passenger_ids_sha256=hashlib.sha256(json.dumps(test_ids.tolist()).encode()).hexdigest(),
            checkpoint_sha256=base.WEIGHT_SHA,input_columns=composition.COLUMNS,feature_count=19,
            original13_prefix_exact=True,full_prediction_support=True,query_chunk=2318)
        predictor=base.MitraPredictor(prep,config,model.state_dict(),support,y,'mps',meta,query_chunk=2318)
        model=None;base.release_memory()
        first=predictor.predict_proba(q)[:,1]
        if first.shape!=(4277,) or not np.isfinite(first).all() or np.any((first<0)|(first>1)):
            raise RuntimeError('Complete finite probabilities required.')
        write(out/'fit_metadata.json',meta);joblib.dump(predictor,out/'model.joblib')
        predictor.release();predictor=None;base.release_memory()
        restored=joblib.load(out/'model.joblib')
        if (restored.device!='cpu' or restored._engine is not None or restored.support.shape!=(8693,19)
                or restored.query_chunk!=2318 or restored.preprocessor.columns!=composition.COLUMNS
                or restored.metadata!=meta):raise RuntimeError('Portable full19 recipe differs.')
        np.testing.assert_array_equal(restored.labels,y)
        np.testing.assert_array_equal(restored.support,restored.preprocessor.transform(x))
        restored.device='mps';again=restored.predict_proba(q)[:,1]
        difference=common.compare(first.tolist(),again.tolist())
        if difference!=0 or not np.array_equal(first,again):raise RuntimeError('Complete4277 reload must be exact0; no tolerance rescue.')
        restored.release();restored=None;base.release_memory();counts.require_false()
        pd.DataFrame(dict(PassengerId=test_ids,probability=first,reload_probability=again)).to_csv(out/'test_probabilities.csv',index=False)
        pd.DataFrame(dict(PassengerId=test_ids,Transported=first>=.5)).to_csv(out/'submission.csv',index=False)
        import kaggle_workflow
        check=kaggle_workflow.validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
        for path in (ROOT/'reports').rglob('online_attempts/*.json'):
            if read(path).get('file_sha256')==check['sha256']:raise RuntimeError('Duplicate CSV already attempted.')
        for path,digest in sources.items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Frozen source/evidence changed: '+path)
        if base.environment_lock()!=lock:raise RuntimeError('Environment changed.')
        base.verify_checkpoint();base.check_deadline()
        write(out/'reload_metadata.json',dict(rows=4277,max_absolute_difference=difference,threshold_equal=True,
            complete_support_rows=8693,actual_features=19,checkpoint_sha256=base.WEIGHT_SHA))
        selection=dict(version=NAME,components=[NAME],threshold=.5,optimizer_steps=0,
            stage='prepared_pending_separate_release',A_eligible=None,S_supported=False,
            automatic_upload_authorized=False,one_fold_adaptive_evidence_only=True)
        manifest=dict(**check,status='complete',version=NAME,proposal=PROPOSAL,training_rows=8693,fit_rows=8693,
            feature_count=19,threshold=.5,model_sha256=sha(out/'model.joblib'),test_probabilities_sha256=sha(out/'test_probabilities.csv'),
            export_reload_inference_verified=True,reload_rows=4277,reload_max_absolute_difference=difference,
            reload_threshold_decisions_equal=True,checkpoint_sha256=base.WEIGHT_SHA,
            policy_path=POLICY,policy_sha256=sha(ROOT/POLICY),source_commit=commit,source_hashes=sources,
            environment_sha256=frozen['environment_sha256'],selection=selection,no_new_local_score=True,
            seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat())
        for name,value in [('selection.json',selection),('manifest.json',manifest)]:
            for folder in (out,report):write(folder/name,value)
        print(json.dumps(dict(status='complete_pending_release',proposal=PROPOSAL,sha256=check['sha256'],reload_difference=difference)),flush=True)
        return manifest
    except BaseException as error:
        for folder in (out,report):
            for name in ('manifest.json','selection.json','submission.csv'):(folder/name).unlink(missing_ok=True)
            write(folder/'failure.json',dict(status='failed_not_upload_eligible',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        for active in (predictor,restored):
            if active is not None:active.release()
        model=predictor=restored=None;base.release_memory()
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':export()
