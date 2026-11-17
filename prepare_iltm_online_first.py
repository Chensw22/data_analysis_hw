"""One online-first iLTM full fit/export. No CV gate, score search, or upload."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import weakref

import iltm_r128bn_full8_probe as full
import iltm_false_runtime_bridge as bridge

ROOT=Path(__file__).resolve().parent
POLICY='reports/online_iteration_20261117/online_first_policy.json'
VERSION='iLTM_r128bn_native8_online_first'
PROPOSAL='proposal_online_first_iltm'
OUT=ROOT/'artifacts'/PROPOSAL
REPORT=ROOT/'reports'/PROPOSAL
BUDGET=7200
OWN=['prepare_iltm_online_first.py','test_prepare_iltm_online_first.py',POLICY,
     'iltm_false_runtime_bridge.py','exaone_validation_data.py','reports/exaone_validation_features.json',
     'reports/official_data_verification.json']
read,write,sha=full.read,full.write,full.sha


def check_deadline(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Shared7200s online-first preparation budget exceeded.')


def contract():
    # Identity checks only; deliberately no CV/resource-score eligibility gate.
    full.require_committed(OWN)
    policy=read(ROOT/POLICY)
    if (policy.get('competition')!='spaceship-titanic' or policy.get('allow_full_export') is not True
            or VERSION not in policy.get('versions',[])):
        raise RuntimeError('Current online-first policy must include this full export.')
    identity=read(full.REPORT/'freeze.json');full.assert_frozen(identity)
    official=read(ROOT/'reports/official_data_verification.json')
    proof=read(ROOT/'reports/exaone_validation_features.json')
    if proof.get('status')!='exact_compact13_match' or proof.get('columns')!=full.COLUMNS:
        raise RuntimeError('Original compact13 identity proof required.')
    inputs={}
    for name in ['train.csv','test.csv']:
        relative='data/raw/'+name;expected=official['files'][name]['official_sha256']
        if sha(ROOT/relative)!=expected or proof['source_data_sha256'].get(relative)!=expected:
            raise RuntimeError('Official input identity differs: '+name)
        inputs[relative]=expected
    for name,digest in proof['source_data_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Feature proof input/source changed: '+name)
    sources={**identity['sources'],**{name:sha(ROOT/name) for name in OWN}}
    full.require_committed(sources)
    return dict(resource_identity=identity,source_hashes=sources,input_hashes=inputs,policy_path=POLICY,
        policy_sha256=sha(ROOT/POLICY),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


def assert_frozen(frozen):
    full.assert_frozen(frozen['resource_identity'])
    for name,digest in {**frozen['source_hashes'],**frozen['input_hashes']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Prepared source/input changed: '+name)


def load_full_inputs(frozen):
    assert_frozen(frozen)
    import numpy as np
    import pandas as pd
    import exaone_validation_data as data
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    if (len(train)!=8693 or len(test)!=4277 or 'Transported' in test
            or train.Transported.isna().any() or set(train.Transported.unique())!={False,True}):
        raise RuntimeError('Exact complete official training targets/test rows required.')
    y=train.Transported.astype(np.int64).to_numpy()
    attrs=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
    features=data.compact_features(attrs)
    if data.feature_hash(features)!=read(ROOT/data.PROOF)['compact_values_sha256']:
        raise RuntimeError('Original ordered compact13 values changed.')
    return features.iloc[:8693].reset_index(drop=True),y,features.iloc[8693:].reset_index(drop=True),test.PassengerId.astype(str).to_numpy(),train.PassengerId.astype(str).to_numpy()


def worker(deadline):
    record=dict(status='failed_inconclusive',version=VERSION,training_rows=8693,query_rows=4277,
        seed=full.SEED,fit_eval_set=None,fit_max_time=None,no_cv_performed=True)
    started=time.monotonic();model=restored=native_class=None;observing=False;old_profile=None
    def stage(name):
        bridge.require_false(name);check_deadline(deadline);record['current_stage']=name;write(OUT/'training_log.json',record)
    try:
        stage('before_numeric_imports');frozen=read(OUT/'protocol.json');assert_frozen(frozen)
        if os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')!=str(Path(full.importlib.util.find_spec('torch').origin).parent/'lib'):
            raise RuntimeError('Exact isolated child loader required.')
        import numpy as np
        import pandas as pd
        import joblib
        import torch
        from threadpoolctl import threadpool_info,threadpool_limits
        sys.path.insert(0,str(full.VENDOR))
        from iltm import iLTMClassifier
        import iltm.inference_interface as interface
        if Path(interface.__file__).resolve()!=(full.VENDOR/'iltm/inference_interface.py').resolve():raise RuntimeError('Wrong native source.')
        native_class=iLTMClassifier;torch.set_num_threads(4);torch.set_num_interop_threads(4)
        stage('load_official_full_training_labels');x,y,q,passengers,train_passengers=load_full_inputs(frozen)
        record.update(training_ids_sha256=full.array_hash(train_passengers.astype(str)),query_ids_sha256=full.array_hash(passengers.astype(str)),
            labels_sha256=full.array_hash(y),training_input_sha256=full.frame_hash(x),query_input_sha256=full.frame_hash(q))
        expected=full.expected_parameters(iLTMClassifier)
        with threadpool_limits(limits=4):
            stage('construct_native8')
            model=iLTMClassifier(device='cpu',n_ensemble=8,seed=full.SEED,checkpoint=str(full.WEIGHT),
                cat_features=full.CATEGORICAL,**full.OVERRIDES,**full.FAIL_CLOSED)
            full.check_model(model,expected);record['configuration']=expected
            record['constructed_cache_entries']=len(native_class._model_cache)
            observer=full.NativeObservation(full.VENDOR/'iltm',lambda:stage('native_training_progress'))
            record['native_observation']=observer.records;stage('fit_all8693');tick=time.monotonic()
            old_profile=sys.getprofile();sys.setprofile(observer);observing=True
            try:model.fit(x,y,eval_set=None,fit_max_time=None,return_partial_on_timeout=False)
            finally:sys.setprofile(old_profile);observing=False
            record['fit_seconds']=time.monotonic()-tick
            if observer.error:raise RuntimeError('Native completion/flag observation failed: '+observer.error)
            full.check_model(model,expected,True)
            record.update(predictor_count=len(model.predictors_),preprocessor_count=len(model.preprocessors_),
                inference_retrieval_rows=[len(p['y_ctxt_superset']) for p in model.predictors_],
                postfit_cache_entries=len(native_class._model_cache),postfit_engine_released=model._model is None)
            full.validate_native_completion(record)
            record['hit_epoch_cap']=[row['epochs_entered']==2048 for row in observer.records['finetuning']]
            stage('predict_all4277');first=full.probabilities(model,q);full.check_model(model,expected,True)
            stage('save_portable');joblib.dump(model,OUT/'model.joblib')
            reference=weakref.ref(model);model._release_training_model();model=None;native_class._model_cache.clear();gc.collect()
            if reference() is not None:raise RuntimeError('Original estimator retained across reload.')
            stage('reload_portable');restored=joblib.load(OUT/'model.joblib');full.check_model(restored,expected,True)
            if restored._model is not None or native_class._model_cache:raise RuntimeError('Portable model retained generator.')
            full.tensor_inventory(restored.predictors_)  # CPU only; no alias-layout equality requirement.
            second=full.probabilities(restored,q);difference=full.compare(first,second,2e-6)
            full.check_model(restored,expected,True)
            if native_class._model_cache:raise RuntimeError('Reload inference recreated generator cache.')
            stage('reload_verified');np.savez(OUT/'predictions.npz',first=first,reloaded=second,passenger_ids=passengers)
            pd.DataFrame({'PassengerId':passengers,'Transported':second[:,1]>=.5}).to_csv(OUT/'submission.csv',index=False)
            record.update(status='worker_complete',rows=4277,export_reload_inference_verified=True,
                reload_max_absolute_difference=difference,threshold_equal=True,model_sha256=sha(OUT/'model.joblib'),
                sha256=sha(OUT/'submission.csv'),prediction_cache_sha256=sha(OUT/'predictions.npz'),
                openmp_runtimes=[{k:v for k,v in item.items() if k in ['filepath','num_threads','prefix','internal_api']}
                    for item in threadpool_info() if item.get('user_api')=='openmp'])
            assert_frozen(frozen);stage('final_identity')
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
    finally:
        if observing:sys.setprofile(old_profile)
        try:
            model=restored=None
            if native_class is not None:native_class._model_cache.clear()
            gc.collect();bridge.require_false('released');check_deadline(deadline);record['release_complete']=True
        except BaseException as cleanup:
            record.update(status='failed_inconclusive',cleanup_error=str(cleanup))
            record.setdefault('error_type',type(cleanup).__name__);record.setdefault('error',str(cleanup))
        record['monotonic_seconds']=time.monotonic()-started
        if record['status']!='worker_complete':remove_uploadable()
        write(OUT/'worker_result.json',record)
    return record


def remove_uploadable():
    for directory in [OUT,REPORT]:
        for name in ['submission.csv','manifest.json','selection.json','authorization.json']:
            path=directory/name
            if path.exists():path.unlink()


def verify_completed(record):
    if (record.get('status')!='worker_complete' or record.get('release_complete') is not True
            or record.get('training_rows')!=8693 or record.get('rows')!=4277
            or record.get('export_reload_inference_verified') is not True):
        raise RuntimeError('Complete full export required: '+str(record.get('error_type'))+': '+str(record.get('error')))
    full.validate_native_completion(record)
    import numpy as np
    import pandas as pd
    if sha(OUT/'model.joblib')!=record['model_sha256'] or sha(OUT/'submission.csv')!=record['sha256']:
        raise RuntimeError('Prepared model/CSV changed.')
    if sha(OUT/'predictions.npz')!=record['prediction_cache_sha256']:raise RuntimeError('Reload proof changed.')
    with np.load(OUT/'predictions.npz',allow_pickle=False) as saved:
        difference=full.compare(saved['first'],saved['reloaded'],2e-6)
        frame=pd.read_csv(OUT/'submission.csv');test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
        if (list(frame.columns)!=['PassengerId','Transported'] or len(frame)!=4277 or frame.isna().any().any()
                or not np.array_equal(frame.PassengerId.astype(str),test.PassengerId.astype(str))
                or not np.array_equal(frame.Transported.to_numpy(),saved['reloaded'][:,1]>=.5)):
            raise RuntimeError('Complete aligned boolean submission required.')
    if difference!=record['reload_max_absolute_difference']:raise RuntimeError('Reload difference metadata differs.')


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('One full preparation; no overwrite/resume/retry.')
    started=time.monotonic();deadline=started+BUDGET
    env=bridge.entry_environment();os.environ[bridge.FLAG]='FALSE'
    frozen=contract();check_deadline(deadline)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True);process=None
    for folder in [OUT,REPORT]:write(folder/'protocol.json',frozen)
    try:
        with (OUT/'native.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--deadline',str(deadline)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            peak=full.supervise(process,deadline,OUT)
        record=read(OUT/'worker_result.json');verify_completed(record);assert_frozen(frozen);check_deadline(deadline)
        duplicates=[str(p.relative_to(ROOT)) for p in (ROOT/'reports').rglob('online_attempts/*.json')
                    if read(p).get('file_sha256')==record['sha256']]
        if duplicates:raise RuntimeError('Identical CSV already attempted: '+str(duplicates))
        selection=dict(version=VERSION,proposal=PROPOSAL,mode='online_first',threshold=.5,training_rows=8693,
            model_seed=full.SEED,n_ensemble=8,cv_gate_required=False,policy_path=POLICY,policy_sha256=frozen['policy_sha256'])
        manifest=dict(status='complete',version=VERSION,proposal=PROPOSAL,rows=4277,training_rows=8693,
            sha256=record['sha256'],model_sha256=record['model_sha256'],export_reload_inference_verified=True,
            reload_max_absolute_difference=record['reload_max_absolute_difference'],threshold_equal=True,
            source_commit=frozen['source_commit'],source_hashes=frozen['source_hashes'],selection=selection,
            provenance=dict(source_hashes=frozen['source_hashes'],source_commit=frozen['source_commit'],policy_path=POLICY,
                policy_sha256=frozen['policy_sha256'],input_hashes=frozen['input_hashes']),
            sampled_tree_rss_highwater_bytes=peak,monotonic_seconds=time.monotonic()-started,
            no_cv_performed=True,online_submission_performed=False)
        for folder in [OUT,REPORT]:
            write(folder/'selection.json',selection);write(folder/'manifest.json',manifest)
        for name in ['training_log.json','worker_result.json','external_monitor.json']:shutil.copy2(OUT/name,REPORT/name)
        check_deadline(deadline);return manifest
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:full.terminate(process,full.process_tree(process.pid)['pids'])
            except BaseException:pass
        remove_uploadable()
        failure=dict(status='failed_inconclusive',version=VERSION,error_type=type(error).__name__,error=str(error),
            monotonic_seconds=time.monotonic()-started,no_uploadable_output=True)
        for name in ['worker_result.json','training_log.json','external_monitor.json']:
            if (OUT/name).exists():
                shutil.copy2(OUT/name,REPORT/name)
                if name=='worker_result.json':failure['worker_error']=read(OUT/name)
        for folder in [OUT,REPORT]:write(folder/'failure.json',failure)
        return failure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:
        if args.deadline is None:parser.error('Parent deadline required.')
        worker(args.deadline)
    else:print(__import__('json').dumps(run(),indent=2))
