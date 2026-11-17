"""Build v101 portable two-environment rawMitra + fixed LightGBM residual package.

No upload. Full-training OOF is a corrector training input, never scored.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
POLICY = ROOT/'reports/error_targeted_20261117/residual_export_policy.json'
PROPOSAL = 'proposal_ready_nonlinear_residual_v2'
OUT = ROOT/'artifacts'/PROPOSAL
REPORT = ROOT/'reports'/PROPOSAL
VERSION = 'v101_rawMitra_nonlinear_residual_ready'
SEED = 20261112
BUDGET = 3600
PYTHON = {'gpu': ROOT/'.venv-mitra/bin/python', 'cpu': ROOT/'.venv/bin/python'}
RAW_FILES = ['raw_model.joblib','raw_probabilities.npz','raw_reload_probabilities.npz','raw_metadata.json']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def environment_lock_bytes():
    import importlib.metadata
    return ''.join(f'{name}=={version}\n' for name,version in sorted(
        (dist.metadata['Name'],dist.version) for dist in importlib.metadata.distributions())).encode()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sources():
    pending=['prepare_residual_ready.py','mitra_candidate.py','nonlinear_error_corrector.py']
    seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import)
                     else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                path=module.replace('.','/')+'.py'
                if (ROOT/path).is_file() and path not in seen:pending.append(path)
    seen.update(str(p.relative_to(ROOT)) for p in (ROOT/'third_party/mitra').rglob('*.py'))
    seen.add(str(POLICY.relative_to(ROOT)))
    return {name:sha(ROOT/name) for name in sorted(seen)}


def check_frozen():
    frozen=read(OUT/'frozen.json')
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes'],**frozen['evidence_files']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    return frozen


def runtime_env(kind):
    env=dict(os.environ);env['KMP_DUPLICATE_LIB_OK']='FALSE'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[key]='2' if kind=='cpu' else '4'
    env['PYTHONHASHSEED']=str(SEED)
    if kind=='cpu' and sys.platform=='darwin':
        library=ROOT/'.venv/lib'
        libraries=list(library.glob('python*/site-packages/sklearn/.dylibs/libomp.dylib'))
        if len(libraries)!=1:raise RuntimeError('Existing sklearn libomp required; no install/fallback.')
        env['DYLD_FALLBACK_LIBRARY_PATH']=str(libraries[0].parent)
    return env


def run_child(stage,deadline):
    kind='gpu' if stage.startswith('gpu') else 'cpu'
    with (OUT/(stage+'.log')).open('w') as log:
        child=subprocess.Popen([str(PYTHON[kind]),str(Path(__file__).resolve()),'--worker',stage],
            cwd=ROOT,env=runtime_env(kind),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=max(.01,deadline-time.monotonic()))
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            child.wait();raise
    if code:raise RuntimeError(f'{stage} failed with exit {code}; see preserved stage log.')
    result=read(OUT/(stage+'_result.json'))
    if result.get('status')!='complete':raise RuntimeError('Incomplete stage '+stage)
    check_frozen()
    return result


def load_frames():
    import numpy as np
    import pandas as pd
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    if len(train)!=8693 or len(test)!=4277 or 'Transported' in test:raise RuntimeError('Unexpected official roster.')
    if not train.Transported.isin([True,False]).all():raise ValueError('Nonbinary full-training labels.')
    y=train.Transported.to_numpy(dtype=np.int64)
    roster=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
    from mitra_candidate import COMPACT,CATEGORICAL
    x=roster[[c for c in COMPACT if c not in ['Deck','CabinNumber','Side']]].copy()
    cabin=roster.Cabin.str.split('/',expand=True).reindex(columns=[0,1,2])
    x['Deck']=cabin[0];x['CabinNumber']=pd.to_numeric(cabin[1],errors='coerce').astype(float);x['Side']=cabin[2]
    for col in CATEGORICAL:x[col]=x[col].fillna('__MISSING__').astype(str)
    return train,test,y,x[COMPACT]


def probability(p,n):
    import numpy as np
    p=np.asarray(p)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities.')
    return p


def gpu_stage(stage):
    import joblib
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold
    import mitra_candidate as base
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback.')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
    train,test,y,x=load_frames();ids=np.arange(8693,dtype=np.int64)
    query=x.iloc[8693:];test_ids=test.PassengerId.astype(str).to_numpy(dtype=str)
    if stage=='gpu_fit':
        oof=np.full(8693,np.nan,dtype=np.float32);fold_ids=np.full(8693,-1,dtype=np.int8)
        for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(ids,y)):
            print(f'raw OOF fold {fold+1}/3: {len(a)} support, {len(b)} query; no score',flush=True)
            model=base.fit_mitra(x.iloc[a],y[a],ids[a],SEED+fold,'mps','zero-shot',
                query_chunk=2318,log_path=OUT/f'raw_oof_fold{fold}.json')
            oof[b]=probability(model.predict_proba(x.iloc[b])[:,1],len(b));fold_ids[b]=fold
            np.savez_compressed(OUT/f'raw_oof_fold{fold}_rows.npz',train_ids=a,query_ids=b)
            model.release();del model;base.release_memory()
        if not np.isfinite(oof).all() or (fold_ids<0).any():raise RuntimeError('Incomplete training OOF.')
        print('Raw final fit uses all 8693 support rows',flush=True)
        model=base.fit_mitra(x.iloc[:8693],y,ids,SEED,'mps','zero-shot',query_chunk=2318,
            log_path=OUT/'raw_full_fit.json')
        p=probability(model.predict_proba(query)[:,1],4277)
        metadata=dict(model.metadata,passenger_ids=train.PassengerId.astype(str).tolist(),
            test_passenger_ids=test_ids.tolist(),support_sha256=hashlib.sha256(model.support.tobytes()).hexdigest(),
            compact_columns=list(model.preprocessor.columns),query_rows=4277)
        joblib.dump(model,OUT/'raw_model.joblib');write(OUT/'raw_metadata.json',metadata)
        np.savez_compressed(OUT/'raw_probabilities.npz',oof_raw=oof,test_raw=p,train_ids=ids,
            train_passenger_ids=train.PassengerId.astype(str).to_numpy(dtype=str),test_ids=test_ids,fold_ids=fold_ids)
        model.release();del model;base.release_memory()
        return dict(status='complete',training_oof_rows=8693,training_oof_scored=False,training_rows=8693,rows=4277)
    model=joblib.load(OUT/'raw_model.joblib')
    if (model.device!='cpu' or model._engine is not None or model.metadata['recipe']!='zero-shot'
            or model.metadata['seed']!=SEED or len(model.support)!=8693):raise RuntimeError('Nonportable or wrong raw model.')
    np.testing.assert_array_equal(model.labels,y)
    np.testing.assert_array_equal(model.support,model.preprocessor.transform(x.iloc[:8693]))
    model.device='mps';p=probability(model.predict_proba(query)[:,1],4277)
    with np.load(OUT/'raw_probabilities.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_ids'],test_ids)
        np.testing.assert_array_equal(saved['test_raw'],p)
    np.savez_compressed(OUT/'raw_reload_probabilities.npz',test_raw=p,test_ids=test_ids)
    model.release();del model;base.release_memory()
    return dict(status='complete',rows=4277,raw_reload_max_abs_diff=0.,bitwise_equal=True)


def cpu_stage(stage):
    import joblib
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from scipy.special import expit
    import nonlinear_error_corrector as residual
    if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Explicit FALSE runtime required.')
    residual.check_protocol()
    x=residual.features();train=x.iloc[:8693];query=x.iloc[8693:]
    y=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['Transported']).Transported.to_numpy(dtype=np.int64)
    with np.load(OUT/'raw_probabilities.npz',allow_pickle=False) as saved:
        oof=probability(saved['oof_raw'].copy(),8693);raw=probability(saved['test_raw'].copy(),4277)
        test_ids=saved['test_ids'].copy();np.testing.assert_array_equal(saved['train_ids'],np.arange(8693))
    if stage=='cpu_fit':
        train,query,vocab=residual.categorical_frames(train,query)
        dataset=lgb.Dataset(train,label=y,init_score=residual.relational.logit(oof),
            categorical_feature=residual.CATS,free_raw_data=True)
        booster=lgb.train(dict(residual.PARAMS,seed=SEED),dataset,num_boost_round=residual.ROUNDS)
        booster.save_model(str(OUT/'residual_booster.txt'))
        write(OUT/'residual_schema.json',dict(columns=residual.COLUMNS,categorical_columns=residual.CATS,
            vocabulary=vocab,parameters=dict(residual.PARAMS,seed=SEED),rounds=residual.ROUNDS))
        p=probability(expit(residual.relational.logit(raw)+booster.predict(query,raw_score=True)),4277)
        np.savez_compressed(OUT/'residual_probabilities.npz',prob=p,raw=raw,test_ids=test_ids)
        return dict(status='complete',training_rows=8693,rounds=residual.ROUNDS,rows=4277,training_oof_scored=False)
    schema=read(OUT/'residual_schema.json')
    if schema['columns']!=residual.COLUMNS or schema['parameters']!=dict(residual.PARAMS,seed=SEED):raise RuntimeError('Residual schema drift.')
    _,query,_=residual.categorical_frames(train,query,schema['vocabulary'])
    booster=lgb.Booster(model_file=str(OUT/'residual_booster.txt'))
    with np.load(OUT/'raw_reload_probabilities.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_ids'],test_ids);reloaded_raw=saved['test_raw'].copy()
    p=probability(expit(residual.relational.logit(reloaded_raw)+booster.predict(query,raw_score=True)),4277)
    with np.load(OUT/'residual_probabilities.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['test_ids'],test_ids);np.testing.assert_array_equal(saved['prob'],p)
    pd.DataFrame(dict(PassengerId=test_ids,probability=p,reload_probability=p)).to_csv(OUT/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test_ids,Transported=p>=.5)).to_csv(OUT/'submission.csv',index=False)
    from kaggle_workflow import validate_submission
    validation=validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')
    reject_duplicate_submission(validation['sha256'])
    # Portable package descriptor: base and corrector live in separately authenticated runtimes.
    package=dict(format='two_environment_pipeline_v1',version=VERSION,
        files={name:sha(OUT/name) for name in ['raw_model.joblib','residual_booster.txt','residual_schema.json']},
        inference='rawMitra(compact13) logit + LightGBM(engineered_no_ids30) raw_score',epsilon=1e-7,
        raw_runtime='.venv-mitra',corrector_runtime='.venv',threshold=.5)
    joblib.dump(package,OUT/'model.joblib')
    if joblib.load(OUT/'model.joblib')!=package:raise RuntimeError('Package descriptor reload mismatch.')
    return dict(status='complete',rows=4277,reload_max_absolute_difference=0.,bitwise_equal=True,submission_validation=validation,
                package_kind='descriptor_plus_authenticated_portable_components',component_files=package['files'])


def reject_duplicate_submission(digest):
    for path in (ROOT/'reports').rglob('online_attempts/*.json'):
        if read(path).get('file_sha256')==digest:
            raise RuntimeError('Identical CSV already attempted: '+str(path.relative_to(ROOT)))


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('Fresh proposal required; no overwrite/resume.')
    plan=read(POLICY)
    if plan['version']!=VERSION or plan['budget_seconds']!=BUDGET or plan['training_oof_scored'] is not False:
        raise RuntimeError('Wrong export policy.')
    code=sources()
    for name,digest in code.items():
        committed=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest()!=digest:raise RuntimeError('Commit source before full export: '+name)
    evidence={p:sha(ROOT/p) for p in plan['evidence_files']}
    if evidence!=plan['evidence_files']:raise RuntimeError('Export evidence changed.')
    data={f'data/raw/{n}.csv':sha(ROOT/f'data/raw/{n}.csv') for n in ['train','test']}
    for name in ['model.safetensors','config.json']:
        path='artifacts/pretrained/mitra-v2/'+name;data[path]=sha(ROOT/path)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    started=time.monotonic();deadline=started+BUDGET
    try:
        environments={}
        for kind,python in PYTHON.items():
            output=subprocess.check_output([str(python),'-c',
                'import sys; from prepare_residual_ready import environment_lock_bytes; sys.stdout.buffer.write(environment_lock_bytes())'],cwd=ROOT,env=runtime_env(kind),
                timeout=max(.01,deadline-time.monotonic()))
            path=OUT/f'requirements-{kind}.lock.txt';path.write_bytes(output);environments[kind]=sha(path)
        frozen=dict(source_hashes=code,data_hashes=data,evidence_files=evidence,environment_hashes=environments,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(OUT/'frozen.json',frozen);write(REPORT/'frozen.json',frozen)
        stages={}
        for stage in ['gpu_fit','cpu_fit','gpu_reload','cpu_reload']:
            print(stage,flush=True);stages[stage]=run_child(stage,deadline)
        if time.monotonic()>deadline:raise TimeoutError('Complete export exceeded 3600 seconds.')
        check_frozen()
        validation=stages['cpu_reload']['submission_validation']
        if validation!={'rows':4277,'sha256':sha(OUT/'submission.csv'),'valid':True}:
            raise RuntimeError('Validated submission changed before publication.')
        reject_duplicate_submission(validation['sha256'])
        component=dict(files={n:sha(OUT/n) for n in RAW_FILES},training_rows=8693,rows=4277,raw_reload_max_abs_diff=0.)
        selection=dict(version=VERSION,threshold=.5,stage='full_training_export_not_new_validation',S_supported=False)
        manifest=dict(status='complete',version=VERSION,proposal=PROPOSAL,rows=4277,training_rows=8693,
            fit_rows=8693,threshold=.5,valid=True,submission_validation=validation,sha256=sha(OUT/'submission.csv'),model_sha256=sha(OUT/'model.joblib'),
            test_probabilities_sha256=sha(OUT/'test_probabilities.csv'),selection=selection,
            export_reload_inference_verified=True,reload_rows=4277,reload_max_absolute_difference=0.,
            reload_threshold_decisions_equal=True,raw_component=component,stages=stages,
            component_files=stages['cpu_reload']['component_files'],artifact_kind='composite_mixed_runtime',model_artifact_kind='descriptor_plus_authenticated_portable_components',
            source_hashes=code,source_commit=frozen['source_commit'],environment_hashes=environments,
            training_oof_scored=False,policy_path=str(POLICY.relative_to(ROOT)),policy_sha256=sha(POLICY),
            provenance=dict(source_hashes=code,evidence_files=evidence,decision_path=str(POLICY.relative_to(ROOT)),decision_sha256=sha(POLICY)),
            seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
            upload_authorized_by_this_export=False)
        for name,value in [('selection.json',selection),('manifest.json',manifest),('reload_metadata.json',stages['cpu_reload'])]:
            write(OUT/name,value);write(REPORT/name,value)
    except BaseException as error:
        for folder in [OUT,REPORT]:
            for name in ['manifest.json','selection.json','submission.csv']:(folder/name).unlink(missing_ok=True)
            write(folder/'failure.json',dict(status='failed_not_ready',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['gpu_fit','cpu_fit','gpu_reload','cpu_reload'])
    args=parser.parse_args()
    if args.worker:
        if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Worker requires startup FALSE.')
        frozen=check_frozen()
        kind='gpu' if args.worker.startswith('gpu') else 'cpu'
        current=environment_lock_bytes()
        if hashlib.sha256(current).hexdigest()!=frozen['environment_hashes'][kind]:
            raise RuntimeError('Worker environment differs from frozen runtime.')
        result=gpu_stage(args.worker) if args.worker.startswith('gpu') else cpu_stage(args.worker)
        check_frozen();write(OUT/(args.worker+'_result.json'),result)
    else:run()


if __name__=='__main__':main()
