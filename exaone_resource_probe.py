"""Frozen EXAONE CPU equivalence and full eight-view MPS resource proof; no scores."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

import exaone_candidate as core

ROOT=Path(__file__).resolve().parent
DESIGN='reports/exaone_resource_design.json'
DESIGN_SHA='d33b5d46e55b515567eae1ee4158fd9490d5029d2f9ad7fbf0ddbd18081b934f'
OWN=['exaone_candidate.py','exaone_mps_compat.py','exaone_resource_probe.py',
    'test_exaone_candidate.py','test_exaone_mps_compat.py','test_exaone_resource_probe.py',
    'requirements-exaone.txt','requirements-exaone.lock.txt',DESIGN,'quota_guard.py']


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def check_quota(session):
    from quota_guard import latest_weekly,assess
    try:allowed=assess(latest_weekly(session),reserve=30)['can_start_more_work']
    except (OSError,ValueError,TypeError,KeyError):allowed=False
    if not allowed:raise RuntimeError('Fresh weekly reserve guard did not allow this stage.')


def contract(require_committed):
    if core.sha(ROOT/DESIGN)!=DESIGN_SHA:raise RuntimeError('Frozen EXAONE resource design changed.')
    manifest=core.compat.verify_vendor();core.verify_checkpoint()
    for line in (ROOT/'requirements-exaone.lock.txt').read_text().splitlines():
        name,version=line.split('==')
        if importlib.metadata.version(name)!=version:raise RuntimeError('Independent dependency differs: '+name)
    names=OWN+['third_party/exaone/SOURCE.json']+['third_party/exaone/'+name for name in manifest['files_sha256']]
    sources={name:core.sha(ROOT/name) for name in sorted(set(names))}
    if require_committed:
        tracked=subprocess.check_output(['git','ls-files','-z','--',*sources],cwd=ROOT).decode()
        if set(filter(None,tracked.split('\0')))!=set(sources) or subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=ROOT).strip():
            raise RuntimeError('Every design/source/environment pin must be committed before MPS.')
    lock=core.environment_lock()
    return dict(sources=sources,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        checkpoint_sha256=core.WEIGHT_SHA,checkpoint_bytes=core.CHECKPOINT.stat().st_size,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        settings=core.SETTINGS,design_sha256=DESIGN_SHA),lock


def assert_frozen(frozen):
    for name,digest in frozen['sources'].items():
        if core.sha(ROOT/name)!=digest:raise RuntimeError('Source changed during resource proof: '+name)
    if hashlib.sha256(core.environment_lock().encode()).hexdigest()!=frozen['environment_sha256']:
        raise RuntimeError('Independent environment changed during resource proof.')
    core.verify_checkpoint()


def fixture(rows,query_rows):
    def frame(n,seed):
        rng=np.random.default_rng(seed)
        x=pd.DataFrame({col:rng.lognormal(2.,1.,n) for col in core.COMPACT})
        for col in core.CATEGORICAL:x[col]=np.asarray(['A','B','C'],dtype=object)[rng.integers(3,size=n)]
        x.loc[::19,'Age']=np.nan;x.loc[::23,'HomePlanet']=None
        return x[core.COMPACT]
    support=frame(rows,20261117);query=frame(query_rows,20261118)
    query.loc[0,'HomePlanet']='unseen_query_category'
    return support,np.arange(rows,dtype=np.int64)%2,query


def compare(actual,expected,tolerance):
    a=core.valid_probabilities(actual,len(expected));b=core.valid_probabilities(expected,len(expected))
    diff=float(np.max(np.abs(a-b)))
    if diff>tolerance or not np.array_equal(a[:,1]>=.5,b[:,1]>=.5):
        raise RuntimeError('Frozen probability/threshold equivalence gate failed: '+str(diff))
    return diff


def snapshot(device):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(process_max_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),device_measurement='snapshots_not_true_peak')
    if device=='mps':
        core.torch.mps.synchronize()
        result.update(mps_current_bytes=core.torch.mps.current_allocated_memory(),mps_driver_bytes=core.torch.mps.driver_allocated_memory(),
            mps_recommended_max_bytes=core.torch.mps.recommended_max_memory())
    return result


def model_description(classifier):
    from safetensors.torch import load_file
    loaded=load_file(str(core.CHECKPOINT),device='cpu');state=classifier.model.state_dict()
    if set(loaded)!=set(state):raise RuntimeError('Strict loaded state keys differ.')
    if any(not core.torch.equal(state[k].detach().cpu(),loaded[k].to(dtype=state[k].dtype)) for k in state):
        raise RuntimeError('Native loader did not preserve checkpoint values with fixed FP16 cast.')
    return dict(manifest=classifier.manifest.to_dict(),native_strict_state_keys_shapes_values=True,
        parameters=sum(v.numel() for v in classifier.model.parameters()),buffers=sum(v.numel() for v in classifier.model.buffers()),
        state_keys=len(state),parameter_dtypes=sorted(set(str(v.dtype) for v in classifier.model.parameters())),
        module_types=dict(Counter(type(module).__name__ for module in classifier.model.modules())),
        state_shapes={k:list(v.shape) for k,v in state.items()})


def cpu_stages():
    x,y,q=fixture(128,33);encoder=core.CompactEncoder().fit(x);encoded=encoder.transform(x);query=encoder.transform(q)
    native=local=None;started=time.monotonic()
    try:
        native=core.new_classifier(local=False,device='cpu');description=model_description(native)
        native.fit(encoded,y,categorical_mask=core.MASK)
        original=core.valid_probabilities(native.predict_proba(query),len(q))
        identity=core.fitted_identity(native,len(x))
        native=None;gc.collect()
        local=core.fit_context(x,y,np.arange(len(x)),device='cpu')
        adapted=local.predict_proba(q)
        if local.metadata['preprocessing']!=identity:raise RuntimeError('Adapter changed original CPU preprocessing/RNG/views.')
        difference=compare(adapted,original,.002)
        partition=np.concatenate([local.predict_proba(q.iloc[:16]),local.predict_proba(q.iloc[16:])])
        partition_difference=compare(partition,adapted,.002)
        with tempfile.TemporaryDirectory(prefix='exaone_cpu_reload_') as directory:
            portable=roundtrip(local,q,adapted,directory,'cpu')
        record=dict(stage='AB_native_and_local_CPU',status='complete',support_rows=128,query_rows=33,views=8,
            same_native_cpu_preprocessing=True,original_vs_adapter_max_difference=difference,
            partition_max_difference=partition_difference,portable_reload=portable,model=description,preprocessing=identity,
            input_support_sha256=core.array_hash(encoded),input_query_sha256=core.array_hash(query),
            label_sha256=core.array_hash(y),query_ids_sha256=core.array_hash(np.arange(33)),
            native_probability_sha256=core.array_hash(original),adapter_probability_sha256=core.array_hash(adapted),
            seconds=time.monotonic()-started,resources=snapshot('cpu'))
        return record,adapted
    finally:
        if local is not None:local.release()
        native=local=None;gc.collect()


def roundtrip(predictor,query,expected,directory,device):
    path=Path(directory)/'portable.joblib'
    if path.exists():raise RuntimeError('Portable temporary artifact exists.')
    identity=dict(predictor.metadata['preprocessing']);support=predictor.support.copy();labels=predictor.labels.copy()
    joblib.dump(predictor,path);predictor.release()
    # Only the caller can discard its own reference; engine must already be gone.
    if predictor._engine is not None:raise RuntimeError('Device engine was not released.')
    restored=None
    try:
        restored=joblib.load(path)
        if restored.device!='cpu' or restored._engine is not None:raise RuntimeError('Portable context must load on CPU without engines.')
        np.testing.assert_array_equal(restored.support,support);np.testing.assert_array_equal(restored.labels,labels)
        restored.device=device;actual=restored.predict_proba(query)
        if restored.metadata['preprocessing']!=identity:raise RuntimeError('Reload changed native schema/view/CPU RNG state.')
        return dict(max_difference=compare(actual,expected,2e-6),query_rows=len(query),full_support_rows=len(labels),
            same_threshold_decisions=True,restored_cpu_context=True,preprocessing_exact=True,
            forward_records=restored.metadata['last_prediction']['forward_records'])
    finally:
        if restored is not None:restored.release()
        restored=None;gc.collect();path.unlink(missing_ok=True)


def mps_case(rows,queries,out,*,cpu_reference=None,repeat=False,partition=False,reload=False):
    x,y,q=fixture(rows,queries);predictor=None;started=time.monotonic()
    try:
        before=snapshot('mps');predictor=core.fit_context(x,y,np.arange(rows),device='mps')
        initial=core.valid_probabilities(predictor.predict_proba(q),queries)
        record=dict(status='complete',support_rows=rows,query_rows=queries,views=8,full_support=True,
            preprocessing=predictor.metadata['preprocessing'],initial_forward_records=predictor.metadata['last_prediction']['forward_records'],
            probabilities_sha256=core.array_hash(initial),input_support_sha256=core.array_hash(predictor.support),
            input_query_sha256=core.array_hash(predictor.encoder.transform(q)),label_sha256=core.array_hash(y),
            resources_before=before,resources_after_forward=snapshot('mps'))
        if cpu_reference is not None:record['cpu_vs_mps_max_difference']=compare(initial,cpu_reference,.002)
        if repeat:record['repeat_max_difference']=compare(predictor.predict_proba(q),initial,2e-6)
        if partition:
            other=np.concatenate([predictor.predict_proba(q.iloc[:16]),predictor.predict_proba(q.iloc[16:])])
            record['partition_max_difference']=compare(other,initial,.002)
        if reload:record['portable_reload']=roundtrip(predictor,q,initial,out,'mps')
        core.check_deadline();record.update(seconds=time.monotonic()-started,resources_after=snapshot('mps'))
        return record
    finally:
        if predictor is not None:predictor.release()
        predictor=None;gc.collect()


def run(mode,session_file):
    if mode not in ['cpu','mps']:raise ValueError('Only cpu proof or full fixed MPS resource mode.')
    out=ROOT/f'artifacts/probes/exaone_{mode}';report=ROOT/f'reports/exaone_resources/{mode}'
    if out.exists() or report.exists():raise RuntimeError('Resource destination exists; no overwrite/resume.')
    check_quota(session_file);frozen,lock=contract(require_committed=mode=='mps')
    out.mkdir(parents=True);report.mkdir(parents=True)
    result=dict(frozen,status='incomplete',mode=mode,budget_seconds=2400,stages=[],accuracy_computed=False,
        all_data_synthetic=True,resource_qualification=False,backend='native_default_device_SDPA_in_locked_PyTorch')
    for folder in [out,report]:(folder/'environment.lock.txt').write_text(lock)
    for name in frozen['sources']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    started=time.monotonic();previous=signal.getsignal(signal.SIGALRM);old_deadline=core.DEADLINE
    def timeout(*_):raise TimeoutError('Fixed2400s full EXAONE resource budget exhausted.')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,2400);core.DEADLINE=started+2400
    try:
        core.torch.set_num_threads(4)
        assert_frozen(frozen);cpu,cpu_p=cpu_stages();result['stages'].append(cpu)
        np.savez(out/'cpu_reference.npz',prob=cpu_p)
        for folder in [out,report]:write(folder/'progress.json',result)
        if mode=='mps':
            if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):raise RuntimeError('Implicit CPU fallback forbidden.')
            if not core.torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
            core.torch.mps.set_per_process_memory_fraction(.65)
            for stage,rows,queries,options in [
                ('C_tiny',128,33,dict(cpu_reference=cpu_p,repeat=True,partition=True,reload=True)),
                ('D_screen',4636,2318,{}),('E_full_support',8693,2318,{}),
                ('F_full_export',8693,4277,dict(reload=True))]:
                assert_frozen(frozen);core.check_deadline()
                record=mps_case(rows,queries,out,**options);record['stage']=stage;result['stages'].append(record)
                for folder in [out,report]:write(folder/'progress.json',result)
                print(json.dumps(dict(stage=stage,seconds=record['seconds'],support_rows=rows,query_rows=queries)),flush=True)
        assert_frozen(frozen);core.check_deadline()
        result.update(status='complete' if mode=='mps' else 'cpu_complete_not_MPS_qualified',resource_qualification=mode=='mps')
        return result
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),resource_qualification=False);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);core.DEADLINE=old_deadline
        result['seconds']=time.monotonic()-started
        for folder in [out,report]:write(folder/'result.json',result);write(folder/'progress.json',result)
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['cpu','mps'])
    parser.add_argument('--session-file',type=Path,required=True)
    args=parser.parse_args();run(args.mode,args.session_file)
