"""Bounded synthetic CPU one-member iLTM feasibility; never a scored candidate.

Native imports occur only after a fresh child receives its isolated runtime env.
Read-only profiling observes scalars/shapes, retains no frames/tensors, and its
cost is included. Two complete cases permit only a separate full8 qualification.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import weakref

ROOT=Path(__file__).resolve().parent
PLAN='reports/iltm_resource_design.json'
PLAN_SHA='38c5558595119fc0244f2a57b66978da7a1b889c2fab32ac1bd45f5c2f2c1d5d'
VENDOR=ROOT/'third_party/iltm'
WEIGHT=ROOT/'artifacts/pretrained/iltm/xgbrconcat.pth'
OUTPUT=ROOT/'artifacts/probes/iltm_native_cpu1'
REPORT=ROOT/'reports/iltm_native_cpu1_resources'
CASES=[(4636,2318),(8693,4277)]
BUDGET,LIMIT,SEED=1200,10*1024**3,20261117
COLUMNS=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CATEGORICAL=[0,1,2,4,10,12]
FAIL_CLOSED=dict(adaptive_memory=False,gpu_oom_retries=0,auto_stop_on_low_cpu_memory=False,
                 auto_amp_on_low_mem=False,auto_disable_retrieval_on_low_mem=False)
OWN=['iltm_resource_probe.py','test_iltm_resource_probe.py','quota_guard.py',PLAN,
     'reports/iltm_setup.json','requirements-iltm.lock.txt','third_party/iltm/SOURCE.json']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def require_committed(names):
    for name in names:
        saved=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(saved).hexdigest()!=sha(ROOT/name):raise RuntimeError('Uncommitted/changed source: '+name)
def environment():
    normalize=lambda name:name.lower().replace('_','-').replace('.','-')
    return {normalize(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}
def quota(session_file):
    from quota_guard import latest_weekly,assess
    if not assess(latest_weekly(session_file),reserve=30)['can_start_more_work']:
        raise RuntimeError('Fresh reserve30 quota guard did not pass.')
def deadline_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Shared1200s monotonic deadline exceeded.')


def contract():
    if sha(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Frozen iLTM resource design changed.')
    plan=read(ROOT/PLAN);setup=read(ROOT/'reports/iltm_setup.json');manifest=read(VENDOR/'SOURCE.json')
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-iltm').resolve() or sys.version.split()[0]!=setup['environment']['python']:
        raise RuntimeError('Exact isolated .venv-iltm Python required.')
    for name,digest in plan['setup_evidence'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Setup evidence changed: '+name)
    normalize=lambda name:name.lower().replace('_','-').replace('.','-')
    expected={normalize(k):v for k,v in setup['environment']['installed_distributions'].items()}
    if environment()!=expected:raise RuntimeError('Pinned installed environment differs.')
    sources={name:sha(ROOT/name) for name in OWN}
    for name,digest in manifest['source_files_sha256'].items():
        path=VENDOR/name
        if sha(path)!=digest:raise RuntimeError('Pinned original vendor changed: '+name)
        sources[str(path.relative_to(ROOT))]=digest
    if (manifest['commit']!='1e644b5420ccc2fe3f78a3b62272a3030e074af4'
            or WEIGHT.stat().st_size!=plan['checkpoint_bytes'] or sha(WEIGHT)!=plan['checkpoint_sha256']):
        raise RuntimeError('Exact source/checkpoint identity required.')
    require_committed(sources)
    return dict(sources=sources,environment=expected,plan_sha256=PLAN_SHA,
                checkpoint_path=str(WEIGHT.relative_to(ROOT)),checkpoint_sha256=plan['checkpoint_sha256'],
                checkpoint_bytes=plan['checkpoint_bytes'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


def assert_frozen(frozen,check_weight=True):
    for name,digest in frozen['sources'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    if environment()!=frozen['environment']:raise RuntimeError('Frozen environment changed.')
    if check_weight and (WEIGHT.stat().st_size!=frozen['checkpoint_bytes'] or sha(WEIGHT)!=frozen['checkpoint_sha256']):
        raise RuntimeError('Frozen checkpoint changed.')


def child_environment():
    env=os.environ.copy()
    if 'KMP_DUPLICATE_LIB_OK' in env:raise RuntimeError('Duplicate OpenMP override forbidden.')
    spec=importlib.util.find_spec('torch')  # No native import/deserialization.
    library=Path(spec.origin).parent/'lib'
    if not library.resolve().is_relative_to((ROOT/'.venv-iltm').resolve()):raise RuntimeError('Torch runtime path outside isolated env.')
    env['DYLD_FALLBACK_LIBRARY_PATH']=str(library)
    for name in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[name]='4'
    return env


def array_hash(array):
    import numpy as np
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
def synthetic(rows,queries):
    import numpy as np
    import pandas as pd
    rng=np.random.default_rng(SEED);frame=pd.DataFrame(index=np.arange(rows+queries))
    for index,name in enumerate(COLUMNS):
        if index in CATEGORICAL:
            values=np.asarray([f'category_{v}' for v in rng.integers(0,3,rows+queries)],dtype=object)
            values[rng.random(len(values))<.03]=None
        else:
            values=rng.normal(size=rows+queries).astype('float64');values[rng.random(len(values))<.03]=np.nan
        frame[name]=values
    y=np.arange(rows,dtype=np.int64)%2
    return frame.iloc[:rows].copy(),y,frame.iloc[rows:].copy()
def frame_hash(frame):
    import pandas as pd
    return array_hash(pd.util.hash_pandas_object(frame,index=True).to_numpy())


class NativeObservation:
    """Read-only return profiling; never stores a frame, tensor or model reference."""
    def __init__(self,source_root,on_progress=None):
        self.on_progress=on_progress
        self.source_root=str(Path(source_root).resolve())+os.sep
        self.starts={};self.records=dict(loads=[],sampling=[],finetuning=[],best_weight_positions=[])
        self.error=None
    def __call__(self,frame,event,arg):
        if not frame.f_code.co_filename.startswith(self.source_root):return
        name=frame.f_code.co_name
        if name not in ('_initialize_model','_sample_data','fine_tune_main_network','validate_and_update'):return
        if event=='call':self.starts[id(frame)]=time.monotonic();return
        if event!='return':return
        try:
            loc=frame.f_locals;elapsed=time.monotonic()-self.starts.pop(id(frame),time.monotonic())
            if name=='_initialize_model' and arg is not None:
                mapping=loc.get('modified_state_dict',{})
                self.records['loads'].append(dict(seconds=elapsed,checkpoint_tensor_keys=len(mapping),
                    strict_native_load=True,model_parameters=sum(p.numel() for p in arg.parameters()),
                    floating_parameter_dtypes=sorted({str(p.dtype) for p in arg.parameters() if p.is_floating_point()})))
            elif name=='_sample_data' and arg is not None:
                self.records['sampling'].append(dict(input_rows=int(loc['len_X']),generation_rows=int(arg[1].shape[0])))
            elif name=='validate_and_update' and arg is not None:
                parent=frame.f_back.f_locals
                position=dict(where=str(loc['where']),epoch_index=int(parent.get('epoch',-1)),
                              updates=int(parent.get('global_batch_idx',0)))
                self.records['last_progress']=position
                if bool(arg):self.records['best_weight_positions'].append(position)
                if self.on_progress is not None:self.on_progress()
            elif name=='fine_tune_main_network':
                self.records['finetuning'].append(dict(normal_return=isinstance(arg,dict),seconds=elapsed,
                    epoch_cap=int(loc['max_epochs']),epochs_entered=int(loc.get('epoch',-1))+1,
                    optimizer_updates=int(loc.get('global_batch_idx',0)),early_stopped=bool(loc.get('stop_training',False)),
                    timed_out=bool(loc.get('timeout_triggered',True)),validation_checks=int(loc.get('es',{}).get('checks_done',0)),
                    selected_weights_restored=bool(loc.get('es',{}).get('checks_done',0)>0),
                    train_rows=int(len(loc['X_train'])) if 'X_train' in loc else None,
                    validation_rows=int(len(loc['y_val'])) if loc.get('y_val') is not None else 0,
                    training_retrieval_rows=int(len(loc['y_ctxt_superset'])) if loc.get('y_ctxt_superset') is not None else 0))
        except Exception as error:self.error=f'{type(error).__name__}: {error}'


def expected_parameters(cls):
    values={k:v.default for k,v in inspect.signature(cls).parameters.items() if v.default is not inspect.Parameter.empty}
    values.update(device='cpu',n_ensemble=1,seed=SEED,cat_features=CATEGORICAL,checkpoint=str(WEIGHT),**FAIL_CLOSED)
    # Exact official xgbrconcat checkpoint overrides and classifier-only normalization.
    values.update(tree_model='XGBoost_hist',preprocessing='realmlp_td_s_v0',tree_embedding=True,
                  concat_tree_with_orig_features=True,clip_predictions=False,normalize_predictions=False)
    return values

def check_model(model,expected,fitted=False):
    if model.get_params(deep=False)!=expected or str(model.device)!='cpu':raise RuntimeError('Native config changed or checkpoint override differs.')
    if model.inference_chunk_rows!=10000 or model.retrieval_context_max_rows!=8192:
        raise RuntimeError('Native query/retrieval limits changed.')
    if fitted:
        import numpy as np
        if (getattr(model,'_fit_succeeded',False) is not True or len(model.predictors_)!=1
                or any(p.get('timed_out') is not False for p in model.predictors_)
                or not np.array_equal(model.classes_,[0,1]) or len(model.preprocessors_)!=1):
            raise RuntimeError('One normally completed native member required; no partial ensemble.')


def probabilities(model,x):
    import numpy as np
    p=np.asarray(model.predict_proba(x))
    if (p.shape!=(len(x),2) or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or not np.allclose(p.sum(axis=1),1.,atol=2e-6,rtol=0) or not np.array_equal(model.classes_,[0,1])):
        raise RuntimeError('Invalid binary class-order probability matrix.')
    return p.copy()
def compare(a,b,tolerance):
    import numpy as np
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise RuntimeError('Incomplete/nonfinite comparison.')
    delta=float(np.max(np.abs(a-b)))
    if delta>tolerance or not np.array_equal(a[:,1]>=.5,b[:,1]>=.5):raise RuntimeError('Probability/threshold invariance failed.')
    return delta


def tensor_inventory(value):
    import torch
    import numpy as np
    seen=set();storages={};dtypes=set()
    def walk(obj):
        if id(obj) in seen:return
        seen.add(id(obj))
        if torch.is_tensor(obj):
            if obj.device.type!='cpu':raise RuntimeError('Portable predictor contains non-CPU tensor.')
            store=obj.untyped_storage();storages[(store.data_ptr(),store.nbytes())]=store.nbytes();dtypes.add(str(obj.dtype))
        elif isinstance(obj,torch.nn.Module):
            for part in list(obj.parameters())+list(obj.buffers()):walk(part)
        elif isinstance(obj,dict):
            for part in obj.values():walk(part)
        elif isinstance(obj,(tuple,list)):
            for part in obj:walk(part)
    walk(value)
    return dict(unique_tensor_storage_bytes=sum(storages.values()),tensor_storage_count=len(storages),dtypes=sorted(dtypes))


def run_case(out,rows,queries,frozen,deadline):
    """Called only in a child with runtime env already set by subprocess.Popen."""
    out=Path(out);record=dict(status='failed_inconclusive',training_rows=rows,query_rows=queries,
        probe_members=1,candidate_members=8,no_real_data=True,no_query_labels=True,no_accuracy_reported=True)
    model=restored=None;old_profile=None;observing=False;native_class=None
    started=time.monotonic();wall=time.time()
    def stage(name):
        record['current_stage']=name;record['monotonic_seconds']=time.monotonic()-started
        write(out/'partial.json',record);deadline_check(deadline)
    try:
        stage('before_native_import');assert_frozen(frozen)
        if os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')!=str(Path(importlib.util.find_spec('torch').origin).parent/'lib'):
            raise RuntimeError('Explicit isolated child loader environment required.')
        sys.path.insert(0,str(VENDOR))
        import numpy as np
        import joblib
        import torch
        from threadpoolctl import threadpool_info,threadpool_limits
        from iltm import iLTMClassifier
        import iltm.inference_interface as interface
        if Path(interface.__file__).resolve()!=(VENDOR/'iltm/inference_interface.py').resolve():raise RuntimeError('Wrong native source imported.')
        native_class=iLTMClassifier;torch.set_num_threads(4);torch.set_num_interop_threads(4)
        x,y,q=synthetic(rows,queries)
        record.update(schema=COLUMNS,categorical_indices=CATEGORICAL,training_input_sha256=frame_hash(x),
                      query_input_sha256=frame_hash(q),synthetic_labels_sha256=array_hash(y),fit_eval_set=None,fit_max_time=None)
        expected=expected_parameters(iLTMClassifier)
        with threadpool_limits(limits=4):
            stage('construct')
            model=iLTMClassifier(device='cpu',n_ensemble=1,seed=SEED,checkpoint=str(WEIGHT),cat_features=CATEGORICAL,**FAIL_CLOSED)
            check_model(model,expected);record['configuration']=expected
            observer=NativeObservation(VENDOR/'iltm',lambda:write(out/'partial.json',record))
            record['native_observation']=observer.records
            stage('native_fit');tick=time.monotonic()
            old_profile=sys.getprofile();sys.setprofile(observer);observing=True
            try:model.fit(x,y,eval_set=None,fit_max_time=None,return_partial_on_timeout=False)
            finally:sys.setprofile(old_profile);observing=False
            record['fit_seconds']=time.monotonic()-tick;record['native_observation']=observer.records
            if observer.error:raise RuntimeError('Native observation failed: '+observer.error)
            check_model(model,expected,True)
            fine=observer.records['finetuning'];sampling=observer.records['sampling'];loads=observer.records['loads']
            if (len(fine)!=1 or not fine[0]['normal_return'] or fine[0]['timed_out'] or fine[0]['epoch_cap']!=2048
                    or not 1<=fine[0]['epochs_entered']<=2048 or fine[0]['optimizer_updates']<1
                    or not fine[0]['selected_weights_restored'] or len(sampling)!=1
                    or sampling[0]!={'input_rows':rows,'generation_rows':min(rows,4096)}
                    or len(loads)!=1 or loads[0]['checkpoint_tensor_keys']<1
                    or loads[0]['floating_parameter_dtypes']!=['torch.float32']):
                raise RuntimeError('Complete native load/generation/epoch/selection observation required.')
            retrieval=len(model.predictors_[0]['y_ctxt_superset'])
            if retrieval!=min(rows,8192) or not 0<fine[0]['training_retrieval_rows']<=8192:
                raise RuntimeError('Native full-input retrieval bounds differ.')
            record['inference_retrieval_rows']=retrieval
            record['retained_predictor_tensors']=tensor_inventory(model.predictors_)
            stage('predict_full');tick=time.monotonic();first=probabilities(model,q)
            record['full_prediction_seconds']=time.monotonic()-tick
            stage('permutation_subset_partition')
            reverse=probabilities(model,q.iloc[::-1])[::-1]
            subset_ids=np.arange(0,queries,7);subset=probabilities(model,q.iloc[subset_ids])
            chunks=[probabilities(model,q.iloc[i:i+1024]) for i in range(0,queries,1024)]
            partition=np.concatenate(chunks)
            record['partition_query_rows']=[len(v) for v in chunks]
            record['invariance']=dict(permutation=compare(first,reverse,2e-5),subset=compare(first[subset_ids],subset,2e-5),
                                     partition=compare(first,partition,2e-5),threshold_equal=True)
            check_model(model,expected,True)
            stage('portable_save');tick=time.monotonic();joblib.dump(model,out/'model.joblib')
            record['save_seconds']=time.monotonic()-tick;record['portable_bytes']=(out/'model.joblib').stat().st_size
            reference=weakref.ref(model);model._release_training_model();native_class._model_cache.clear();model=None;gc.collect()
            if reference() is not None:raise RuntimeError('Original estimator retained across portable reload.')
            stage('portable_reload');tick=time.monotonic();restored=joblib.load(out/'model.joblib')
            if restored._model is not None or native_class._model_cache:raise RuntimeError('Portable reload retained pretrained engine.')
            check_model(restored,expected,True)
            second=probabilities(restored,q);record['reload_seconds']=time.monotonic()-tick
            record['reload_difference']=compare(first,second,2e-6);record['reload_threshold_equal']=True
            if tensor_inventory(restored.predictors_)!=record['retained_predictor_tensors']:raise RuntimeError('Portable predictor tensor schema changed.')
            check_model(restored,expected,True)
            np.savez(out/'probabilities.npz',first=first,reloaded=second,permuted=reverse,subset=subset,subset_ids=subset_ids,partition=partition)
            record.update(model_sha256=sha(out/'model.joblib'),probabilities_sha256=sha(out/'probabilities.npz'),
                original_released_before_reload=True,portable_cpu=True,full_prediction_rows=queries,reload_rows=queries,
                openmp_runtimes=[{k:v for k,v in item.items() if k in ['filepath','num_threads','prefix','internal_api']}
                                 for item in threadpool_info() if item.get('user_api')=='openmp'])
            restored=None;native_class._model_cache.clear();gc.collect()
            stage('final_identity');assert_frozen(frozen)
            record['status']='worker_complete'
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
    finally:
        if observing:sys.setprofile(old_profile)
        try:
            model=restored=None
            if native_class is not None:native_class._model_cache.clear()
            gc.collect();deadline_check(deadline);record['release_complete']=True
        except BaseException as error:
            record.update(status='failed_inconclusive',release_complete=False,cleanup_error=str(error))
            record.setdefault('error_type',type(error).__name__);record.setdefault('error',str(error))
        record.update(monotonic_seconds=time.monotonic()-started,wall_seconds=time.time()-wall,
                      self_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      self_maxrss_units='macOS bytes; OS process highwater, not tree peak',
                      profiling_disclosure='Read-only scalar/shape observation; overhead included; not zero perturbation.')
        write(out/'worker_result.json',record)
    return record


def process_tree(root_pid):
    raw=subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True)
    table={}
    for line in raw.splitlines():
        parts=line.split()
        if len(parts)==3 and all(p.isdigit() for p in parts):table[int(parts[0])]=(int(parts[1]),int(parts[2])*1024)
    members={int(root_pid)};changed=True
    while changed:
        old=len(members);members.update(pid for pid,(parent,_) in table.items() if parent in members);changed=len(members)>old
    return dict(pids=sorted(members),rss_bytes=sum(table.get(pid,(0,0))[1] for pid in members))
def terminate(process,pids):
    # Each worker owns a new session/process group. Never target another job.
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    process.wait(timeout=5)
    for pid in pids:
        if pid==process.pid:continue
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass

def supervise(process,deadline,out):
    peak=0;samples=0;last={'pids':[process.pid],'rss_bytes':0}
    try:
        while process.poll() is None:
            last=process_tree(process.pid);samples+=1;peak=max(peak,last['rss_bytes'])
            write(Path(out)/'external_monitor.json',dict(sampled_tree_rss_highwater_bytes=peak,samples=samples,
                limit_bytes=LIMIT,measurement='sampled sum of process-tree RSS; not exact peak; shared mappings may double-count'))
            if last['rss_bytes']>LIMIT:raise MemoryError('External process-tree RSS exceeded10GiB.')
            deadline_check(deadline);time.sleep(.25)
        if process.returncode!=0:raise RuntimeError('Native child exited abnormally: '+str(process.returncode))
        deadline_check(deadline)
        return peak
    except BaseException:
        try:terminate(process,last['pids'])
        except BaseException as cleanup:
            write(Path(out)/'termination_failure.json',dict(error_type=type(cleanup).__name__,error=str(cleanup)))
        raise


def qualify(record,out):
    import numpy as np
    rows,queries=record.get('training_rows'),record.get('query_rows')
    if ((rows,queries) not in CASES or record.get('status')!='worker_complete' or record.get('release_complete') is not True
            or record.get('probe_members')!=1 or record.get('candidate_members')!=8
            or record.get('full_prediction_rows')!=queries or record.get('reload_rows')!=queries
            or record.get('original_released_before_reload') is not True or record.get('portable_cpu') is not True
            or record.get('no_real_data') is not True or record.get('no_query_labels') is not True
            or record.get('no_accuracy_reported') is not True):raise RuntimeError('Complete single-member synthetic resource case required.')
    for name,key in [('model.joblib','model_sha256'),('probabilities.npz','probabilities_sha256')]:
        if sha(Path(out)/name)!=record[key]:raise RuntimeError('Resource artifact changed.')
    with np.load(Path(out)/'probabilities.npz',allow_pickle=False) as cache:
        first=cache['first']
        if (first.shape!=(queries,2) or not np.isfinite(first).all() or np.any((first<0)|(first>1))
                or not np.allclose(first.sum(axis=1),1.,atol=2e-6,rtol=0)
                or not np.array_equal(cache['subset_ids'],np.arange(0,queries,7))):
            raise RuntimeError('Full valid saved query coverage required.')
        compare(first,cache['reloaded'],2e-6);compare(first,cache['permuted'],2e-5)
        compare(first,cache['partition'],2e-5);compare(first[cache['subset_ids']],cache['subset'],2e-5)
    return True

def projection(records):
    estimates=[]
    for row in records:
        load=sum(v['seconds'] for v in row['native_observation']['loads'])
        fit=max(0.,row['fit_seconds']-load)
        seconds=1.25*(load+8*(fit+row['full_prediction_seconds']+row['save_seconds']+row['reload_seconds']))
        retained=max(row['retained_predictor_tensors']['unique_tensor_storage_bytes'],row['portable_bytes'])
        memory=math.ceil(1.25*(row['sampled_tree_rss_highwater_bytes']+7*retained))
        estimates.append(dict(training_rows=row['training_rows'],seconds=seconds,memory_bytes=memory,retained_per_member_bytes=retained))
    if len(estimates)!=2:raise RuntimeError('Both complete measured cases required for projection.')
    return dict(status='forecast_only_not_eight_member_qualification',margin=1.25,
        formula='time=1.25*(shared_native_load+8*(fit_minus_load+full_predict+save+reload)); memory=1.25*(observed_single_tree_RSS_highwater+7*max(retained_predictor_tensor_bytes,portable_file_bytes)); shared hypernetwork is counted once in measured working RSS',
        cases=estimates,three_fold_seconds=3*estimates[0]['seconds'],full_fit_reload_seconds=estimates[1]['seconds'],
        passed=3*estimates[0]['seconds']<=14400 and estimates[1]['seconds']<=7200 and max(e['memory_bytes'] for e in estimates)<=LIMIT)


def run(session_file):
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError('No overwrite/retry/resume.')
    quota(session_file);started=time.monotonic();deadline=started+BUDGET;wall=time.time()
    frozen=contract();deadline_check(deadline)
    OUTPUT.mkdir(parents=True);REPORT.mkdir(parents=True);process=None
    result=dict(status='failed_inconclusive',resource_qualification=False,automatic_cv=False,automatic_upload=False,
                probe_members=1,candidate_members=8,budget_seconds=BUDGET,rss_limit_bytes=LIMIT,cases=[])
    try:
        for folder in [OUTPUT,REPORT]:write(folder/'freeze.json',frozen)
        for name in frozen['sources']:
            path=OUTPUT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
        for rows,queries in CASES:
            quota(session_file);deadline_check(deadline);assert_frozen(frozen)
            out=OUTPUT/f'{rows}_{queries}';out.mkdir()
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(out),'--rows',str(rows),'--queries',str(queries),'--deadline',str(deadline)]
            with (out/'native.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=child_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                peak=supervise(process,deadline,out)
            child=read(out/'worker_result.json')
            if child.get('status')!='worker_complete':raise RuntimeError('Native child failed: '+str(child.get('error_type'))+': '+str(child.get('error')))
            qualify(child,out);child['sampled_tree_rss_highwater_bytes']=peak
            result['cases'].append(child);write(OUTPUT/'progress.json',result)
            for name in ['worker_result.json','external_monitor.json','partial.json']:
                shutil.copy2(out/name,REPORT/f'{rows}_{queries}_{name}')
        result['projection']=projection(result['cases'])
        if not result['projection']['passed']:raise RuntimeError('Fixed full8 cost/memory forecast exceeds allowed bounds.')
        assert_frozen(frozen);deadline_check(deadline)
        result.update(status='complete',resource_qualification=True,
            permits='Only a separately reviewed complete8-member resource proposal; no candidate score or CV authorization.')
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:terminate(process,process_tree(process.pid)['pids'])
            except BaseException as cleanup:result['termination_error']=str(cleanup)
        result.update(status='failed_inconclusive',resource_qualification=False,error_type=type(error).__name__,error=str(error))
        if 'out' in locals():
            for name in ['worker_result.json','partial.json','external_monitor.json']:
                if (out/name).exists():
                    shutil.copy2(out/name,REPORT/f'{out.name}_{name}')
                    if name=='worker_result.json':result['failed_child_record']=read(out/name)
                    elif name=='partial.json':result['failed_child_partial']=read(out/name)
        result['external_timeout']=isinstance(error,TimeoutError);result['external_rss_limit']=isinstance(error,MemoryError)
    result.update(monotonic_seconds=time.monotonic()-started,wall_seconds=time.time()-wall,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    for folder in [OUTPUT,REPORT]:write(folder/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-file',type=Path);parser.add_argument('--worker',type=Path)
    parser.add_argument('--rows',type=int);parser.add_argument('--queries',type=int);parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:
        if (args.rows,args.queries) not in CASES or args.deadline is None:parser.error('Fixed native case/deadline required.')
        frozen=read(OUTPUT/'freeze.json');require_committed(frozen['sources'])
        run_case(args.worker,args.rows,args.queries,frozen,args.deadline)
    elif args.session_file:print(json.dumps(run(args.session_file),indent=2))
    else:parser.error('--session-file is required for the sole bounded resource run.')
