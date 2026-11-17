"""Bounded FALSE-runtime bridge: one native training member, historical eight-member inference.

No real data, candidate score, CV activation, or eight-member retraining is performed.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import weakref

import iltm_r128bn_resource_probe as single
import iltm_r128bn_full8_probe as full

ROOT=Path(__file__).resolve().parent
PLAN='reports/iltm_false_runtime_bridge_design.json'
OUT=ROOT/'artifacts/probes/iltm_false_runtime_bridge'
REPORT=ROOT/'reports/iltm_false_runtime_bridge'
FLAG='KMP_DUPLICATE_LIB_OK'
BUDGET=1200
OWN=['iltm_false_runtime_bridge.py','test_iltm_false_runtime_bridge.py',PLAN]
read,write,sha=full.read,full.write,full.sha


def deadline_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Shared1200s FALSE bridge deadline exceeded.')


def entry_environment():
    # Called before contract/qualification can import any numeric dependency.
    if FLAG in os.environ:raise RuntimeError('Parent entry must have no duplicate-runtime override, including FALSE.')
    env=full.child_environment();env[FLAG]='FALSE';return env


child_environment=entry_environment


def require_false(stage='runtime'):
    if os.environ.get(FLAG)!='FALSE':raise RuntimeError('FALSE must persist: '+stage)


def validate_plan(plan):
    for key,value in dict(budget_seconds=1200,rss_limit_bytes=full.LIMIT,threads=4,seed=full.SEED,
            rows=8693,queries=4277,no_real_data=True,no_accuracy=True,automatic_cv=False,automatic_upload=False).items():
        if plan.get(key)!=value:raise RuntimeError('Fixed bridge plan differs: '+key)
    if (plan['first_stage']['members']!=1 or plan['first_stage']['baseline_probability_tolerance']!=2e-6
            or plan['second_stage']['members']!=8 or plan['second_stage']['fit'] is not False
            or plan['second_stage']['baseline_probability_tolerance']!=2e-5
            or plan['second_stage']['reload_probability_tolerance']!=2e-6
            or plan['environment_flag']['child_value']!='FALSE'
            or plan['qualification']['eight_FALSE_training_repeated'] is not False):
        raise RuntimeError('Bridge scope/tolerances changed.')


def verify_files(files):
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Bridge evidence changed: '+name)


def contract():
    full.require_committed(OWN)
    plan=read(ROOT/PLAN);validate_plan(plan)
    verify_files(plan['evidence_sha256']);verify_files(plan['installed_import_sources_sha256'])
    full.require_committed(plan['evidence_sha256'])
    smoke=read(ROOT/'reports/iltm_false_runtime/result.json')
    if (smoke.get('status')!='complete' or smoke.get('timed_out') is not False or smoke.get('exit_code')!=0
            or smoke.get('worker',{}).get('status')!='complete'
            or not smoke['worker'].get('stages')
            or any(s.get('duplicate_runtime_tolerance')!='FALSE' for s in smoke['worker']['stages'])):
        raise RuntimeError('Completed limited FALSE native-operations compatibility required.')
    current=full.contract();old=read(full.REPORT/'freeze.json')
    if any(current.get(k)!=old.get(k) for k in set(current)|set(old) if k!='source_commit'):
        raise RuntimeError('Original full8 source/environment freeze changed.')
    result=read(full.REPORT/'result.json')
    if (result.get('status')!='complete' or any(result.get(k) is not True for k in
            ['resource_execution_complete','cost_gate_passed','resource_qualification']) or len(result.get('cases',[]))!=1):
        raise RuntimeError('Original complete8 resources and measured cost gate required.')
    sources={**current['sources'],**{n:sha(ROOT/n) for n in OWN}}
    evidence={**plan['evidence_sha256'],**plan['installed_import_sources_sha256']};records={};freezes={}
    for label,module in [('single',single),('full8',full)]:
        for name in ['freeze.json','result.json']:
            if sha(module.REPORT/name)!=sha(module.OUTPUT/name):raise RuntimeError('Old resource artifact/report mismatch.')
        folder=module.OUTPUT/'8693_4277';record=read(folder/'worker_result.json')
        if sha(folder/'worker_result.json')!=sha(module.REPORT/'8693_4277_worker_result.json'):
            raise RuntimeError('Original resource worker evidence mismatch.')
        module.qualify(record,folder)
        records[label]=record;freezes[label]=read(module.REPORT/'freeze.json');module.assert_frozen(freezes[label])
        for name in ['model.joblib','probabilities.npz','worker_result.json']:
            evidence[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    row=result['cases'][0]
    if any(row.get(k)!=v for k,v in records['full8'].items()):raise RuntimeError('Original full8 summary/worker differs.')
    actual=full.projection([row])
    if not actual['passed'] or result.get('projection')!=actual:raise RuntimeError('Original full8 measured cost gate differs.')
    full.require_committed(sources)
    return dict(sources=sources,evidence=evidence,resource_freezes=freezes,original_records=records,
                plan_sha256=sha(ROOT/PLAN),runtime_flag='FALSE',eight_FALSE_training_repeated=False)


def assert_frozen(frozen):
    verify_files(frozen['sources']);verify_files(frozen['evidence'])
    single.assert_frozen(frozen['resource_freezes']['single']);full.assert_frozen(frozen['resource_freezes']['full8'])
    if frozen.get('runtime_flag')!='FALSE' or frozen.get('eight_FALSE_training_repeated') is not False:
        raise RuntimeError('Wrong bridge freeze.')


class FlagAudit:
    """Scalar-only checks; no frame, tensor, or native return object is retained."""
    def __init__(self,out,deadline):
        self.out=Path(out);self.deadline=deadline
        self.record=dict(stages=[],training_progress_checks=0,original_single_stage_checks=0)
    def check(self,name,progress=False):
        require_false(name)
        deadline_check(self.deadline)
        if progress:self.record['training_progress_checks']+=1
        else:self.record['stages'].append(dict(name=name,value='FALSE'))
        write(self.out/'flag_audit.json',self.record)
    def __call__(self,frame,event,arg):
        file=frame.f_code.co_filename;name=frame.f_code.co_name
        if event=='call' and file==str(Path(single.__file__).resolve()) and name=='stage':
            self.record['original_single_stage_checks']+=1;self.check('single_'+str(frame.f_locals['name']))
        if name=='validate_and_update' and file.startswith(str(full.VENDOR/'iltm')+os.sep):
            if event=='call':
                frame.f_trace_lines=False;frame.f_trace_opcodes=False
                return self
            if event=='return':self.check('native_training_progress',progress=True)
        return None


def run_single(out,frozen,deadline,audit):
    previous=sys.gettrace();audit.check('before_original_single_run_case')
    try:
        sys.settrace(audit)
        result=single.run_case(out,8693,4277,frozen['resource_freezes']['single'],deadline)
    finally:sys.settrace(previous)
    audit.check('after_original_single_run_case')
    if result.get('status')!='worker_complete':
        raise RuntimeError('Original single native run failed: '+str(result.get('error_type'))+': '+str(result.get('error')))
    single.qualify(result,out)
    original=frozen['original_records']['single']
    for key in ['configuration','training_input_sha256','query_input_sha256','synthetic_labels_sha256']:
        if result.get(key)!=original.get(key):raise RuntimeError('Single replay input/config identity differs: '+key)
    import numpy as np
    with np.load(single.OUTPUT/'8693_4277/probabilities.npz',allow_pickle=False) as old, \
         np.load(out/'probabilities.npz',allow_pickle=False) as new:
        difference=single.compare(old['first'],new['first'],2e-6)
        reload_difference=single.compare(old['first'],new['reloaded'],2e-6)
    if audit.record['training_progress_checks']<1 or audit.record['original_single_stage_checks']<1:
        raise RuntimeError('Actual native training/stage flag observations required.')
    write(out/'bridge_comparison.json',dict(baseline_max_difference=difference,
        baseline_to_reload_max_difference=reload_difference,threshold_equal=True))
    return result


def check_loaded_eight(model,expected,original,cls):
    full.check_model(model,expected,True)
    if model._model is not None or cls._model_cache:raise RuntimeError('Inference bridge must not restore generator engine/cache.')
    if [len(p['y_ctxt_superset']) for p in model.predictors_]!=[8192]*8:raise RuntimeError('Original eight retrieval contexts changed.')
    inventory=full.tensor_inventory(model.predictors_)  # Rejects any non-CPU tensor.
    if (inventory['dtypes']!=original['retained_predictor_tensors']['dtypes']
            or inventory['unique_tensor_storage_bytes']<=0):
        raise RuntimeError('CPU stored predictor tensor dtypes/content missing.')
    return inventory  # Storage alias count/bytes are observed, not an equality gate.


def run_eight(out,frozen,deadline,audit):
    import numpy as np
    import joblib
    from iltm import iLTMClassifier
    from threadpoolctl import threadpool_limits
    model=restored=None;original=frozen['original_records']['full8'];expected=full.expected_parameters(iLTMClassifier)
    record=dict(status='failed_inconclusive',members=8,training_repeated=False,queries=4277)
    try:
        out.mkdir();audit.check('before_historical_eight_hash_load')
        path=full.OUTPUT/'8693_4277/model.joblib'
        if sha(path)!=original['model_sha256']:raise RuntimeError('Historical model SHA changed before load.')
        model=joblib.load(path);record['loaded_tensor_inventory']=check_loaded_eight(model,expected,original,iLTMClassifier)
        _,_,q=single.synthetic(8693,4277)
        if single.frame_hash(q)!=original['query_input_sha256']:raise RuntimeError('Historical query input differs.')
        with np.load(full.OUTPUT/'8693_4277/probabilities.npz',allow_pickle=False) as saved:baseline=saved['first'].copy()
        with threadpool_limits(limits=4):
            audit.check('before_eight_full_prediction');first=full.probabilities(model,q)
            audit.check('after_eight_full_prediction')
            reverse=full.probabilities(model,q.iloc[::-1])[::-1];audit.check('after_eight_reverse')
            ids=np.arange(0,len(q),7);subset=full.probabilities(model,q.iloc[ids]);audit.check('after_eight_subset')
            chunks=[]
            for start in range(0,len(q),1024):
                chunks.append(full.probabilities(model,q.iloc[start:start+1024]));audit.check('after_eight_chunk_'+str(start))
            partition=np.concatenate(chunks)
            record['baseline_differences']=dict(full=full.compare(baseline,first,2e-5),
                reverse=full.compare(baseline,reverse,2e-5),subset=full.compare(baseline[ids],subset,2e-5),
                partition=full.compare(baseline,partition,2e-5))
            check_loaded_eight(model,expected,original,iLTMClassifier)
            audit.check('before_eight_save');joblib.dump(model,out/'model.joblib')
            reference=weakref.ref(model);model._release_training_model();model=None;iLTMClassifier._model_cache.clear();gc.collect()
            if reference() is not None:raise RuntimeError('Historical estimator retained across reload.')
            audit.check('after_eight_release_before_reload')
            restored=joblib.load(out/'model.joblib')
            record['reloaded_tensor_inventory']=check_loaded_eight(restored,expected,original,iLTMClassifier)
            second=full.probabilities(restored,q);record['reload_max_difference']=full.compare(first,second,2e-6)
            full.compare(baseline,second,2e-5);check_loaded_eight(restored,expected,original,iLTMClassifier)
            audit.check('after_eight_complete_reload')
            np.savez(out/'probabilities.npz',first=first,reloaded=second,permuted=reverse,subset=subset,subset_ids=ids,partition=partition)
            record.update(status='complete',configuration=expected,portable_cpu=True,full_prediction_rows=4277,reload_rows=4277,
                original_released_before_reload=True,threshold_equal=True,partition_query_rows=[len(v) for v in chunks],
                model_sha256=sha(out/'model.joblib'),probabilities_sha256=sha(out/'probabilities.npz'))
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        active_error=sys.exc_info()[1]
        try:
            model=restored=None;iLTMClassifier._model_cache.clear();gc.collect()
            audit.check('after_eight_final_release')
        except BaseException as cleanup:
            record.update(status='failed_inconclusive',cleanup_error=str(cleanup))
            record.setdefault('error_type',type(cleanup).__name__);record.setdefault('error',str(cleanup))
            if active_error is None:raise
        finally:
            if out.exists():write(out/'result.json',record)
    return record


def worker(deadline):
    started=time.monotonic();record=dict(status='failed_inconclusive',eight_FALSE_training_repeated=False,
        no_real_data=True,no_accuracy=True,automatic_cv=False,automatic_upload=False)
    audit=FlagAudit(OUT,deadline)
    try:
        audit.check('worker_before_any_numeric_import')
        if any(name in sys.modules for name in ['numpy','torch','sklearn','threadpoolctl']):
            raise RuntimeError('Fresh worker before numeric imports required.')
        frozen=read(OUT/'freeze.json');assert_frozen(frozen);full.require_committed(frozen['sources'])
        single_out=OUT/'single';record['current_stage']='single_FALSE_training';write(OUT/'partial.json',record)
        single_result=run_single(single_out,frozen,deadline,audit)
        record['single_worker_sha256']=sha(single_out/'worker_result.json')
        record['current_stage']='historical_eight_FALSE_inference';write(OUT/'partial.json',record)
        eight=run_eight(OUT/'eight',frozen,deadline,audit)
        record['eight_result_sha256']=sha(OUT/'eight/result.json')
        assert_frozen(frozen);audit.check('worker_finished')
        record.update(status='complete',single_FALSE_training_completed=True,historical_eight_FALSE_inference_and_reload_completed=True,
            original_single_baseline_difference=read(single_out/'bridge_comparison.json'),eight_inference=eight)
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
        path=OUT/'single/worker_result.json'
        if path.exists() and read(path).get('status')!='worker_complete':record['single_native_failure']=read(path)
    finally:
        record.update(monotonic_seconds=time.monotonic()-started,flag_audit=audit.record)
        write(OUT/'worker_result.json',record)
    return record


def qualify(record,out,frozen):
    import numpy as np
    if (record.get('status')!='complete' or record.get('single_FALSE_training_completed') is not True
            or record.get('historical_eight_FALSE_inference_and_reload_completed') is not True
            or record.get('eight_FALSE_training_repeated') is not False or record.get('no_real_data') is not True
            or record.get('no_accuracy') is not True):raise RuntimeError('Complete limited bridge evidence required.')
    flags=record['flag_audit']
    if (flags.get('training_progress_checks',0)<1 or flags.get('original_single_stage_checks',0)<1
            or not flags.get('stages') or any(row.get('value')!='FALSE' for row in flags['stages'])):
        raise RuntimeError('FALSE import/stage/native-training audit required.')
    if record['single_worker_sha256']!=sha(out/'single/worker_result.json') or record['eight_result_sha256']!=sha(out/'eight/result.json'):
        raise RuntimeError('Bridge records changed.')
    single.qualify(read(out/'single/worker_result.json'),out/'single')
    with np.load(single.OUTPUT/'8693_4277/probabilities.npz',allow_pickle=False) as old, \
         np.load(out/'single/probabilities.npz',allow_pickle=False) as new:
        full.compare(old['first'],new['first'],2e-6);full.compare(old['first'],new['reloaded'],2e-6)
    eight=read(out/'eight/result.json')
    if (eight.get('status')!='complete' or eight.get('members')!=8 or eight.get('training_repeated') is not False
            or eight.get('full_prediction_rows')!=4277 or eight.get('reload_rows')!=4277
            or eight.get('original_released_before_reload') is not True or eight.get('portable_cpu') is not True
            or eight.get('partition_query_rows')!=[1024,1024,1024,1024,181]):raise RuntimeError('Complete historical8 inference/reload required.')
    if eight['configuration']!=frozen['original_records']['full8']['configuration']:raise RuntimeError('Historical8 configuration differs.')
    for name,key in [('model.joblib','model_sha256'),('probabilities.npz','probabilities_sha256')]:
        if sha(out/'eight'/name)!=eight[key]:raise RuntimeError('Bridge eight persistence changed.')
    with np.load(full.OUTPUT/'8693_4277/probabilities.npz',allow_pickle=False) as old, \
         np.load(out/'eight/probabilities.npz',allow_pickle=False) as new:
        ids=np.arange(0,4277,7)
        if not np.array_equal(ids,new['subset_ids']):raise RuntimeError('Subset IDs changed.')
        for key in ['first','reloaded','permuted','partition']:
            values=new[key]
            if (values.shape!=(4277,2) or not np.isfinite(values).all() or np.any((values<0)|(values>1))
                    or not np.allclose(values.sum(axis=1),1.,atol=2e-6,rtol=0)):
                raise RuntimeError('Complete finite binary matrix required.')
            full.compare(old['first'],values,2e-5)
        full.compare(old['first'][ids],new['subset'],2e-5);full.compare(new['first'],new['reloaded'],2e-6)
    return True


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('One bridge attempt; no overwrite/resume/retry.')
    started=time.monotonic();deadline=started+BUDGET
    env=child_environment()  # Before all numeric/qualification imports, including numpy.
    frozen=contract();deadline_check(deadline)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True);process=None
    result=dict(status='failed_inconclusive',bridge_qualified=False,eight_FALSE_training_repeated=False,
        automatic_cv=False,automatic_upload=False,budget_seconds=BUDGET,rss_limit_bytes=full.LIMIT)
    try:
        for folder in [OUT,REPORT]:write(folder/'freeze.json',frozen)
        with (OUT/'native.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--deadline',str(deadline)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            peak=full.supervise(process,deadline,OUT)
        record=read(OUT/'worker_result.json');result['worker']=record
        qualify(record,OUT,frozen);assert_frozen(frozen);deadline_check(deadline)
        result.update(status='complete',bridge_qualified=True,sampled_tree_rss_highwater_bytes=peak,
            permits='Separate review of a new environment-migration CV wrapper only; no eight-member FALSE retraining proof.')
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:full.terminate(process,full.process_tree(process.pid)['pids'])
            except BaseException as cleanup:result['termination_error']=str(cleanup)
        result.update(error_type=type(error).__name__,error=str(error))
        for name in ['worker_result.json','partial.json','flag_audit.json','external_monitor.json']:
            if (OUT/name).exists():result[name.removesuffix('.json')]=read(OUT/name)
        result['external_timeout']=isinstance(error,TimeoutError);result['external_rss_limit']=isinstance(error,MemoryError)
    for path in OUT.rglob('*.json'):
        if path.name=='result.json' and path.parent==OUT:continue
        target=REPORT/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    result['monotonic_seconds']=time.monotonic()-started
    if time.monotonic()>=deadline:
        result.update(status='failed_inconclusive',bridge_qualified=False,external_timeout=True)
        result.setdefault('error_type','TimeoutError');result.setdefault('error','Shared1200s bridge budget exceeded.')
        result.pop('permits',None)
    for folder in [OUT,REPORT]:write(folder/'result.json',result)
    return result


def resource_gate():
    """Read-only committed bridge gate for a separately reviewed future wrapper."""
    current=contract()
    names=['result.json','freeze.json','worker_result.json','flag_audit.json','external_monitor.json',
           'single/worker_result.json','single/bridge_comparison.json','eight/result.json']
    full.require_committed([str((REPORT/name).relative_to(ROOT)) for name in names])
    evidence={}
    for name in names:
        if sha(REPORT/name)!=sha(OUT/name):raise RuntimeError('Bridge public/artifact evidence differs: '+name)
        for directory in [OUT,REPORT]:evidence[str((directory/name).relative_to(ROOT))]=sha(directory/name)
    saved=read(REPORT/'freeze.json')
    if current!=saved:raise RuntimeError('Bridge current sources/environment/evidence differ from qualification.')
    result=read(REPORT/'result.json');record=read(REPORT/'worker_result.json')
    if (result.get('status')!='complete' or result.get('bridge_qualified') is not True
            or result.get('eight_FALSE_training_repeated') is not False or result.get('budget_seconds')!=1200
            or result.get('rss_limit_bytes')!=full.LIMIT or result.get('worker')!=record):
        raise RuntimeError('Complete committed limited bridge required; no implicit eight-member retraining.')
    monitor=read(REPORT/'external_monitor.json')
    if (not 0<monitor.get('sampled_tree_rss_highwater_bytes',0)<=full.LIMIT or monitor.get('samples',0)<1
            or result.get('sampled_tree_rss_highwater_bytes')!=monitor['sampled_tree_rss_highwater_bytes']
            or result.get('monotonic_seconds',float('inf'))>1200):raise RuntimeError('Bridge external bounds failed.')
    if record.get('flag_audit')!=read(REPORT/'flag_audit.json'):raise RuntimeError('Final flag audit differs.')
    qualify(record,OUT,current);assert_frozen(current)
    for sub in ['single','eight']:
        for name in ['model.joblib','probabilities.npz']:
            path=OUT/sub/name;evidence[str(path.relative_to(ROOT))]=sha(path)
    current['evidence'].update(evidence)
    return current


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:
        if args.deadline is None:parser.error('Parent deadline required.')
        worker(args.deadline)
    else:print(__import__('json').dumps(run(),indent=2))
