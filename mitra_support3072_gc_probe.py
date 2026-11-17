"""One state-preserving GC diagnostic; synthetic targets, no accuracy or repair claim."""
import argparse
import copy
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import mitra_candidate as base
import mitra_full_episode as original
import mitra_long_context_probe as update_probe
import mitra_support3072 as training
import mitra_lifecycle_guard as guard
import exaone_validation_data as attributes

ROOT=Path(__file__).resolve().parent
PLAN='reports/support3072_gc_devicefix_plan.json'
PLAN_SHA='7cf319a7e2848f9e40282305d67449200a73bc12ca33b631a707985c1276d93d'
OUTPUT=ROOT/'artifacts/probes/mitra_support3072_gc_devicefix'
REPORT=ROOT/'reports/mitra_support3072_gc_devicefix_resources'
BUDGET,SEED=1200,20261112
ATTRIBUTES=['PassengerId','HomePlanet','CryoSleep','Destination','Age','VIP','RoomService',
            'FoodCourt','ShoppingMall','Spa','VRDeck','Cabin']
OWN=['mitra_lifecycle_guard.py','test_mitra_lifecycle_guard.py','mitra_support3072_gc_probe.py',
     'test_mitra_support3072_gc_probe.py',PLAN,'quota_guard.py','exaone_validation_data.py',
     'audit_top5_references.py','three_family_mean_screen.py','prior_adapt_mean_screen.py','reports/official_data_verification.json']
read,write=update_probe.read,guard.write


def contract():
    if base.sha256(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Fixed lifecycle diagnostic plan changed.')
    plan=read(ROOT/PLAN);historical=read(ROOT/'reports/mitra_long_context_resources/freeze.json')
    evidence={**plan['evidence_sha256'],plan['review_path']:plan['review_sha256']}
    official=read(ROOT/'reports/official_data_verification.json')
    evidence['data/raw/train.csv']=official['files']['train.csv']['official_sha256']
    sources={**historical['sources'],**{name:base.sha256(ROOT/name) for name in OWN}}
    lock=base.environment_lock()
    if lock!=(ROOT/'reports/mitra_long_context_resources/requirements.lock').read_text():
        raise RuntimeError('Exact original Mitra environment required.')
    frozen=dict(sources=sources,evidence_files=evidence,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        checkpoint_sha256=base.WEIGHT_SHA,backend=base.SETTINGS,plan_sha256=PLAN_SHA,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    check_frozen(frozen)
    update_probe.require_committed([*sources,*[p for p in evidence if not p.startswith(('artifacts/','data/'))]])
    return frozen,lock


def check_frozen(frozen):
    for name,digest in {**frozen['sources'],**frozen['evidence_files']}.items():
        if base.sha256(ROOT/name)!=digest:raise RuntimeError('Frozen diagnostic input changed: '+name)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=frozen['environment_sha256']:
        raise RuntimeError('Frozen diagnostic environment changed.')
    base.verify_checkpoint()
    if base.WEIGHT_SHA!=frozen['checkpoint_sha256'] or base.SETTINGS!=frozen['backend']:
        raise RuntimeError('Pinned model/backend changed.')


def load_synthetic_target_split():
    raw=pd.read_csv(ROOT/'data/raw/train.csv',usecols=ATTRIBUTES,dtype={'PassengerId':str})
    if len(raw)!=8693 or 'Transported' in raw:raise RuntimeError('Exactly8693 attribute-only rows required.')
    x=attributes.compact_features(raw);y=np.arange(8693,dtype=np.int64)%2
    inner,valid=train_test_split(np.arange(8693),test_size=.2,stratify=y,random_state=SEED)
    if len(inner)!=6954 or len(valid)!=1739 or np.intersect1d(inner,valid).size:
        raise RuntimeError('Fixed synthetic-target split differs.')
    return x,y,inner,valid


def validation(model,optimizer,data,targets,query,label):
    records=[]
    def before(module,args):records.append(dict(support_rows=int(args[0].shape[1]),label_rows=int(args[1].shape[1]),query_rows=int(args[2].shape[1])))
    handle=model.register_forward_pre_hook(before)
    try:probability=training.validation_prediction(model,optimizer,data,targets,query)
    finally:handle.remove()
    expected=[dict(support_rows=len(data),label_rows=len(data),query_rows=len(query))]
    if records!=expected:raise RuntimeError('Actual full-context validation coverage differs.')
    return probability,dict(stage=label,status='complete',support_rows=len(data),query_rows=len(query),forward_records=records,
        optimizer_steps=update_probe.optimizer_steps(optimizer),finite_normalized_probabilities=True,
        resources=update_probe.synchronized_snapshot(data.device.type))


def compare_small(before,after):
    if before.shape!=(928,2) or after.shape!=before.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise RuntimeError('Complete finite928-row small comparison required.')
    difference=float(np.max(np.abs(before-after)))
    if difference>2e-6 or not np.array_equal(before[:,1]>=.5,after[:,1]>=.5):
        raise RuntimeError('Barrier changed small-case probability/threshold decisions.')
    return difference


def run_stage(kind,out,report,frozen,deadline):
    """Fresh child-local model/optimizer/preprocessing/RNG; no cross-stage reuse."""
    out,report=Path(out),Path(report);base.DEADLINE=deadline
    result=dict(status='incomplete',stage=kind,real_targets_read=False,synthetic_alternating_targets=True,
        exact_replay_of_real_label_split=False,no_accuracy_computed=True,updates=[],validations=[],barriers=[],release_complete=False)
    model=optimizer=data=targets=query=prep=rng=reference_rng=None
    def persist():
        write(out/'partial.json',result);write(report/'partial.json',result)
    def checkpoint(name,snapshot):
        update_probe.check_deadline(deadline)
        result['current_stage']=kind+':'+name;result['current_resources']=snapshot;persist()
    try:
        check_frozen(frozen);update_probe.check_deadline(deadline)
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0') or not base.torch.backends.mps.is_available():
            raise RuntimeError('Original MPS backend without fallback required.')
        base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)
        x,y,inner,valid=load_synthetic_target_split()
        if kind=='small':inner,valid=inner[:3708],valid[:928]
        elif kind!='full':raise ValueError('Only fixed small/full diagnostic stages permitted.')
        result.update(support_rows=len(inner),query_rows=len(valid),support_ids_sha256=base.ids_hash(inner),
            query_ids_sha256=base.ids_hash(valid),synthetic_labels_sha256=base.ids_hash(y[inner]),
            seed=SEED,allocator_limit_bytes=int(base.torch.mps.recommended_max_memory()*.65))
        base.seed_everything(SEED,'mps')
        prep=base.MitraPreprocessor().fit(x.iloc[inner],y[inner],SEED)
        if prep.n_features!=13:raise RuntimeError('Exactly13 original active features required.')
        model,_=original.fresh_model('mps')
        data=base.torch.as_tensor(prep.transform(x.iloc[inner]),device='mps')
        targets=base.torch.as_tensor(y[inner],device='mps')
        query=base.torch.as_tensor(prep.transform(x.iloc[valid]),device='mps')
        optimizer=base.optimizer_for(model);rng=np.random.default_rng(SEED);reference_rng=np.random.default_rng(SEED)
        probability,row=validation(model,optimizer,data,targets,query,'V0');result['validations'].append(row);persist()
        optimizer_id=id(optimizer)
        for step in range(1,2 if kind=='small' else 3):
            checkpoint(f'U{step}:start',update_probe.synchronized_snapshot('mps'))
            updated=update_probe.update(model,optimizer,data,targets,rng,reference_rng,step,checkpoint)
            if id(optimizer)!=optimizer_id or update_probe.optimizer_steps(optimizer)!=[step]:
                raise RuntimeError('Optimizer identity/counters changed.')
            result['updates'].append(updated);persist()
            if kind=='small':
                first,row=validation(model,optimizer,data,targets,query,'V1_before_barrier');result['validations'].append(row);persist()
            path=out/f'barrier_{step}.json'
            checkpoint(f'barrier{step}:start',update_probe.synchronized_snapshot('mps'))
            proof=guard.lifecycle_barrier(model,optimizer,data,targets,query,rng,reference_rng,'mps',path)
            shutil.copy2(path,report/path.name)
            result['barriers'].append(dict(step=step,file=path.name,sha256=base.sha256(path),state_preserved=proof['state_preserved'],gc_collected=proof['gc_collected']))
            persist()
            probability,row=validation(model,optimizer,data,targets,query,'V1_after_barrier' if kind=='small' else f'V{step}')
            result['validations'].append(row);persist()
            if kind=='small':
                if guard.capture_state(model,optimizer,data,targets,query,rng,reference_rng,'mps')!=proof['after']:
                    raise RuntimeError('Small comparison validation changed updated state/RNG.')
                result['probability_max_difference']=compare_small(first,probability);result['threshold_equal']=True
                np.savez(out/'probability_comparison.npz',before=first,after=probability,query_ids=valid)
                shutil.copy2(out/'probability_comparison.npz',report/'probability_comparison.npz')
                result['probability_comparison_sha256']=base.sha256(out/'probability_comparison.npz')
            check_frozen(frozen);update_probe.check_deadline(deadline)
        result['optimizer_identity_persistent']=id(optimizer)==optimizer_id
        result['current_stage']=kind+':sequence_complete'
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error));persist()
    finally:
        model=optimizer=data=targets=query=prep=rng=reference_rng=None
        try:
            base.release_memory();result['resources_after_release']=update_probe.synchronized_snapshot('mps')
            update_probe.check_deadline(deadline)
            if result['resources_after_release']['mps_current_bytes']!=0:raise RuntimeError('Live MPS tensors remain after stage release.')
            result['release_complete']=True
            if result['status']=='incomplete':result['status']='worker_complete'
        except BaseException as error:
            result.update(status='failed_inconclusive',release_complete=False,cleanup_error_type=type(error).__name__,cleanup_error=str(error))
            if 'error_type' not in result:result.update(error_type=type(error).__name__,error=str(error))
        persist();write(out/'worker_result.json',result);write(report/'worker_result.json',result)


def qualify_stage(record,out,kind):
    support,query=(3708,928) if kind=='small' else (6954,1739)
    expected_stages=['V0','V1_before_barrier','V1_after_barrier'] if kind=='small' else ['V0','V1','V2']
    steps=[1] if kind=='small' else [1,2]
    if (record.get('status')!='worker_complete' or record.get('stage')!=kind or record.get('release_complete') is not True
            or record.get('real_targets_read') is not False or record.get('no_accuracy_computed') is not True
            or record.get('exact_replay_of_real_label_split') is not False or record.get('support_rows')!=support or record.get('query_rows')!=query
            or record.get('optimizer_identity_persistent') is not True
            or [v['stage'] for v in record['validations']]!=expected_stages or [v['step'] for v in record['updates']]!=steps
            or len(record['barriers'])!=len(steps) or record['resources_after_release']['mps_current_bytes']!=0):
        raise RuntimeError('Complete original-sequence stage evidence required.')
    for row,step in zip(record['updates'],steps):
        if (row['optimizer_state_steps']!=[step] or row['support_rows']!=3072 or row['query_rows']!=256
                or row['parameters_with_finite_grad']!=75670026 or row['optimizer_state_tensor_bytes']!=update_probe.OPTIMIZER_BYTES
                or not all(row.get(k) is True for k in ['disjoint_ids','query_equal_v81','support_prefix_equal_v81','rng_state_equal_v81'])
                or any(row['updated_blocks'].get(f'layers.{k}') is not True for k in range(12))):
            raise RuntimeError('Original update/paired-sampler/Adam proof differs.')
    counters=[[],[1],[1]] if kind=='small' else [[],[1],[2]]
    for row,steps_at_validation in zip(record['validations'],counters):
        if (row['status']!='complete' or row['forward_records']!=[dict(support_rows=support,label_rows=support,query_rows=query)]
                or row['optimizer_steps']!=steps_at_validation or row['finite_normalized_probabilities'] is not True):
            raise RuntimeError('Complete validation coverage and resident moments required.')
    for index,row in enumerate(record['barriers'],1):
        path=out/f'barrier_{index}.json';proof=read(path)
        state=proof.get('before',{})
        required={'model_object_id','optimizer_object_id','model_training','module_training','parameters','buffers',
                  'all_gradients_none','all_input_gradients_none','optimizer_steps','optimizer_groups','optimizer_state','inputs','rng'}
        if (not required.issubset(state) or not state['parameters'] or not state['optimizer_state']
                or state['all_gradients_none'] is not True or state['all_input_gradients_none'] is not True
                or set(state['inputs'])!={'data','targets','query'} or not state['optimizer_steps']
                or set(state['optimizer_steps'].values())!={float(index)}
                or not {'python_complete_state_sha256','numpy_global_complete_state_sha256','sampler',
                        'counterfactual_sampler','torch_cpu','torch_mps'}.issubset(state['rng'])):
            raise RuntimeError('Full model/moments/inputs/RNG state proof is missing.')
        if (row['step']!=index or row['file']!=path.name or row['sha256']!=base.sha256(path)
                or proof.get('status')!='complete' or proof.get('state_preserved') is not True
                or proof['before']!=proof['after'] or proof.get('barrier')!=guard.BARRIER or proof.get('device')!='mps'
                or type(proof.get('gc_collected')) is not int or proof['gc_collected']!=row['gc_collected']
                or [v['stage'] for v in proof['observations']]!=['before_pre_hash','after_pre_hash','barrier_after_synchronize1',
                    'barrier_after_gc_collect','barrier_after_empty_cache','barrier_after_synchronize2','before_post_hash','after_post_hash']):
            raise RuntimeError('Complete state-preservation and substep observations required.')
    if kind=='small':
        path=out/'probability_comparison.npz'
        if base.sha256(path)!=record['probability_comparison_sha256']:raise RuntimeError('Small saved probability proof changed.')
        with np.load(path,allow_pickle=False) as cache:
            difference=compare_small(cache['before'],cache['after'])
            if base.ids_hash(cache['query_ids'])!=record['query_ids_sha256']:raise RuntimeError('Small comparison IDs changed.')
        if record['probability_max_difference']!=difference or record.get('threshold_equal') is not True:
            raise RuntimeError('Small saved probability comparison differs.')
    return True


def supervise(process,deadline):
    while process.is_alive() and time.monotonic()<deadline:process.join(timeout=min(5,max(0,deadline-time.monotonic())))
    if process.is_alive():
        process.kill();process.join(timeout=5)
        if process.is_alive():raise RuntimeError('Diagnostic worker could not be reaped.')
        raise TimeoutError('External supervisor killed diagnostic at fixed1200s deadline.')
    if process.exitcode!=0:raise RuntimeError('Diagnostic worker exited abnormally: '+str(process.exitcode))


def run(session_file):
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError('No overwrite/resume of diagnostic evidence.')
    update_probe.quota_guard(session_file);started=time.monotonic();deadline=started+BUDGET
    OUTPUT.mkdir(parents=True);REPORT.mkdir(parents=True);process=None
    result=dict(status='failed_inconclusive',diagnostic_qualification=False,budget_seconds=BUDGET,
        externally_supervised=True,no_real_targets=True,no_accuracy_computed=True,no_automatic_fullfit=True,stages=[])
    try:
        frozen,lock=contract()
        for directory in [OUTPUT,REPORT]:write(directory/'freeze.json',frozen);(directory/'requirements.lock').write_text(lock)
        for name in frozen['sources']:
            path=OUTPUT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
        for kind in ['small','full']:
            out,report=OUTPUT/kind,REPORT/kind;out.mkdir();report.mkdir()
            check_frozen(frozen);update_probe.check_deadline(deadline)
            process=multiprocessing.get_context('spawn').Process(target=run_stage,args=(kind,str(out),str(report),frozen,deadline))
            process.start();supervise(process,deadline);update_probe.check_deadline(deadline)
            child=read(out/'worker_result.json')
            if child.get('status')!='worker_complete':
                result.update(worker_error_type=child.get('error_type'),worker_error=child.get('error'),worker_current_stage=child.get('current_stage'))
                raise RuntimeError('Diagnostic stage failed: '+str(child.get('error_type'))+': '+str(child.get('error')))
            qualify_stage(child,out,kind)
            result['stages'].append(dict(stage=kind,worker_result_sha256=base.sha256(out/'worker_result.json'),release_complete=True))
            process.close();process=None
        check_frozen(frozen);update_probe.check_deadline(deadline)
        result.update(status='complete',diagnostic_qualification=True,seconds=time.monotonic()-started,
            conclusion='This fixed synthetic-target lifecycle diagnostic passed. Not a proven root cause, CV equivalence, repaired export or full-fit authorization.')
        for directory in [OUTPUT,REPORT]:write(directory/'result.json',result)
        update_probe.check_deadline(deadline)
        return result
    except BaseException as error:
        if process is not None and process.is_alive():process.kill();process.join(timeout=5)
        result.update(status='failed_inconclusive',diagnostic_qualification=False,error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        if isinstance(error,TimeoutError):write(OUTPUT/'external_timeout.json',dict(budget_seconds=BUDGET,child_reaped=process is None or not process.is_alive(),partial_evidence_preserved=True))
        for directory in [OUTPUT,REPORT]:write(directory/'result.json',result)
        raise
    finally:
        if process is not None and not process.is_alive():process.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    print(__import__('json').dumps(run(parser.parse_args().session_file),indent=2))
