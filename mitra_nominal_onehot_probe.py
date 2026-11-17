"""Externally supervised nominal19 resource proof; attributes and synthetic targets only."""
import argparse
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
import weakref

import joblib
import numpy as np
import pandas as pd

import mitra_nominal_onehot as core
import exaone_validation_data as data

base = core.base
ROOT = Path(__file__).resolve().parent
REVIEW = 'reports/top5_nominal_onehot_review.json'
REVIEW_SHA = 'b1a02ca0aac320b8388f0e09763e58eccbe0d6341cedb31909189081493e39af'
OUTPUT = ROOT/'artifacts/probes/mitra_nominal_onehot'
REPORT = ROOT/'reports/mitra_nominal_onehot_resources'
BUDGET = 1200
ATTRIBUTES = ['PassengerId','HomePlanet','CryoSleep','Destination','Age','VIP','RoomService',
              'FoodCourt','ShoppingMall','Spa','VRDeck','Cabin']
OWN = ['mitra_nominal_onehot.py','test_mitra_nominal_onehot.py','mitra_nominal_onehot_probe.py',
       'test_mitra_nominal_onehot_probe.py','quota_guard.py','exaone_validation_data.py',
       'audit_top5_references.py','three_family_mean_screen.py','prior_adapt_mean_screen.py',
       REVIEW,'reports/official_data_verification.json',data.PROOF]


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def check_deadline(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Fixed1200s nominal resource budget exhausted.')


def check_quota(session_file):
    from quota_guard import latest_weekly,assess
    try:allowed=assess(latest_weekly(session_file),reserve=30)['can_start_more_work']
    except (OSError,ValueError,TypeError,KeyError):allowed=False
    if not allowed:raise RuntimeError('Fresh reserve30 guard did not allow a new resource stage.')


def require_committed(names):
    names=sorted(set(names))
    tracked=subprocess.check_output(['git','ls-files','-z','--',*names],cwd=ROOT).decode()
    if (set(filter(None,tracked.split('\0')))!=set(names)
            or subprocess.check_output(['git','status','--porcelain','--',*names],cwd=ROOT).strip()):
        raise RuntimeError('Commit all resource sources/design/evidence before MPS.')


def check_frozen(frozen):
    for name,digest in {**frozen['sources'],**frozen['evidence_files']}.items():
        if base.sha256(ROOT/name)!=digest:raise RuntimeError('Frozen resource input changed: '+name)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=frozen['environment_sha256']:
        raise RuntimeError('Original raw Mitra environment changed.')
    base.verify_checkpoint()
    if frozen['settings']!=core.SETTINGS or frozen['backend']!=base.SETTINGS or frozen['checkpoint_sha256']!=base.WEIGHT_SHA:
        raise RuntimeError('Original pinned weights/backend or fixed nominal recipe changed.')


def contract():
    if base.sha256(ROOT/REVIEW)!=REVIEW_SHA:raise RuntimeError('Fixed nominal design changed.')
    review=read(ROOT/REVIEW);original=read(ROOT/'artifacts/run_038/protocol.json')
    if original['configs']['v71_Mitra2_zeroshot']['query_chunk']!=2318:
        raise RuntimeError('Original raw-Mitra full-support query chunk differs.')
    sources=dict(original['sources'])
    evidence={**review['source_sha256'],**review['source_review']}
    for name,digest in sources.items():
        if base.sha256(ROOT/name)!=digest:raise RuntimeError('Original raw Mitra source changed: '+name)
    sources.update({name:base.sha256(ROOT/name) for name in OWN})
    verification=read(ROOT/'reports/official_data_verification.json')
    for name in ['train.csv','test.csv']:
        evidence['data/raw/'+name]=verification['files'][name]['official_sha256']
    for name in ['artifacts/run_038/protocol.json','reports/run_038/protocol.json']:
        evidence[name]=base.sha256(ROOT/name)
    if evidence['artifacts/run_038/protocol.json']!=evidence['reports/run_038/protocol.json']:
        raise RuntimeError('Original raw protocol artifact/report differs.')
    lock=base.environment_lock()
    if hashlib.sha256(lock.encode()).hexdigest()!=original['mitra_environment_sha256']:
        raise RuntimeError('Use exactly the original raw-Mitra environment.')
    frozen=dict(sources=sources,evidence_files=evidence,environment_sha256=original['mitra_environment_sha256'],
        checkpoint_sha256=base.WEIGHT_SHA,backend=base.SETTINGS,settings=core.SETTINGS,review_sha256=REVIEW_SHA,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    check_frozen(frozen)
    require_committed([*sources,*[name for name in evidence if not name.startswith(('artifacts/','data/'))]])
    return frozen,lock


def attributes_only():
    frames=[pd.read_csv(ROOT/'data/raw'/name,usecols=ATTRIBUTES,dtype={'PassengerId':str}) for name in ['train.csv','test.csv']]
    if [len(frame) for frame in frames]!=[8693,4277] or any('Transported' in frame for frame in frames):
        raise RuntimeError('Complete attribute-only rosters required.')
    features=data.compact_features(pd.concat(frames,ignore_index=True))
    proof=read(ROOT/data.PROOF)
    if (proof.get('status')!='exact_compact13_match' or proof.get('holdout_targets_read') is not False
            or data.feature_hash(features)!=proof['compact_values_sha256']):
        raise RuntimeError('Attribute-only compact13 differs from baseline proof.')
    return features.iloc[:8693],features.iloc[8693:]


def snapshot(device):
    if device=='mps':base.torch.mps.synchronize()
    return base.snapshot(device)


def coverage(rows,queries):
    return [dict(support_rows=rows,label_rows=rows,query_rows=min(2318,queries-start)) for start in range(0,queries,2318)]


def predict(model,query):
    records=[]
    def before(module,args):
        records.append(dict(support_rows=int(args[0].shape[1]),label_rows=int(args[1].shape[1]),query_rows=int(args[2].shape[1])))
    engine=model.engine();hook=engine.register_forward_pre_hook(before)
    try:p=core.valid_probabilities(model.predict_proba(query),len(query))
    finally:hook.remove()
    if records!=coverage(len(model.labels),len(query)):raise RuntimeError('Actual full-support forward coverage changed.')
    return p.copy(),records


def same_predictions(actual,expected):
    core.valid_probabilities(actual,len(expected));core.valid_probabilities(expected,len(expected))
    difference=float(np.max(np.abs(actual-expected)))
    if difference>2e-6 or not np.array_equal(actual[:,1]>=.5,expected[:,1]>=.5):
        raise RuntimeError('Same-backend subset/reload probability or threshold mismatch.')
    return difference


def input_transfer_exact(array):
    transferred=base.torch.as_tensor(array,device='mps').cpu().numpy()
    if transferred.dtype!=array.dtype or transferred.shape!=array.shape or transferred.tobytes()!=array.tobytes():
        raise RuntimeError('CPU/MPS preprocessing input transfer is not bitwise identical.')


def tiny_case():
    x,q=base.synthetic_frame(128,42),base.synthetic_frame(33,43)
    x.loc[np.arange(7,128,29),'Destination']=None
    y=np.arange(128,dtype=np.int64)%2;model=None
    try:
        model=core.fit_nominal_onehot(x,y,np.arange(128),42,'cpu')
        if model.support.shape!=(128,19):raise RuntimeError('Tiny19-active-feature proof required.')
        support=model.support.copy();query=model.preprocessor.transform(q)
        cpu,cpu_records=predict(model,q)
        model.release();base.release_memory();model.device='mps'
        input_transfer_exact(support);input_transfer_exact(query)
        mps,mps_records=predict(model,q)
        difference=float(np.max(np.abs(cpu-mps)))
        if not np.isfinite(difference) or difference>.02:raise RuntimeError('Fixed tiny CPU/MPS probability tolerance failed.')
        return dict(status='complete',support_rows=128,query_rows=33,actual_retained_features=19,
            cpu_mps_input_exact=True,cpu_mps_max_difference=difference,threshold_equality_required=False,
            cpu_forward_records=cpu_records,mps_forward_records=mps_records,optimizer_steps=0)
    finally:
        if model is not None:model.release()
        model=None;base.release_memory()


def case(x,q,out):
    rows,queries=len(x),len(q);model=restored=None;started=time.monotonic()
    path=out/f'portable_{rows}.joblib'
    if path.exists():raise FileExistsError('Portable artifact exists; no overwrite.')
    y=np.arange(rows,dtype=np.int64)%2
    try:
        model=core.fit_nominal_onehot(x,y,np.arange(rows),42,'mps')
        support=model.support.copy();query=model.preprocessor.transform(q);schema=model.preprocessor.schema()
        if support.shape!=(rows,19) or query.shape!=(queries,19):raise RuntimeError('Exactly19 active full-support features required.')
        input_transfer_exact(support);input_transfer_exact(query)
        first,records=predict(model,q);small,subset_records=predict(model,q.iloc[:128])
        subset=same_predictions(small,first[:128])
        first_resources=snapshot('mps')
        joblib.dump(model,path);previous=weakref.ref(model)
        model.release();model=None;base.release_memory();gc.collect()
        if previous() is not None:raise RuntimeError('Original predictor retained across reload.')
        restored=joblib.load(path)
        if (restored.device!='cpu' or restored._engine is not None
                or any(t.device.type!='cpu' for t in restored.state.values())
                or restored.query_chunk!=2318 or restored.preprocessor.schema()!=schema):
            raise RuntimeError('Portable CPU schema/state/chunk differs.')
        np.testing.assert_array_equal(restored.support,support);np.testing.assert_array_equal(restored.labels,y)
        np.testing.assert_array_equal(restored.preprocessor.transform(q),query)
        restored.device='mps';second,reload_records=predict(restored,q)
        difference=same_predictions(second,first)
        result=dict(status='complete',support_rows=rows,query_rows=queries,actual_support_rows=len(support),
            actual_retained_features=19,full_support=True,optimizer_steps=0,cpu_mps_input_exact=True,schema=schema,
            transformed_support_sha256=hashlib.sha256(support.tobytes()).hexdigest(),
            transformed_query_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
            first_forward_records=records,subset_forward_records=subset_records,subset_max_difference=subset,subset_passed=True,
            reload_forward_records=reload_records,reload_max_difference=difference,reload_passed=True,
            reload_threshold_equal=True,portable_cpu_state=True,original_predictor_released=True,schema_reload_exact=True,
            portable_sha256=base.sha256(path),resources_after_first=first_resources,resources_after_reload=snapshot('mps'))
        restored.release();restored=None;base.release_memory()
        base.check_deadline();result['seconds']=time.monotonic()-started
        return result
    finally:
        for obj in [model,restored]:
            if obj is not None:obj.release()
        model=restored=None;base.release_memory()
        # The metadata/hash remains; disposable synthetic-label models are never uploaded.
        path.unlink(missing_ok=True)


def qualify(record):
    if (record.get('status')!='worker_complete' or record.get('release_complete') is not True
            or record.get('settings')!=core.SETTINGS or record.get('budget_seconds')!=BUDGET
            or record.get('no_accuracy_computed') is not True or record.get('real_targets_read') is not False):
        raise RuntimeError('Complete released score-free worker required.')
    tiny=record['tiny']
    if (tiny.get('status')!='complete' or tiny.get('support_rows')!=128 or tiny.get('query_rows')!=33
            or tiny.get('actual_retained_features')!=19 or tiny.get('cpu_mps_input_exact') is not True
            or tiny.get('optimizer_steps')!=0 or tiny.get('threshold_equality_required') is not False
            or not np.isfinite(tiny['cpu_mps_max_difference']) or not 0<=tiny['cpu_mps_max_difference']<=.02
            or tiny['cpu_forward_records']!=coverage(128,33) or tiny['mps_forward_records']!=coverage(128,33)):
        raise RuntimeError('Complete tiny CPU/MPS native-model comparison required.')
    if len(record['cases'])!=2:raise RuntimeError('Both full resource cases required.')
    for row,(rows,queries) in zip(record['cases'],[(4636,2318),(8693,4277)]):
        if (row.get('status')!='complete' or row.get('support_rows')!=rows or row.get('query_rows')!=queries
                or row.get('actual_support_rows')!=rows or row.get('actual_retained_features')!=19 or row.get('optimizer_steps')!=0
                or row['schema']['actual_retained_features']!=19 or row['schema']['original11_exact'] is not True
                or row['schema']['mirror_rule']!=core.MIRROR_RULE
                or any(row.get(k) is not True for k in ['full_support','cpu_mps_input_exact','subset_passed','reload_passed',
                    'reload_threshold_equal','portable_cpu_state','original_predictor_released','schema_reload_exact'])
                or row['first_forward_records']!=coverage(rows,queries) or row['reload_forward_records']!=coverage(rows,queries)
                or row['subset_forward_records']!=coverage(rows,128)
                or any(not np.isfinite(row[k]) or not 0<=row[k]<=2e-6 for k in ['subset_max_difference','reload_max_difference'])):
            raise RuntimeError('Complete19-feature full-context subset/reload resource proof required.')
    return True


def worker(out,report,frozen,deadline):
    out,report=Path(out),Path(report);base.DEADLINE=deadline
    result=dict(status='incomplete',settings=core.SETTINGS,budget_seconds=BUDGET,no_accuracy_computed=True,
        real_targets_read=False,real_attributes_used=True,synthetic_alternating_labels=True,cases=[],release_complete=False)
    def persist():
        for directory in [out,report]:write(directory/'partial.json',result)
    try:
        check_deadline(deadline);check_frozen(frozen)
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0') or not base.torch.backends.mps.is_available():
            raise RuntimeError('MPS without fallback required.')
        base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)
        result['allocator_limit_bytes']=int(base.torch.mps.recommended_max_memory()*.65)
        result['current_stage']='tiny_cpu_mps';persist();result['tiny']=tiny_case();persist()
        train,test=attributes_only()
        for x,q in [(train.iloc[:4636],train.iloc[4636:6954]),(train,test)]:
            check_deadline(deadline);check_frozen(frozen)
            result['current_stage']=f'{len(x)}x{len(q)}';persist()
            completed=case(x,q,out);result['cases'].append(completed)
            for directory in [out,report]:write(directory/f'case_{len(x)}.json',completed)
            persist()
        check_frozen(frozen);check_deadline(deadline)
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error));persist()
    finally:
        try:
            base.release_memory();result['resources_after_release']=snapshot('mps');check_deadline(deadline)
            result['release_complete']=True
            if result['status']=='incomplete':result['status']='worker_complete';qualify(result)
        except BaseException as error:
            result.update(status='failed_inconclusive',release_complete=False,cleanup_error_type=type(error).__name__,cleanup_error=str(error))
            if 'error_type' not in result:result.update(error_type=type(error).__name__,error=str(error))
        persist()
        for directory in [out,report]:write(directory/'worker_result.json',result)


def supervise(process,deadline):
    while process.is_alive() and time.monotonic()<deadline:
        process.join(timeout=min(5,max(0,deadline-time.monotonic())))
    if process.is_alive():
        process.kill();process.join(timeout=5)
        if process.is_alive():raise RuntimeError('Resource worker could not be reaped.')
        raise TimeoutError('External supervisor killed nominal probe at fixed1200s deadline.')
    if process.exitcode!=0:raise RuntimeError('Resource worker exited abnormally: '+str(process.exitcode))


def run(session_file):
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError('No overwrite/resume of resource evidence.')
    check_quota(session_file);started=time.monotonic();deadline=started+BUDGET
    OUTPUT.mkdir(parents=True);REPORT.mkdir(parents=True);process=None
    result=dict(status='failed_inconclusive',resource_qualification=False,settings=core.SETTINGS,
        budget_seconds=BUDGET,no_accuracy_computed=True,real_targets_read=False,externally_supervised=True)
    try:
        frozen,lock=contract()
        for directory in [OUTPUT,REPORT]:write(directory/'freeze.json',frozen);(directory/'requirements.lock').write_text(lock)
        for name in frozen['sources']:
            target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        check_deadline(deadline)
        process=multiprocessing.get_context('spawn').Process(target=worker,args=(str(OUTPUT),str(REPORT),frozen,deadline))
        process.start();supervise(process,deadline);check_deadline(deadline)
        child=read(OUTPUT/'worker_result.json')
        if child.get('status')!='worker_complete':
            result.update(worker_error_type=child.get('error_type'),worker_error=child.get('error'),worker_current_stage=child.get('current_stage'))
            raise RuntimeError('Resource worker failed: '+str(child.get('error_type'))+': '+str(child.get('error')))
        qualify(child);check_frozen(frozen);check_deadline(deadline)
        result.update(child,status='complete',resource_qualification=True,
            worker_result_sha256=base.sha256(OUTPUT/'worker_result.json'),seconds=time.monotonic()-started)
        for directory in [OUTPUT,REPORT]:write(directory/'result.json',result)
        check_deadline(deadline)
        return result
    except BaseException as error:
        if process is not None and process.is_alive():process.kill();process.join(timeout=5)
        result.update(status='failed_inconclusive',resource_qualification=False,error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        if isinstance(error,TimeoutError):
            write(OUTPUT/'external_timeout.json',dict(status='external_deadline_exceeded',budget_seconds=BUDGET,
                child_reaped=process is None or not process.is_alive(),partial_evidence_preserved=True))
        for directory in [OUTPUT,REPORT]:write(directory/'result.json',result)
        raise
    finally:
        if process is not None and not process.is_alive():process.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().session_file),indent=2))
