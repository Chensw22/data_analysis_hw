"""Externally supervised fixed support3072 development screen; no export/upload."""
import argparse
import importlib
import json
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
import mitra_full_episode as old_core
import mitra_long_context_probe as probe
import exaone_validation_data as data  # Pure attributes/cache audit only, no EXAONE runtime.

ROOT=Path(__file__).resolve().parent
NAME='v93_Mitra_support3072'
REFERENCE='v81_Mitra_full_episode50'
RUN='run_056'
PLAN='reports/mitra_support3072_screen_review.json'
PLAN_SHA='e90165ae596e2d251b775cf6fba4f073281ada655ce13b147cebdf498fb8341a'
RESOURCE='reports/mitra_long_context_resources'
BUDGET=18000
SETTINGS={**old_core.SETTINGS,'episode_support':3072,'sampler':probe.RECIPE['sampler']}
OWN=['mitra_support3072_screen.py','test_mitra_support3072_screen.py','mitra_support3072.py','test_mitra_support3072.py',
    'exaone_validation_data.py','audit_top5_references.py','three_family_mean_screen.py','prior_adapt_mean_screen.py',data.PROOF,PLAN]
DISCLOSURE='Adaptive development after earlier experiments and online v81; reused folds, not independent confirmation. Same maximum query-target updates, larger FLOPs; selected refit counts may differ. No S claim.'


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);probe.write(path,value)


def resource_contract():
    if base.sha256(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Fixed prospective support3072 plan changed.')
    plan=probe.read(ROOT/PLAN);evidence={PLAN:PLAN_SHA}
    for name,digest in plan['resource_evidence'].items():
        if base.sha256(ROOT/name)!=digest:raise RuntimeError('Committed resource evidence changed: '+name)
        evidence[name]=digest
    probe.require_committed(evidence)
    frozen,lock=probe.contract();saved=probe.read(ROOT/RESOURCE/'freeze.json')
    for key in ['recipe','sources','design_sha256','checkpoint_sha256','backend','environment_sha256']:
        if saved[key]!=frozen[key]:raise RuntimeError('Resource source/environment contract differs: '+key)
    result=probe.read(ROOT/RESOURCE/'result.json');worker=probe.read(ROOT/RESOURCE/'worker_result.json')
    cost=probe.qualify(worker)
    if (result.get('status')!='complete' or result.get('resource_qualification') is not True
            or result.get('cost_qualified_for_root_review') is not True or cost.get('passed') is not True
            or result.get('synthetic_only') is not True or result.get('accuracy_computed') is not False
            or result.get('externally_supervised') is not True or result.get('release_complete') is not True
            or result.get('recipe')!=probe.RECIPE or result.get('cost_gate')!=cost
            or cost!=plan['cost_estimates_seconds'] or result.get('worker_result_sha256')!=base.sha256(ROOT/RESOURCE/'worker_result.json')
            or result.get('updates')!=worker['updates'] or result.get('validations')!=worker['validations']):
        raise RuntimeError('Complete matching cost-qualified synthetic resource evidence required before targets.')
    original=ROOT/'artifacts/probes/mitra_long_context_3072'
    for name in ['freeze.json','result.json','worker_result.json','requirements.lock']:
        if base.sha256(original/name)!=base.sha256(ROOT/RESOURCE/name):raise RuntimeError('Resource artifact/report differs.')
        evidence[str((original/name).relative_to(ROOT))]=base.sha256(original/name)
    # Authenticate reference protocol and historical checkpoint/environment before parsing targets.
    _,files=data.common.verify_protocol(ROOT/'artifacts/run_046');evidence.update(files)
    for name in ['protocol.json','selection.json','screen_results.csv','full_episode_environment.lock.txt']:
        data.report_pair('run_046',name,evidence)
    if (ROOT/'artifacts/run_046/full_episode_environment.lock.txt').read_text()!=lock:
        raise RuntimeError('Reference and qualified candidate environments differ.')
    return dict(resource=frozen,sources={},evidence=evidence),lock


def assert_frozen(frozen):
    probe.check_frozen(frozen['resource'])
    for name,digest in {**frozen['sources'],**frozen['evidence']}.items():
        if base.sha256(ROOT/name)!=digest:raise RuntimeError('Screen source/evidence changed: '+name)


def paired_rows(rows,old_rows,ids,seed,new):
    rng=np.random.default_rng(seed);old_by_step={r['step']:r for r in old_rows}
    for step,row in enumerate(rows,1):
        if row.get('step')!=step:raise RuntimeError('Incomplete/reordered training episode history.')
        perm=rng.permutation(len(ids));support=np.concatenate([perm[:1024],perm[1280:3328]]);query=perm[1024:1280]
        if len(ids)<3328:raise RuntimeError('Full distinct3072 support and256 query required.')
        expected=dict(query_ids_sha256=base.ids_hash(ids[query]))
        if new:
            expected.update(support_ids_sha256=base.ids_hash(ids[support]),support_prefix_ids_sha256=base.ids_hash(ids[perm[:1024]]),
                local_support_ids_sha256=base.ids_hash(support),local_query_ids_sha256=base.ids_hash(query))
            if any(row.get(k) is not True for k in ['query_equal_v81','support_prefix_equal_v81','rng_state_equal_v81']):
                raise RuntimeError('Paired query/prefix/RNG counterfactual evidence missing.')
        else:expected['support_ids_sha256']=base.ids_hash(ids[perm[:1024]])
        if (any(row.get(k)!=v for k,v in expected.items()) or row.get('support_rows')!=(3072 if new else 1024)
                or row.get('query_rows')!=256 or row.get('finite_gradients') is not True or row.get('disjoint_ids') is not True
                or row.get('cumulative_optimizer_steps')!=step
                or abs(row['learning_rate']-base.literal_learning_rate(step))>1e-15):
            raise RuntimeError('Actual global IDs/schedule/episode dimensions differ from paired fixed sampler.')
        if step in old_by_step and new:
            previous=old_by_step[step]
            if row['query_ids_sha256']!=previous['query_ids_sha256'] or row['support_prefix_ids_sha256']!=previous['support_ids_sha256']:
                raise RuntimeError('New update differs from actual old v81 query/support-prefix.')


def check_log(log,old,train_ids,query_ids,labels,seed,new):
    data.adapted.check_adaptation_log(log,train_ids,query_ids,labels,seed)
    inner,valid=train_test_split(np.arange(len(labels)),test_size=.2,stratify=labels,random_state=seed)
    if (log.get('recipe')!=(SETTINGS if new else old_core.SETTINGS) or log.get('checkpoint_sha256')!=base.WEIGHT_SHA
            or log.get('inner_training_rows')!=len(inner) or log.get('inner_validation_rows')!=len(valid)
            or log.get('query_chunk')!=2318 or log.get('parameters')!=75670026):
        raise RuntimeError('Full-parameter context/selection/checkpoint recipe differs.')
    if new and ([r.get('stage') for r in log['inner_history']]!=['inner']*51
            or any(r.get('stage')!='refit' for r in log['refit_history'])):
        raise RuntimeError('Explicit inner/refit stage identity required.')
    if new:
        for step,row in enumerate(log['inner_history']):
            if (row.get('validation_support_rows')!=len(inner) or row.get('validation_query_rows')!=len(valid)
                    or row.get('validation_full_support') is not True
                    or row.get('validation_optimizer_steps')!=([] if step==0 else [step])
                    or row.get('validation_optimizer_state_tensor_bytes')!=(0 if step==0 else probe.OPTIMIZER_BYTES)):
                raise RuntimeError('Full-context inner validation with persistent optimizer required at all51 stages.')
    paired_rows(log['inner_history'][1:],old['inner_history'][1:],train_ids[inner],seed,new)
    paired_rows(log['refit_history'],old['refit_history'],train_ids,seed,new)


def reference_inputs(frozen):
    inputs=data.load_data();splits,references,evidence=data.random_references(inputs)
    frozen['evidence'].update(inputs['evidence']);frozen['evidence'].update(evidence);logs=[]
    for fold,(a,b) in enumerate(splits):
        name=f'{REFERENCE}_seed{20261112+fold}_{base.ids_hash(inputs["dev"][a])[:12]}.json'
        old=probe.read(data.report_pair('run_046',name,frozen['evidence']))
        check_log(old,old,inputs['dev'][a],inputs['dev'][b],inputs['y'][a],20261112+fold,False);logs.append(old)
    reference=references[REFERENCE];score=data.metrics(reference,inputs['y'],splits)
    if abs(score['accuracy']-.817946505608283)>1e-12 or abs(score['log_loss']-.3750928747453614)>1e-12:
        raise RuntimeError('Primary frozen v81 control score changed.')
    probe.require_committed([p for p in frozen['evidence'] if p.startswith('reports/')]);assert_frozen(frozen)
    return inputs,splits,reference,logs


def continuation(candidate,reference):
    folds=np.asarray(candidate['fold_accuracy'])+1e-12>=np.asarray(reference['fold_accuracy'])
    if folds.shape!=(3,):raise RuntimeError('All3 folds required.')
    checks=dict(pooled_accuracy=candidate['accuracy']+1e-12>=reference['accuracy'],
        pooled_log_loss=candidate['log_loss']<=reference['log_loss']+1e-12,
        two_nonlower_folds=int(folds.sum())>=2,
        strict_pooled_improvement=candidate['accuracy']>reference['accuracy']+1e-12 or candidate['log_loss']<reference['log_loss']-1e-12)
    return dict(passed=all(checks.values()),checks=checks,nonlower_folds=int(folds.sum()),S_supported=False,automatic_export=False,automatic_upload=False)


def paired(prob,reference,y,splits):
    def one(indices):
        a=prob[indices]>=.5;b=reference[indices]>=.5;t=y[indices]
        return dict(corrected=int(np.sum((a==t)&(b!=t))),regressed=int(np.sum((a!=t)&(b==t))),
            disagreements=int(np.sum(a!=b)),accuracy_delta=float(np.mean(a==t)-np.mean(b==t)))
    return dict(overall=one(np.arange(len(y))),folds=[one(b) for _,b in splits])


def train_folds(out,report,inputs,splits,reference,old_logs,frozen,deadline,session_file):
    core=importlib.import_module('mitra_support3072')
    if core.SETTINGS!=SETTINGS:raise RuntimeError('New core differs from fixed support-only recipe.')
    dev,y=inputs['dev'],inputs['y'];prob=np.full(len(dev),np.nan);fold_ids=np.full(len(dev),-1,dtype=np.int64)
    records=[]
    for fold,(a,b) in enumerate(splits):
        probe.quota_guard(session_file);probe.check_deadline(deadline);assert_frozen(frozen)
        model=None;seed=20261112+fold;tick=time.monotonic();log_path=out/f'{NAME}_seed{seed}_{base.ids_hash(dev[a])[:12]}.json'
        if log_path.exists():raise RuntimeError('Existing fit log: no implicit resume.')
        try:
            model=core.fit_support3072(inputs['features'].iloc[dev[a]],y[a],dev[a],seed,device='mps',log_path=log_path,max_steps=50)
            if (len(model.support)!=len(a) or model.query_chunk!=2318 or not np.array_equal(model.labels,y[a])):
                raise RuntimeError('Original full-support prediction context required.')
            predicted=np.asarray(model.predict_proba(inputs['features'].iloc[dev[b]]))
            if (predicted.shape!=(len(b),2) or not np.isfinite(predicted).all() or np.any((predicted<0)|(predicted>1))
                    or not np.allclose(predicted.sum(axis=1),1,rtol=0,atol=2e-6)):
                raise RuntimeError('Incomplete/nonfinite final predictions.')
            metadata=dict(model.metadata,prediction_ids_sha256=base.ids_hash(dev[b]),prediction_rows=len(b),
                final_prediction_path='original_MitraPredictor',automatic_further_extension_allowed=False)
            expected_sources={name:{**frozen['resource']['sources'],**frozen['sources']}[name] for name in core.SOURCES}
            if metadata.get('source_hashes')!=expected_sources or metadata.get('environment_sha256')!=frozen['resource']['environment_sha256']:
                raise RuntimeError('Actual fit source/environment differs from frozen screen.')
            check_log(metadata,old_logs[fold],dev[a],dev[b],y[a],seed,True)
            write(log_path,metadata)
            forward_resources=probe.synchronized_snapshot('mps')
        finally:
            if model is not None:model.release()
            model=None;base.release_memory()
        probe.check_deadline(deadline);assert_frozen(frozen)
        prob[b]=predicted[:,1];fold_ids[b]=fold
        metrics=data.metrics(prob[b],y[b],[(np.array([],dtype=int),np.arange(len(b)))])
        matched=data.metrics(reference[b],y[b],[(np.array([],dtype=int),np.arange(len(b)))])
        record=dict(status='complete',fold=fold,seed=seed,metrics=metrics,reference=matched,selected_steps=metadata['selected_steps'],
            train_ids_sha256=base.ids_hash(dev[a]),validation_ids_sha256=base.ids_hash(dev[b]),
            paired=paired(prob[b],reference[b],y[b],[(np.array([],dtype=int),np.arange(len(b)))]),
            paired_inner_steps=50,paired_common_refit_steps=min(metadata['selected_steps'],old_logs[fold]['selected_steps']),
            training_log_sha256=base.sha256(log_path),seconds=time.monotonic()-tick,
            resources_after_forward=forward_resources,resources_after_release=probe.synchronized_snapshot('mps'))
        for folder in [out,report]:write(folder/f'fold{fold}.json',record)
        shutil.copy2(log_path,report/log_path.name)
        np.savez(out/f'fold{fold}_rows.npz',train_ids=dev[a],validation_ids=dev[b])
        np.savez(out/'partial_oof.npz',prob=prob,dev_ids=dev,fold_id=fold_ids)
        records.append(record)
        for folder in [out,report]:write(folder/'progress.json',dict(status='incomplete',completed_folds=len(records),folds=records,partial_not_a_winner=True))
        print(json.dumps(dict(fold=fold,accuracy=metrics['accuracy'],log_loss=metrics['log_loss'],reference_accuracy=matched['accuracy'],
            reference_log_loss=matched['log_loss'],selected_steps=metadata['selected_steps'],seconds=record['seconds'])),flush=True)
    if len(records)!=3 or np.any(fold_ids<0):raise RuntimeError('Three complete folds required.')
    return dict(status='worker_complete',completed_folds=3,metrics=data.metrics(prob,y,splits),reference=data.metrics(reference,y,splits),folds=records)


def worker(out,report,inputs,splits,reference,logs,frozen,deadline,session_file):
    out,report=Path(out),Path(report);base.DEADLINE=deadline
    result=dict(status='failed_inconclusive',release_complete=False)
    try:
        probe.check_deadline(deadline);assert_frozen(frozen)
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0') or not base.torch.backends.mps.is_available():
            raise RuntimeError('MPS without implicit CPU fallback required.')
        base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)
        result=train_folds(out,report,inputs,splits,reference,logs,frozen,deadline,session_file)
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
        logs=sorted(out.glob(f'{NAME}_seed*.json'))
        if logs:
            try:result['current_stage']=probe.read(logs[-1]).get('current_stage')
            except (OSError,ValueError):pass
    finally:
        try:
            base.release_memory();result['resources_after_release']=probe.synchronized_snapshot('mps')
            probe.check_deadline(deadline);result['release_complete']=True
        except BaseException as error:result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),release_complete=False)
        write(out/'worker_result.json',result)


def supervise(process,deadline):
    while process.is_alive() and time.monotonic()<deadline:
        process.join(timeout=min(5,max(0,deadline-time.monotonic())))
    if process.is_alive():
        process.kill();process.join(timeout=5)
        if process.is_alive():raise RuntimeError('Screen worker could not be reaped.')
        raise TimeoutError('External supervisor killed support3072 screen at fixed18000s deadline.')
    if process.exitcode!=0:raise RuntimeError('Screen worker exited abnormally: '+str(process.exitcode))


def publish(out,report,inputs,splits,reference,child,frozen,deadline):
    if child.get('status')!='worker_complete' or child.get('completed_folds')!=3 or child.get('release_complete') is not True:
        raise RuntimeError('Complete3-fold released worker required; partial results are not a winner. '+str(child.get('error_type',''))+': '+str(child.get('error','')))
    with np.load(out/'partial_oof.npz',allow_pickle=False) as cache:
        prob=cache['prob'].copy();dev=cache['dev_ids'];fold_ids=cache['fold_id']
    if not np.array_equal(dev,inputs['dev']) or any(not np.array_equal(np.flatnonzero(fold_ids==k),b) for k,(_,b) in enumerate(splits)):
        raise RuntimeError('Worker OOF identity differs.')
    score=data.metrics(prob,inputs['y'],splits);ref=data.metrics(reference,inputs['y'],splits)
    if score!=child['metrics'] or ref!=child['reference']:raise RuntimeError('Worker final metrics differ from cache.')
    assessment=dict(status='complete',candidate=NAME,completed_folds=3,metrics=score,reference=ref,continuation=continuation(score,ref),
        paired=paired(prob,reference,inputs['y'],splits),disclosure=DISCLOSURE,old_holdout_labels_read=False,S_supported=False,
        automatic_export=False,automatic_upload=False,externally_supervised=True,budget_seconds=BUDGET)
    assert_frozen(frozen);probe.check_deadline(deadline)
    cache=out/f'{NAME}_seed20261112.npz';np.savez(cache,prob=prob,fold_accuracy=score['fold_accuracy'],dev_ids=dev,fold_id=fold_ids)
    table=pd.DataFrame([dict(version=NAME,accuracy=score['accuracy'],log_loss=score['log_loss'],fold_std=float(np.std(score['fold_accuracy'])),seconds=BUDGET-max(0,deadline-time.monotonic()))])
    for folder in [out,report]:
        table.to_csv(folder/'screen_results.csv',index=False);write(folder/'assessment.json',assessment)
        write(folder/'selection.json',dict(version=NAME,screen_accuracy=score['accuracy'],continuation_passed=assessment['continuation']['passed'],
            stage='screen_only_pending_separate_decision' if assessment['continuation']['passed'] else 'archived_failed_fixed_gate',automatic_eligibility=False))
    assessment['result_files_sha256']={p.name:base.sha256(p) for p in [cache,out/'screen_results.csv',out/'assessment.json',out/'selection.json',out/'worker_result.json']}
    for folder in [out,report]:write(folder/'summary.json',assessment);write(folder/'progress.json',assessment)
    assert_frozen(frozen);probe.check_deadline(deadline)
    return assessment


def run(session_file):
    out,report=ROOT/'artifacts'/RUN,ROOT/'reports'/RUN
    if out.exists() or report.exists():raise RuntimeError('No overwrite/resume of prior screen evidence.')
    probe.quota_guard(session_file);started=time.monotonic();deadline=started+BUDGET
    frozen,lock=resource_contract()  # Complete resource/source/environment checks before any targets.
    probe.require_committed(OWN);frozen['sources']={name:base.sha256(ROOT/name) for name in OWN}
    core=importlib.import_module('mitra_support3072')
    if core.SETTINGS!=SETTINGS:raise RuntimeError('Fixed support3072 core recipe differs.')
    inputs,splits,reference,logs=reference_inputs(frozen);probe.check_deadline(deadline)
    out.mkdir(parents=True);report.mkdir(parents=True);process=None
    try:
        protocol=dict(candidate=NAME,seed=20261112,folds=3,development_rows=6954,threshold=.5,settings=SETTINGS,
            budget_seconds=BUDGET,externally_supervised=True,plan_sha256=PLAN_SHA,frozen=frozen,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            development_ids_sha256=base.ids_hash(inputs['dev']),development_labels_sha256=base.ids_hash(inputs['y']),
            disclosure=DISCLOSURE,old_holdout_labels_read=False,automatic_export=False,automatic_upload=False)
        for folder in [out,report]:write(folder/'protocol.json',protocol);(folder/'requirements.lock').write_text(lock)
        for name in {**frozen['resource']['sources'],**frozen['sources']}:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        process=multiprocessing.get_context('spawn').Process(target=worker,args=(str(out),str(report),inputs,splits,reference,logs,frozen,deadline,session_file))
        process.start();supervise(process,deadline);probe.check_deadline(deadline)
        return publish(out,report,inputs,splits,reference,probe.read(out/'worker_result.json'),frozen,deadline)
    except BaseException as error:
        if process is not None and process.is_alive():process.kill();process.join(timeout=5)
        failed=dict(status='failed_inconclusive',candidate=NAME,error_type=type(error).__name__,error=str(error),
            seconds=time.monotonic()-started,budget_seconds=BUDGET,externally_supervised=True,
            partial_not_a_winner=True,automatic_export=False,automatic_upload=False,S_supported=False)
        if (out/'worker_result.json').exists():
            try:
                child=probe.read(out/'worker_result.json')
                failed.update(worker_error_type=child.get('error_type'),worker_error=child.get('error'),
                    worker_current_stage=child.get('current_stage'),worker_status=child.get('status'))
            except (OSError,ValueError):pass
        for folder in [out,report]:
            for name in [f'{NAME}_seed20261112.npz','screen_results.csv','assessment.json','selection.json']:(folder/name).unlink(missing_ok=True)
            write(folder/'summary.json',failed);write(folder/'progress.json',failed)
        raise
    finally:
        if process is not None and not process.is_alive():process.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    args=parser.parse_args();run(args.session_file)
