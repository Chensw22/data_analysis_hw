"""Two separate, prospectively frozen EXAONE screens; never uploads or reads holdout labels."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd

import exaone_candidate as core
import exaone_resource_probe as resource
import exaone_validation_data as data

ROOT=Path(__file__).resolve().parent
PLAN='reports/exaone_screen_review.json'
PLAN_SHA='815146bf7ba6b7a47bf683ebf556c29de0066d6416c648902f0a7ed4ecd581e0'
RESOURCE='reports/exaone_resources/mps/result.json'
OWN=['exaone_validation.py','exaone_validation_data.py','test_exaone_validation.py',
     'audit_top5_references.py','three_family_mean_screen.py','prior_adapt_mean_screen.py',PLAN,data.PROOF]
DIRECTORIES={'random':'run_054','group':'exaone_group_20261117'}
BUDGET=1800


def coverage(records,rows,queries):
    expected=Counter()
    for start in range(0,queries,1024):expected[min(1024,queries-start)]+=8
    if (Counter(r['query_rows'] for r in records)!=expected
            or any(r.get('support_rows')!=rows or r.get('views')!=1 or r.get('input_dtype')!='float32' for r in records)):
        raise RuntimeError('Full support/eight-view/query coverage differs from frozen resource recipe.')


def bounded(value,limit):
    if not np.isfinite(value) or not 0<=value<=limit:raise RuntimeError('Resource numerical comparison failed.')


def reload_gate(record,rows,queries):
    if (record.get('query_rows')!=queries or record.get('full_support_rows')!=rows
            or any(record.get(k) is not True for k in ['same_threshold_decisions','restored_cpu_context','preprocessing_exact'])):
        raise RuntimeError('Complete portable reload proof required.')
    bounded(record['max_difference'],2e-6);coverage(record['forward_records'],rows,queries)


def validate_resource(record,current,lock):
    if (record.get('status')!='complete' or record.get('mode')!='mps'
            or record.get('resource_qualification') is not True or record.get('accuracy_computed') is not False
            or record.get('all_data_synthetic') is not True or record.get('budget_seconds')!=2400
            or record.get('backend')!='native_default_device_SDPA_in_locked_PyTorch'):
        raise RuntimeError('Completed score-free MPS resource qualification required before labels or scoring.')
    for key in ['sources','environment_sha256','checkpoint_sha256','checkpoint_bytes','settings','design_sha256']:
        if record.get(key)!=current[key]:raise RuntimeError('Qualified resource identity differs: '+key)
    if hashlib.sha256(lock.encode()).hexdigest()!=record['environment_sha256']:raise RuntimeError('Resource environment lock differs.')
    stages=record['stages'];expected=[('AB_native_and_local_CPU',128,33),('C_tiny',128,33),
        ('D_screen',4636,2318),('E_full_support',8693,2318),('F_full_export',8693,4277)]
    if len(stages)!=len(expected):raise RuntimeError('All resource stages required.')
    for r,(name,rows,queries) in zip(stages,expected):
        if any(r.get(k)!=v for k,v in dict(stage=name,status='complete',support_rows=rows,query_rows=queries,views=8).items()):
            raise RuntimeError('Resource stage identity/coverage differs.')
        if name.startswith('AB'):
            if r.get('same_native_cpu_preprocessing') is not True or r['model'].get('native_strict_state_keys_shapes_values') is not True:
                raise RuntimeError('Original CPU preprocessing and strict checkpoint proof required.')
            bounded(r['original_vs_adapter_max_difference'],.002);bounded(r['partition_max_difference'],.002)
        else:
            if r.get('full_support') is not True:raise RuntimeError('Resource support truncation forbidden.')
            coverage(r['initial_forward_records'],rows,queries)
        if name=='C_tiny':
            bounded(r['cpu_vs_mps_max_difference'],.002);bounded(r['repeat_max_difference'],2e-6)
            bounded(r['partition_max_difference'],.002)
        if name in ['AB_native_and_local_CPU','C_tiny','F_full_export']:reload_gate(r['portable_reload'],rows,queries)


def resource_gate():
    if core.sha(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Frozen prospective screen plan changed.')
    data.common.require_committed(OWN+[RESOURCE,'reports/exaone_resources/mps/environment.lock.txt'])
    current,lock=resource.contract(require_committed=True)
    report=ROOT/RESOURCE;local=ROOT/'artifacts/probes/exaone_mps/result.json'
    if core.sha(report)!=core.sha(local):raise RuntimeError('Resource artifact/report mismatch.')
    stored=(report.parent/'environment.lock.txt').read_text()
    if stored!=lock or (local.parent/'environment.lock.txt').read_text()!=lock:raise RuntimeError('Resource environment changed.')
    validate_resource(json.loads(report.read_text()),current,lock)
    sources={name:core.sha(ROOT/name) for name in OWN}
    evidence={str(p.relative_to(ROOT)):core.sha(p) for p in [report,local,report.parent/'environment.lock.txt',local.parent/'environment.lock.txt']}
    return dict(resource=current,sources=sources,evidence=evidence),lock


def assert_frozen(frozen):
    resource.assert_frozen(frozen['resource'])
    for name,digest in {**frozen['sources'],**frozen['evidence']}.items():
        if core.sha(ROOT/name)!=digest:raise RuntimeError('Validation input changed: '+name)


def random_gate(evidence):
    directory='run_054'
    for name in ['summary.json','protocol.json','screen_results.csv','assessment.json']:
        data.report_pair(directory,name,evidence)
    folder=ROOT/'artifacts'/directory
    summary=json.loads((folder/'summary.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    assessment=json.loads((folder/'assessment.json').read_text())
    if (summary.get('status')!='complete' or summary.get('completed_folds')!=3 or protocol.get('phase')!='random'
            or protocol.get('plan_sha256')!=PLAN_SHA or summary.get('candidate')!=core.NAME
            or summary.get('continue_to_group') is not True or assessment.get('passed') is not True):
        raise RuntimeError('Complete committed random A-band screen required before group screen.')
    assert_frozen(protocol['frozen'])
    frame=pd.read_csv(folder/'screen_results.csv');rows=frame[frame.version==core.NAME]
    if len(rows)!=1 or float(rows.iloc[0].accuracy)+1e-12<.8098159332758125:
        raise RuntimeError('Random screen below fixed A band.')
    for name,digest in summary['result_files_sha256'].items():
        path=folder/name
        if core.sha(path)!=digest:raise RuntimeError('Random completed result changed.')
        data.remember(path,evidence)


def paired(p,reference,y,splits):
    def one(indices):
        a=p[indices]>=.5;b=reference[indices]>=.5;t=y[indices]
        return dict(corrected=int(np.sum((a==t)&(b!=t))),regressed=int(np.sum((a!=t)&(b==t))),
            decision_disagreements=int(np.sum(a!=b)),mean_absolute_probability_difference=float(np.mean(np.abs(p[indices]-reference[indices]))))
    return dict(overall=one(np.arange(len(y))),folds=[one(b) for _,b in splits])


def assess(phase,score,references):
    if phase=='random':checks={'accuracy_within_fixed_A_band':score['accuracy']+1e-12>=.8098159332758125}
    else:
        anchors={'v71_Mitra2_zeroshot':.8169398907103825,'v44_seed_ensemble':.814495254529767}
        if any(abs(references[k]['accuracy']-v)>1e-12 for k,v in anchors.items()):raise RuntimeError('Matched group anchors changed.')
        checks={k:score['accuracy']+1e-12>=v-.01 for k,v in anchors.items()}
    return dict(passed=all(checks.values()),checks=checks,S_supported=False,automatic_upload=False,
        meaning='Exploratory eligibility only; reused development folds, no independent confirmation or stable superiority.')


def evaluate(inputs,splits,references,out,report,frozen,phase):
    dev,y=inputs['dev'],inputs['y'];prob=np.full(len(dev),np.nan);fold_id=np.full(len(dev),-1,dtype=np.int64)
    records=[];started=time.monotonic()
    final_names=[f'{core.NAME}_seed20261112.npz','screen_results.csv','group_metrics.csv','assessment.json']
    def write(name,value):
        for folder in [out,report]:resource.write(folder/name,value)
    try:
        for fold,(a,b) in enumerate(splits):
            assert_frozen(frozen);core.check_deadline();model=None;tick=time.monotonic()
            try:
                model=core.fit_context(inputs['features'].iloc[dev[a]],y[a],dev[a],seed=0,device='mps')
                predicted=core.valid_probabilities(model.predict_proba(inputs['features'].iloc[dev[b]]),len(b))
                metadata=model.metadata
                if (metadata['training_ids_sha256']!=core.array_hash(dev[a]) or metadata['labels_sha256']!=core.array_hash(y[a].astype(np.int64))
                        or metadata['training_rows']!=len(a) or metadata['settings']!=core.SETTINGS
                        or metadata['preprocessing']['support_rows']!=len(a) or metadata['preprocessing']['views']!=8
                        or metadata['last_prediction']['probabilities_sha256']!=core.array_hash(predicted)):
                    raise RuntimeError('Actual fold context/labels/recipe identity differs.')
                coverage(metadata['last_prediction']['forward_records'],len(a),len(b))
                before_release=resource.snapshot('mps')
            finally:
                if model is not None:model.release()
                model=None;gc.collect()
            core.check_deadline();assert_frozen(frozen)
            prob[b]=predicted[:,1];fold_id[b]=fold
            score=data.metrics(prob[b],y[b],[(np.array([],dtype=int),np.arange(len(b)))])
            record=dict(fold=fold,status='complete',ensemble_seed=0,split_seed=20261112,train_rows=len(a),validation_rows=len(b),
                train_ids_sha256=data.audit.row_hash(dev[a]),validation_ids_sha256=data.audit.row_hash(dev[b]),
                train_group_ids_sha256=data.audit.string_hash(inputs['groups'][a]),validation_group_ids_sha256=data.audit.string_hash(inputs['groups'][b]),
                group_overlap_count=len(set(inputs['groups'][a])&set(inputs['groups'][b])),
                accuracy=score['accuracy'],log_loss=score['log_loss'],seconds=time.monotonic()-tick,metadata=metadata,
                resources_after_forward=before_release,resources_after_release=resource.snapshot('mps'))
            record['references']={k:data.metrics(v[b],y[b],[(np.array([],dtype=int),np.arange(len(b)))]) for k,v in references.items()}
            records.append(record);write(f'fold{fold}.json',record)
            np.savez(out/f'fold{fold}_rows.npz',train_ids=dev[a],validation_ids=dev[b])
            np.savez(out/'partial_oof.npz',prob=prob,dev_ids=dev,fold_id=fold_id)
            write('progress.json',dict(status='incomplete',completed_folds=len(records),folds=records,partial_results_not_a_winner=True))
            print(json.dumps(dict(phase=phase,fold=fold,accuracy=score['accuracy'],log_loss=score['log_loss'],seconds=record['seconds'],
                reference_accuracy={k:m['accuracy'] for k,m in record['references'].items()})),flush=True)
        if len(records)!=3 or np.any(fold_id<0):raise RuntimeError('All three complete folds required.')
        score=data.metrics(prob,y,splits);ref_scores={k:data.metrics(v,y,splits) for k,v in references.items()}
        assessment=assess(phase,score,ref_scores);core.check_deadline();assert_frozen(frozen)
        cache=out/f'{core.NAME}_seed20261112.npz'
        np.savez(cache,prob=prob,fold_accuracy=score['fold_accuracy'],dev_ids=dev,fold_id=fold_id,
                 passenger_ids=inputs['passengers'],group_ids=inputs['groups'])
        table=pd.DataFrame([dict(version=core.NAME,accuracy=score['accuracy'],log_loss=score['log_loss'],
            fold_std=float(np.std(score['fold_accuracy'])),seconds=time.monotonic()-started)])
        csv='screen_results.csv' if phase=='random' else 'group_metrics.csv'
        for folder in [out,report]:table.to_csv(folder/csv,index=False)
        write('assessment.json',assessment)
        result=dict(status='complete',candidate=core.NAME,phase=phase,completed_folds=3,metrics=score,references=ref_scores,
            paired={k:paired(prob,v,y,splits) for k,v in references.items()},seconds=time.monotonic()-started,
            continue_to_group=phase=='random' and assessment['passed'],continue_to_root_decision=phase=='group' and assessment['passed'],
            common_local_tier='A practical parity band' if phase=='random' and assessment['passed'] else 'not classified by this phase',
            automatic_upload=False,S_supported=False,no_old_holdout_targets_used=True,
            disclosure='Recipe was chosen after earlier development experiments; these folds are reused, not independent confirmation.',
            result_files_sha256={p.name:core.sha(p) for p in [cache,out/csv,out/'assessment.json',*[out/f'fold{k}.json' for k in range(3)],*[out/f'fold{k}_rows.npz' for k in range(3)]]})
        core.check_deadline();assert_frozen(frozen)
        write('summary.json',result);write('progress.json',result)
        core.check_deadline();assert_frozen(frozen)
        return result
    except BaseException as error:
        for folder in [out,report]:
            for name in final_names:(folder/name).unlink(missing_ok=True)
        failed=dict(status='failed_inconclusive',candidate=core.NAME,phase=phase,completed_folds=len(records),
            error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started,
            partial_results_not_a_winner=True,automatic_upload=False,S_supported=False)
        write('summary.json',failed);write('progress.json',failed);raise


def run(phase,session_file):
    directory=DIRECTORIES[phase];out=ROOT/'artifacts'/directory;report=ROOT/'reports'/directory
    if out.exists() or report.exists():raise RuntimeError('Destination exists; no overwrite or implicit resume.')
    resource.check_quota(session_file)
    frozen,lock=resource_gate()  # All resource gates precede development target parsing.
    if phase=='group':random_gate(frozen['evidence'])
    inputs=data.load_data()
    splits,references,evidence=(data.random_references if phase=='random' else data.group_references)(inputs)
    frozen['evidence'].update(inputs['evidence']);frozen['evidence'].update(evidence)
    public=[name for name in frozen['evidence'] if name.startswith('reports/')]
    data.common.require_committed(public);assert_frozen(frozen)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0') or not core.torch.backends.mps.is_available():
        raise RuntimeError('MPS without implicit CPU fallback required.')
    protocol=dict(candidate=core.NAME,phase=phase,seed=20261112,ensemble_seed_every_fold=0,folds=3,development_rows=6954,
        threshold=.5,budget_seconds=BUDGET,plan_sha256=PLAN_SHA,settings=core.SETTINGS,frozen=frozen,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        development_ids_sha256=data.audit.row_hash(inputs['dev']),development_labels_sha256=core.array_hash(inputs['y']),
        passenger_ids_sha256=data.audit.string_hash(inputs['passengers']),group_ids_sha256=data.audit.string_hash(inputs['groups']),
        fold_memberships=[dict(train_ids_sha256=data.audit.row_hash(inputs['dev'][a]),validation_ids_sha256=data.audit.row_hash(inputs['dev'][b])) for a,b in splits],
        no_old_holdout_targets_used=True,automatic_upload=False)
    out.mkdir(parents=True);report.mkdir(parents=True)
    for folder in [out,report]:resource.write(folder/'protocol.json',protocol);(folder/'environment.lock.txt').write_text(lock)
    for name in {**frozen['resource']['sources'],**frozen['sources']}:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    started=time.monotonic();previous=signal.getsignal(signal.SIGALRM);old=core.DEADLINE
    def timeout(*_):raise TimeoutError('Fixed1800s EXAONE validation phase budget exhausted.')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET);core.DEADLINE=started+BUDGET
    try:
        core.torch.set_num_threads(4);core.torch.mps.set_per_process_memory_fraction(.65)
        return evaluate(inputs,splits,references,out,report,frozen,phase)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);core.DEADLINE=old;gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['random','group'])
    parser.add_argument('--session-file',type=Path,required=True)
    args=parser.parse_args();run(args.phase,args.session_file)
