"""Conditional native8 iLTM development CV. Native libraries load only in isolated workers."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import weakref

import iltm_r128bn_full8_probe as resource

ROOT=Path(__file__).resolve().parent
PLAN='reports/iltm_r128bn_cv_design.json'
PLAN_SHA='21d7453dba8ebd69adf2230d9a74cf5a9ca6927e8d51f20533055de4e9f1d4fd'
NAME='iLTM_r128bn_native8'
BUDGET=14400
OUTPUT=ROOT/'artifacts/iltm_r128bn_cv'
REPORT=ROOT/'reports/iltm_r128bn_cv'
OWN=['iltm_r128bn_cv.py','test_iltm_r128bn_cv.py',PLAN,'reports/iltm_r128bn_cv_design_review.json',
     'exaone_validation_data.py','reports/exaone_validation_features.json','audit_top5_references.py',
     'three_family_mean_screen.py','prior_adapt_mean_screen.py']
read,write,sha=resource.read,resource.write,resource.sha


def deadline_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Entire three-fold14400s monotonic deadline exceeded.')


def verify_files(files):
    for name,digest in files.items():
        path=ROOT/name
        if not path.resolve().is_relative_to(ROOT.resolve()) or sha(path)!=digest:
            raise RuntimeError('Frozen CV identity changed: '+name)


def validate_resource(result,worker,monitor):
    if (result.get('status')!='complete' or any(result.get(k) is not True for k in
            ['resource_execution_complete','cost_gate_passed','resource_qualification'])
            or result.get('probe_members')!=8 or result.get('candidate_members')!=8
            or result.get('budget_seconds')!=7200 or result.get('rss_limit_bytes')!=resource.LIMIT
            or len(result.get('cases',[]))!=1):
        raise RuntimeError('Committed complete8 resource execution AND cost qualification required.')
    row=result['cases'][0]
    if any(row.get(k)!=v for k,v in worker.items()):raise RuntimeError('Resource summary/worker mismatch.')
    if (monitor.get('limit_bytes')!=resource.LIMIT or monitor.get('samples',0)<1
            or row.get('sampled_tree_rss_highwater_bytes')!=monitor.get('sampled_tree_rss_highwater_bytes')):
        raise RuntimeError('External resource RSS evidence differs.')
    actual=resource.projection([row])
    if not actual['passed'] or result.get('projection')!=actual:raise RuntimeError('Measured full8 cost gate differs/fails.')


def resource_gate():
    # This gate must complete before the first call to the development-target reader.
    resource.require_committed(OWN)
    if sha(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Prospective CV recipe changed.')
    plan=read(ROOT/PLAN);verify_files(plan['prerequisite_evidence_sha256'])
    resource.require_committed(plan['prerequisite_evidence_sha256'])
    evidence=dict(plan['prerequisite_evidence_sha256'])
    files=['result.json','freeze.json','8693_4277_worker_result.json','8693_4277_external_monitor.json','root_audit.json']
    resource.require_committed([str((resource.REPORT/name).relative_to(ROOT)) for name in files])
    for name in files:
        public=resource.REPORT/name;evidence[str(public.relative_to(ROOT))]=sha(public)
        if name=='root_audit.json':continue
        local=resource.OUTPUT/(name if name in ['result.json','freeze.json'] else '8693_4277/'+name.removeprefix('8693_4277_'))
        if sha(public)!=sha(local):raise RuntimeError('Resource report/artifact mismatch: '+name)
        evidence[str(local.relative_to(ROOT))]=sha(local)
    result=read(resource.REPORT/'result.json');worker=read(resource.REPORT/'8693_4277_worker_result.json')
    validate_resource(result,worker,read(resource.REPORT/'8693_4277_external_monitor.json'))
    resource.qualify(worker,resource.OUTPUT/'8693_4277')
    current=resource.contract();saved=read(resource.REPORT/'freeze.json')
    for key in set(current)|set(saved):
        if key!='source_commit' and current.get(key)!=saved.get(key):raise RuntimeError('Qualified resource freeze differs: '+key)
    resource.assert_frozen(saved)
    for name in ['model.joblib','probabilities.npz']:
        path=resource.OUTPUT/'8693_4277'/name;evidence[str(path.relative_to(ROOT))]=sha(path)
    return dict(resource=current,sources={name:sha(ROOT/name) for name in OWN},evidence=evidence,
                plan_sha256=PLAN_SHA)


def assert_frozen(frozen):
    if frozen.get('plan_sha256')!=PLAN_SHA:raise RuntimeError('Wrong CV freeze.')
    resource.assert_frozen(frozen['resource'])
    verify_files(frozen['sources']);verify_files(frozen['evidence'])


def validate_splits(inputs,splits,phase):
    import numpy as np
    dev,y=inputs['dev'],inputs['y'];seen=np.zeros(len(dev),dtype=int)
    if len(dev)!=6954 or len(y)!=6954 or len(splits)!=3 or len(np.unique(dev))!=6954:
        raise RuntimeError('Original6954 development and exactly3folds required.')
    for a,b in splits:
        if (len(a)<4096 or len(a)+len(b)!=6954 or len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b)
                or np.intersect1d(a,b).size or set(np.r_[a,b])!=set(range(6954))):
            raise RuntimeError('Outer row overlap/coverage or unsupported shape.')
        if phase=='group' and set(inputs['groups'][a])&set(inputs['groups'][b]):raise RuntimeError('Outer group overlap.')
        seen[b]+=1
    if not np.all(seen==1):raise RuntimeError('Outer validation rows must be covered exactly once.')


def save_packet(out,inputs,a,b,phase,fold):
    import numpy as np
    dev=inputs['dev'];x=inputs['features'].iloc[dev[a]].copy();q=inputs['features'].iloc[dev[b]].copy()
    x=x.reset_index(drop=True);q=q.reset_index(drop=True)
    if list(x.columns)!=resource.COLUMNS or list(q.columns)!=resource.COLUMNS:raise RuntimeError('Exact compact13 required.')
    arrays=dict(train_ids=dev[a],query_ids=dev[b],train_labels=inputs['y'][a].astype(np.int64))
    for prefix,frame in [('train',x),('query',q)]:
        for i,name in enumerate(resource.COLUMNS):
            arrays[prefix+'__'+name]=(frame[name].astype(str).to_numpy(dtype=str) if i in resource.CATEGORICAL
                                      else frame[name].to_numpy())
    np.savez(out/'input.npz',**arrays)
    meta=dict(phase=phase,fold=fold,seed=resource.SEED,training_rows=len(a),query_rows=len(b),
        training_ids_sha256=resource.array_hash(dev[a]),query_ids_sha256=resource.array_hash(dev[b]),
        training_labels_sha256=resource.array_hash(arrays['train_labels']),
        training_input_sha256=resource.frame_hash(x),query_input_sha256=resource.frame_hash(q),
        packet_sha256=sha(out/'input.npz'),query_labels_present=False)
    write(out/'input.json',meta);return meta


def load_packet(out):
    import numpy as np
    import pandas as pd
    meta=read(out/'input.json')
    if (meta['packet_sha256']!=sha(out/'input.npz') or meta.get('query_labels_present') is not False
            or meta.get('seed')!=resource.SEED or meta.get('phase') not in ['random','group'] or meta.get('fold') not in [0,1,2]):
        raise RuntimeError('Isolated fold packet identity differs.')
    keys={'train_ids','query_ids','train_labels'}|{p+'__'+c for p in ['train','query'] for c in resource.COLUMNS}
    with np.load(out/'input.npz',allow_pickle=False) as source:
        if set(source.files)!=keys:raise RuntimeError('Packet may contain only train labels and unlabeled query attributes.')
        arrays={k:source[k].copy() for k in source.files}
    train,query,y=arrays['train_ids'],arrays['query_ids'],arrays['train_labels']
    if (len(train)!=meta['training_rows'] or len(query)!=meta['query_rows'] or len(train)<4096
            or len(train)+len(query)!=6954 or train.ndim!=1 or query.ndim!=1 or y.shape!=train.shape
            or len(np.unique(train))!=len(train) or len(np.unique(query))!=len(query)
            or np.intersect1d(train,query).size or set(np.unique(y))!={0,1}):raise RuntimeError('Packet row/label boundary failed.')
    frames=[]
    for prefix in ['train','query']:
        values={name:(arrays[prefix+'__'+name].astype(object) if i in resource.CATEGORICAL else arrays[prefix+'__'+name])
                for i,name in enumerate(resource.COLUMNS)}
        frames.append(pd.DataFrame(values,columns=resource.COLUMNS))
    x,q=frames
    checks=dict(training_ids_sha256=resource.array_hash(train),query_ids_sha256=resource.array_hash(query),
        training_labels_sha256=resource.array_hash(y),training_input_sha256=resource.frame_hash(x),query_input_sha256=resource.frame_hash(q))
    if any(meta.get(k)!=v for k,v in checks.items()):raise RuntimeError('Packet data/ID hashes differ.')
    return x,y,q,meta


def validate_native_fold(record):
    resource.validate_native_completion(record)
    if record.get('classes')!=[0,1]:raise RuntimeError('Native class order differs.')
    for member in record['native_observation']['finetuning']:
        if (member['train_rows']+member['validation_rows']!=record['training_rows']
                or not 0<member['training_retrieval_rows']<=member['train_rows']):
            raise RuntimeError('Native internal split/retrieval exceeds outer training context.')


def worker(out,deadline):
    record=dict(status='failed_inconclusive',candidate=NAME,no_query_labels=True,fit_eval_set=None,fit_max_time=None)
    started=time.monotonic();model=native_class=None;observing=False;old_profile=None
    def stage(name):
        record['current_stage']=name;write(out/'partial.json',record);deadline_check(deadline)
    try:
        stage('identity_before_native_import');frozen=read(out.parent/'protocol.json')['frozen'];assert_frozen(frozen)
        resource.require_committed(frozen['sources'])
        if os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')!=str(Path(importlib.util.find_spec('torch').origin).parent/'lib'):
            raise RuntimeError('Isolated child loader environment required.')
        x,y,q,meta=load_packet(out);record.update(meta)
        sys.path.insert(0,str(resource.VENDOR))
        import numpy as np
        import torch
        from threadpoolctl import threadpool_info,threadpool_limits
        from iltm import iLTMClassifier
        import iltm.inference_interface as interface
        if Path(interface.__file__).resolve()!=(resource.VENDOR/'iltm/inference_interface.py').resolve():raise RuntimeError('Wrong native source.')
        native_class=iLTMClassifier;torch.set_num_threads(4);torch.set_num_interop_threads(4)
        expected=resource.expected_parameters(iLTMClassifier)
        with threadpool_limits(limits=4):
            stage('construct')
            model=iLTMClassifier(device='cpu',n_ensemble=8,seed=resource.SEED,checkpoint=str(resource.WEIGHT),
                cat_features=resource.CATEGORICAL,**resource.OVERRIDES,**resource.FAIL_CLOSED)
            resource.check_model(model,expected);record['configuration']=expected
            record['constructed_cache_entries']=len(native_class._model_cache)
            observer=resource.NativeObservation(resource.VENDOR/'iltm',lambda:write(out/'partial.json',record))
            record['native_observation']=observer.records;stage('native_fit');tick=time.monotonic()
            old_profile=sys.getprofile();sys.setprofile(observer);observing=True
            try:model.fit(x,y,eval_set=None,fit_max_time=None,return_partial_on_timeout=False)
            finally:sys.setprofile(old_profile);observing=False
            record['fit_seconds']=time.monotonic()-tick
            if observer.error:raise RuntimeError('Native observation failed: '+observer.error)
            resource.check_model(model,expected,True)
            record.update(predictor_count=len(model.predictors_),preprocessor_count=len(model.preprocessors_),
                classes=model.classes_.tolist(),
                inference_retrieval_rows=[len(v['y_ctxt_superset']) for v in model.predictors_],
                postfit_cache_entries=len(native_class._model_cache),postfit_engine_released=model._model is None)
            validate_native_fold(record)
            record['hit_epoch_cap']=[v['epochs_entered']==2048 for v in observer.records['finetuning']]
            stage('predict_outer_validation');tick=time.monotonic();p=resource.probabilities(model,q)
            record['prediction_seconds']=time.monotonic()-tick;record['prediction_rows']=len(q)
            resource.check_model(model,expected,True)
            record['postpredict_cache_entries']=len(native_class._model_cache)
            if record['postpredict_cache_entries']!=1:raise RuntimeError('Native cache changed during prediction.')
            np.savez(out/'predictions.npz',probabilities=p,query_ids=np.asarray(read_ids(out)))
            record['predictions_sha256']=sha(out/'predictions.npz')
            record['openmp_runtimes']=[{k:v for k,v in item.items() if k in ['filepath','num_threads','prefix','internal_api']}
                for item in threadpool_info() if item.get('user_api')=='openmp']
            stage('final_identity');assert_frozen(frozen);record['status']='worker_complete'
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
    finally:
        if observing:sys.setprofile(old_profile)
        try:
            ref=weakref.ref(model) if model is not None else None
            if model is not None:model._release_training_model()
            model=None
            if native_class is not None:native_class._model_cache.clear()
            gc.collect()
            if ref is not None and ref() is not None:raise RuntimeError('Fitted native estimator retained after release.')
            deadline_check(deadline);record['release_complete']=True;record['released_cache_entries']=0
        except BaseException as error:
            record.update(status='failed_inconclusive',release_complete=False,cleanup_error=str(error))
            record.setdefault('error_type',type(error).__name__);record.setdefault('error',str(error))
        record['monotonic_seconds']=time.monotonic()-started;write(out/'worker_result.json',record)
    return record


def read_ids(out):
    import numpy as np
    with np.load(out/'input.npz',allow_pickle=False) as packet:return packet['query_ids'].copy()


def validate_worker(record,meta,out):
    import numpy as np
    if (record.get('status')!='worker_complete' or record.get('release_complete') is not True
            or record.get('released_cache_entries')!=0 or record.get('no_query_labels') is not True
            or record.get('prediction_rows')!=meta['query_rows'] or record.get('postpredict_cache_entries')!=1):
        raise RuntimeError('Complete native fold/release required: '+str(record.get('error_type'))+': '+str(record.get('error')))
    if any(record.get(k)!=v for k,v in meta.items()):raise RuntimeError('Fitted fold metadata differs from input packet.')
    validate_native_fold(record)
    if sha(out/'input.npz')!=meta['packet_sha256']:raise RuntimeError('Fold input changed after fit.')
    if sha(out/'predictions.npz')!=record['predictions_sha256']:raise RuntimeError('Fold predictions changed.')
    with np.load(out/'predictions.npz',allow_pickle=False) as saved:
        p=saved['probabilities'].copy();ids=saved['query_ids'].copy()
    if (p.shape!=(meta['query_rows'],2) or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or not np.allclose(p.sum(axis=1),1,atol=2e-6,rtol=0)
            or resource.array_hash(ids)!=meta['query_ids_sha256']):raise RuntimeError('Complete finite validation coverage required.')
    return p[:,1]


def assess(phase,metrics,references):
    if phase=='random':checks={'current_fixed_A_band':metrics['accuracy']+1e-12>=.811110152430256}
    elif phase=='group':
        anchors={'v71_Mitra2_zeroshot':.8169398907103825,'v44_seed_ensemble':.814495254529767}
        if any(abs(references[k]['accuracy']-v)>1e-12 for k,v in anchors.items()):raise RuntimeError('Matched group reference changed.')
        checks={k:metrics['accuracy']+1e-12>=v-.01 for k,v in anchors.items()}
    else:raise ValueError('Unknown phase.')
    return dict(passed=all(checks.values()),checks=checks,S_supported=False,automatic_export=False,automatic_upload=False,
                meaning='Adaptive development practical-band eligibility only; no stable superiority or online qualification.')


def paired(p,reference,y,splits):
    import numpy as np
    def one(ids):
        a=(p[ids]>=.5)==y[ids];b=(reference[ids]>=.5)==y[ids]
        corrected=int(np.sum(a&~b));regressed=int(np.sum(~a&b))
        defined=np.std(a)>0 and np.std(b)>0
        return dict(both_correct=int(np.sum(a&b)),both_wrong=int(np.sum(~a&~b)),corrected=corrected,
            regressed=regressed,net_correct=corrected-regressed,decision_disagreements=int(np.sum(a!=b)),
            error_correlation=float(np.corrcoef(~a,~b)[0,1]) if defined else None,
            error_correlation_undefined_reason=None if defined else 'At least one correctness vector is constant.',
            mean_absolute_probability_difference=float(np.mean(np.abs(p[ids]-reference[ids]))))
    return dict(overall=one(np.arange(len(y))),folds=[one(b) for _,b in splits])


def random_gate(inputs,frozen):
    import numpy as np
    import exaone_validation_data as data
    out=OUTPUT/'random';public=REPORT/'random'
    for name in ['protocol.json','summary.json','assessment.json']:
        resource.require_committed([str((public/name).relative_to(ROOT))])
        if sha(public/name)!=sha(out/name):raise RuntimeError('Random prerequisite report/artifact mismatch.')
        for path in [public/name,out/name]:frozen['evidence'][str(path.relative_to(ROOT))]=sha(path)
    protocol=read(out/'protocol.json');summary=read(out/'summary.json');assessment=read(out/'assessment.json')
    if (summary.get('status')!='complete' or summary.get('completed_folds')!=3 or summary.get('candidate')!=NAME
            or protocol.get('phase')!='random' or summary.get('continue_to_group') is not True or not assessment.get('passed')):
        raise RuntimeError('Complete passed random screen required before group fits.')
    assert_frozen(protocol['frozen'])
    for name,digest in summary['result_files_sha256'].items():
        if sha(out/name)!=digest:raise RuntimeError('Random evidence hash changed.')
        frozen['evidence'][str((out/name).relative_to(ROOT))]=digest
    splits,refs,evidence=data.random_references(inputs);frozen['evidence'].update(evidence)
    with np.load(out/'oof.npz',allow_pickle=False) as cache:
        if not np.array_equal(cache['dev_ids'],inputs['dev']):raise RuntimeError('Random development alignment differs.')
        p=cache['probability'].copy();ids=cache['fold_id'].copy()
    expected=np.empty(len(p),dtype=int)
    for fold,(_,b) in enumerate(splits):expected[b]=fold
    if not np.array_equal(ids,expected):raise RuntimeError('Random saved fold IDs differ.')
    actual=data.metrics(p,inputs['y'],splits)
    for fold,(a,b) in enumerate(splits):
        name=f'fold{fold}.json';resource.require_committed([str((public/name).relative_to(ROOT))])
        if sha(public/name)!=sha(out/name):raise RuntimeError('Random fold report/artifact differs.')
        frozen['evidence'][str((public/name).relative_to(ROOT))]=sha(public/name)
        record=read(out/name);folder=out/f'fold{fold}';meta=read(folder/'input.json')
        for filename,digest in record['files_sha256'].items():
            if sha(folder/filename)!=digest:raise RuntimeError('Random native fold evidence changed.')
            frozen['evidence'][str((folder/filename).relative_to(ROOT))]=digest
        if (meta.get('training_ids_sha256')!=resource.array_hash(inputs['dev'][a])
                or meta.get('query_ids_sha256')!=resource.array_hash(inputs['dev'][b])
                or meta.get('training_labels_sha256')!=resource.array_hash(inputs['y'][a].astype(np.int64))):
            raise RuntimeError('Random fitted training/query identity differs.')
        measured=validate_worker(read(folder/'worker_result.json'),meta,folder)
        if not np.array_equal(measured,p[b]):raise RuntimeError('Random native prediction/OOF differs.')
    if actual!=summary['metrics'] or assess('random',actual,{})!=assessment:raise RuntimeError('Random eligibility recomputation differs.')
    if not assessment['passed']:raise RuntimeError('Random A-band gate failed.')


def run(phase):
    if phase not in ['random','group']:raise ValueError('Unknown phase.')
    out,public=OUTPUT/phase,REPORT/phase
    if out.exists() or public.exists():raise FileExistsError('No overwrite/resume/retry.')
    started=time.monotonic();deadline=started+BUDGET;process=None;records=[]
    frozen=resource_gate();deadline_check(deadline)
    out.mkdir(parents=True);public.mkdir(parents=True)
    def publish(name,value):
        write(out/name,value);write(public/name,value)
    try:
        import numpy as np
        import exaone_validation_data as data
        inputs=data.load_data();frozen['evidence'].update(inputs['evidence'])
        if phase=='group':random_gate(inputs,frozen)
        splits,refs,evidence=(data.random_references if phase=='random' else data.group_references)(inputs)
        frozen['evidence'].update(evidence);validate_splits(inputs,splits,phase);assert_frozen(frozen)
        protocol=dict(phase=phase,candidate=NAME,plan_sha256=PLAN_SHA,budget_seconds=BUDGET,model_seed=resource.SEED,
            split_seed=20261112,development_ids_sha256=data.audit.row_hash(inputs['dev']),frozen=frozen)
        publish('protocol.json',protocol)
        probabilities=np.full(len(inputs['dev']),np.nan);fold_ids=np.full(len(inputs['dev']),-1,dtype=int)
        for fold,(a,b) in enumerate(splits):
            deadline_check(deadline);assert_frozen(frozen)
            folder=out/f'fold{fold}';folder.mkdir();meta=save_packet(folder,inputs,a,b,phase,fold)
            publish('progress.json',dict(status='incomplete',completed_folds=len(records),current_fold=fold))
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(folder),'--deadline',str(deadline)]
            with (folder/'native.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=resource.child_environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                peak=resource.supervise(process,deadline,folder)
            record=read(folder/'worker_result.json');p=validate_worker(record,meta,folder)
            record['sampled_tree_rss_highwater_bytes']=peak
            probabilities[b]=p;fold_ids[b]=fold;record['metrics']=data.metrics(p,inputs['y'][b],[(np.array([],dtype=int),np.arange(len(b)))])
            record['files_sha256']={name:sha(folder/name) for name in ['input.json','input.npz','predictions.npz','worker_result.json','external_monitor.json']}
            records.append(record);publish(f'fold{fold}.json',record)
            np.savez(out/'partial_oof.npz',probability=probabilities,dev_ids=inputs['dev'],fold_id=fold_ids)
            print(json.dumps(dict(phase=phase,fold=fold,accuracy=record['metrics']['accuracy'],log_loss=record['metrics']['log_loss'],seconds=record['monotonic_seconds'])),flush=True)
        if len(records)!=3 or not np.isfinite(probabilities).all() or set(fold_ids)!={0,1,2}:raise RuntimeError('All three folds required before pooled results.')
        metrics=data.metrics(probabilities,inputs['y'],splits);reference_metrics={k:data.metrics(v,inputs['y'],splits) for k,v in refs.items()}
        assessment=assess(phase,metrics,reference_metrics)
        np.savez(out/'oof.npz',probability=probabilities,dev_ids=inputs['dev'],fold_id=fold_ids,
            passenger_ids=inputs['passengers'],group_ids=inputs['groups'])
        publish('assessment.json',assessment);assert_frozen(frozen);deadline_check(deadline)
        results={'oof.npz':sha(out/'oof.npz'),'assessment.json':sha(out/'assessment.json'),'protocol.json':sha(out/'protocol.json')}
        results.update({f'fold{k}.json':sha(out/f'fold{k}.json') for k in range(3)})
        summary=dict(status='complete',candidate=NAME,phase=phase,completed_folds=3,metrics=metrics,references=reference_metrics,
            paired={k:paired(probabilities,v,inputs['y'],splits) for k,v in refs.items()},assessment=assessment,
            continue_to_group=phase=='random' and assessment['passed'],continue_to_root_decision=phase=='group' and assessment['passed'],
            S_supported=False,automatic_export=False,automatic_upload=False,result_files_sha256=results,
            monotonic_seconds=time.monotonic()-started,validation_kind=phase,
            disclosure='Previously explored development folds; adaptive recipe, no independent validation or stable-superiority claim.')
        publish('summary.json',summary);publish('progress.json',dict(status='complete',completed_folds=3));return summary
    except BaseException as error:
        failure=dict(status='failed_inconclusive',phase=phase,candidate=NAME,error_type=type(error).__name__,error=str(error),
            completed_folds=len(records),monotonic_seconds=time.monotonic()-started,automatic_export=False,automatic_upload=False)
        if process is not None and process.poll() is None:
            try:resource.terminate(process,resource.process_tree(process.pid)['pids'])
            except BaseException as cleanup:failure['termination_error']=str(cleanup)
        if 'folder' in locals():
            for name in ['worker_result.json','partial.json','external_monitor.json']:
                if (folder/name).exists():
                    failure[name.removesuffix('.json')]=read(folder/name)
                    shutil.copy2(folder/name,public/f'{folder.name}_{name}')
        for directory in [out,public]:
            for name in ['oof.npz','assessment.json','summary.json']:
                path=directory/name
                if path.exists():path.rename(path.with_name(path.stem+'.partial'+path.suffix))
        publish('failure.json',failure);publish('progress.json',dict(status='failed_inconclusive',completed_folds=len(records)))
        return failure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',nargs='?',choices=['random','group'])
    parser.add_argument('--worker',type=Path);parser.add_argument('--deadline',type=float);args=parser.parse_args()
    if args.worker:
        if args.deadline is None:parser.error('Worker requires parent deadline.')
        worker(args.worker,args.deadline)
    elif args.phase:print(json.dumps(run(args.phase),indent=2))
    else:parser.error('Choose random or group.')
