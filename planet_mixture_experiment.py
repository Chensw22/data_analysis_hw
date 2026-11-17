"""Nested HomePlanet mixtures of unchanged rawMitra and Kumo; no upload.

The reused development set remains adaptive evidence. Each native worker gets
only its own training labels, unlabeled full-roster attributes and row IDs.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import audit_top5_references as audit
import three_family_mean_screen as history

ROOT=Path(__file__).resolve().parent
DIRECTORY='planet_mixture_20261117'
PLAN=f'reports/{DIRECTORY}/protocol.json'
OUTPUT=ROOT/'artifacts'/DIRECTORY
REPORT=ROOT/'reports'/DIRECTORY
OWN=['planet_mixture_experiment.py','test_planet_mixture_experiment.py','audit_top5_references.py',
     'three_family_mean_screen.py','iterate.py','experiment.py',PLAN,'reports/official_data_verification.json']
MODELS={'raw':('run_038','v71_Mitra2_zeroshot','.venv-mitra'),
        'kumo':('run_048','v83_Kumo_medium8','.venv-kumo')}
RECIPE={'development_rows': 6954, 'outer_folds': 3, 'outer_seed': 20261112, 'inner_folds': 3, 'inner_seed_base': 20261117, 'fit_seed_base': 20261117, 'fit_seed_outer_multiplier': 100, 'models': ['rawMitra', 'Kumo'], 'raw_version': 'v71_Mitra2_zeroshot', 'kumo_version': 'v83_Kumo_medium8', 'outer_prediction_source': 'Authenticated original run038/run048 exact outerOOF; no retraining needed for unchanged outer fits.', 'inner_prediction_source': 'Fresh inner fits only on each outer-training partition, 9splits times2models=18fits, fresh sequential process per fit.', 'gate_feature': 'HomePlanet', 'categories': ['Earth', 'Europa', 'Mars', '__MISSING__'], 'alpha_bounds': [0.0, 1.0], 'alpha_initial': 0.5, 'mixture': '(1-alpha)*raw + alpha*Kumo', 'objective': 'mean binary logloss + 0.01*mean((alpha-0.5)^2)', 'regularization': 0.01, 'optimizer': 'L-BFGS-B', 'optimizer_maxiter': 500, 'optimizer_ftol': 1e-12, 'optimization_probability_clip': 1e-07, 'final_prediction_uses_original_base_probabilities': True, 'unknown_category_fallback': 'matched constant alpha', 'threshold': 0.5, 'budget_seconds': 7200, 'threads': 4, 'device': 'mps', 'backend_fallback': False}
CONTINUATION={'accuracy_at_least_each_of': ['rawMitra', 'Kumo', 'constant_control'], 'log_loss_at_most': 'constant_control', 'fold_accuracy_at_least_constant_minimum': 2, 'at_least_one_overall_strict_improvement_vs_constant': True, 'if_pass': 'Only separate review for frozen group confirmation; no automatic export/upload.', 'if_fail': 'Archive; do not upload constant as fallback or change planet/features/shrinkage/folds/weights.', 'stable_superiority_claim': False}
CATEGORIES=RECIPE['categories']
BUDGET=7200


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def ids_hash(ids):return audit.row_hash(ids)
def frame_hash(frame):
    digest=hashlib.sha256(json.dumps([(str(c),str(t)) for c,t in zip(frame.columns,frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes());return digest.hexdigest()
def verify_hashes(mapping):
    for name,digest in mapping.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(ROOT/path)!=digest:
            raise RuntimeError('Frozen source/input changed: '+name)
def deadline_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Shared nested experiment7200s deadline exceeded.')
def environment_lock():
    return ''.join(f'{name}=={version}\n' for name,version in sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))


def normalize_planets(values):
    return np.asarray(['__MISSING__' if pd.isna(v) or not str(v).strip() else str(v).strip() for v in values],dtype=str)
def valid_vector(p,n=None):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=1 or (n is not None and len(p)!=n) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Complete finite aligned probability vector required.')
    return p

def gate_objective(alpha,raw,kumo,y,index,constant_fallback=.5):
    """Convex objective/analytic derivative; components clipped only for fitting."""
    raw=np.clip(raw,1e-7,1-1e-7);kumo=np.clip(kumo,1e-7,1-1e-7)
    known=index>=0;weights=np.full(len(y),constant_fallback,dtype=float);weights[known]=alpha[index[known]]
    p=(1-weights)*raw+weights*kumo;difference=kumo-raw
    loss=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))+.01*np.mean((alpha-.5)**2)
    derivative=((p-y)/(p*(1-p)))*difference/len(y)
    gradient=np.bincount(index[known],weights=derivative[known],minlength=len(alpha))+.02*(alpha-.5)/len(alpha)
    return float(loss),gradient

def fit_gate(raw,kumo,y,planets,kind,constant_alpha=None):
    y=np.asarray(y)
    if y.ndim!=1 or not np.isin(y,[0,1]).all():raise ValueError('Binary innerOOF labels required.')
    y=y.astype(np.int64);raw=valid_vector(raw,len(y));kumo=valid_vector(kumo,len(y))
    planets=normalize_planets(planets)
    if y.ndim!=1 or not len(y) or not np.isin(y,[0,1]).all() or len(planets)!=len(y):raise ValueError('Aligned innerOOF labels/planet required.')
    if kind=='constant':index=np.zeros(len(y),dtype=int);count=1;fallback=.5
    elif kind=='planet':
        if constant_alpha is None or not np.isfinite(constant_alpha) or not 0<=constant_alpha<=1:raise ValueError('Matched learned constant required.')
        index=np.array([CATEGORIES.index(v) if v in CATEGORIES else -1 for v in planets]);count=4;fallback=float(constant_alpha)
    else:raise ValueError('Only constant or planet gate is frozen.')
    objective=lambda alpha:gate_objective(alpha,raw,kumo,y,index,fallback)
    result=minimize(objective,np.full(count,.5),jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*count,
                    options={'maxiter':500,'ftol':1e-12})
    if (not result.success or not np.isfinite(result.fun) or not np.isfinite(result.jac).all()
            or not np.isfinite(result.x).all() or np.any((result.x<0)|(result.x>1))):
        raise RuntimeError('Fixed gate optimizer did not converge: '+str(result.message))
    return dict(kind=kind,alpha=result.x.tolist(),constant_fallback=float(result.x[0]) if kind=='constant' else fallback,
        success=True,objective=float(result.fun),gradient=np.asarray(result.jac).tolist(),iterations=int(result.nit),
        training_rows=len(y),training_labels_sha256=ids_hash(y),raw_inner_oof_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        kumo_inner_oof_sha256=hashlib.sha256(kumo.tobytes()).hexdigest(),unknown_training_planets=int((index<0).sum()),
        input_scope='complete innerOOF on outer training rows only')

def apply_gate(gate,raw,kumo,planets):
    raw=valid_vector(raw);kumo=valid_vector(kumo,len(raw));planets=normalize_planets(planets)
    if len(planets)!=len(raw):raise ValueError('Planet alignment differs.')
    if gate['kind']=='constant':weights=np.full(len(raw),gate['alpha'][0]);unknown=0
    else:
        index=np.array([CATEGORIES.index(v) if v in CATEGORIES else -1 for v in planets]);known=index>=0
        weights=np.full(len(raw),gate['constant_fallback']);weights[known]=np.asarray(gate['alpha'])[index[known]];unknown=int((~known).sum())
    return (1-weights)*raw+weights*kumo,unknown


def validate_rows(train_ids,query_ids,outer_train_ids,outer_validation_ids,y):
    arrays=[np.asarray(v) for v in [train_ids,query_ids,outer_train_ids,outer_validation_ids]]
    if any(a.ndim!=1 or a.dtype.kind not in 'iu' or not len(a) or len(np.unique(a))!=len(a) for a in arrays):
        raise ValueError('Unique nonempty integer row sets required.')
    a,b,outer,held=arrays
    if (np.intersect1d(a,b).size or np.intersect1d(outer,held).size
            or not np.array_equal(np.sort(np.r_[a,b]),np.sort(outer))
            or np.intersect1d(np.r_[a,b],held).size or np.any(np.r_[outer,held]<0)):
        raise ValueError('Inner folds must partition outertrain and exclude outervalidation.')
    labels=np.asarray(y)
    if labels.shape!=a.shape or not np.isin(labels,[0,1]).all():raise ValueError('Worker gets exactly its own training labels.')
    return a.astype(np.int64),b.astype(np.int64)


def protocol_contract():
    history.require_committed(OWN)
    plan=read(ROOT/PLAN)
    if (plan.get('status')!='prospective_nested_experiment_before_new_inner_predictions'
            or plan.get('recipe')!=RECIPE or plan.get('continuation')!=CONTINUATION):
        raise RuntimeError('Frozen nested recipe/continuation differs.')
    sources={name:sha(ROOT/name) for name in OWN};evidence={};models={}
    for name in plan['diagnostic_evidence']:
        history.require_committed([name]);evidence[name]=sha(ROOT/name)
    for kind,(run,version,venv) in MODELS.items():
        protocol=read(ROOT/'reports'/run/'protocol.json')
        for name,digest in protocol['sources'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Current base source differs from completed outer fit: '+name)
            sources[name]=digest
        if kind=='raw':
            env_sha=protocol['mitra_environment_sha256'];lock=ROOT/'reports'/run/'requirements-mitra.lock.txt'
            weights={'artifacts/pretrained/mitra-v2/model.safetensors':protocol['mitra_checkpoint_sha256']}
            sources['artifacts/pretrained/mitra-v2/config.json']=sha(ROOT/'artifacts/pretrained/mitra-v2/config.json')
            resources=read(ROOT/'reports/mitra_protocol.json')['runtime_decision']['evidence_paths']
            # Preserve the original failed diagnostic; the two approved full-context results passed.
            for name in resources[:2]:
                value=read(ROOT/name)
                if value.get('status')!='complete' or value.get('full_context') is not True:raise RuntimeError('Original raw full-context resource proof required.')
            for name in resources:evidence[name]=sha(ROOT/name)
        else:
            env_sha=protocol['kumo_policy']['source_environment_sha256'];lock=ROOT/'reports/kumo_resources/full8_mps/environment.lock.txt'
            freeze=read(ROOT/'reports/kumo_resources/full8_mps/freeze.json')
            weights={'artifacts/pretrained/kumo/medium-classifier.pt':freeze['checkpoint_sha256']}
        if sha(lock)!=env_sha:raise RuntimeError('Historical isolated environment lock differs.')
        evidence[str(lock.relative_to(ROOT))]=env_sha
        command=[str(ROOT/venv/'bin/python'),'-c',
            'import importlib.metadata; print("".join(f"{n}=={v}\\n" for n,v in sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions())),end="")']
        actual=subprocess.check_output(command,cwd=ROOT,timeout=30)
        if hashlib.sha256(actual).hexdigest()!=env_sha:raise RuntimeError('Current isolated base environment differs: '+kind)
        verify_hashes(weights);evidence.update(weights)
        models[kind]=dict(run=run,version=version,environment=venv,environment_sha256=env_sha,weights=weights)
    history.require_committed([name for name in sources if not name.startswith('artifacts/')])
    official=read(ROOT/'reports/official_data_verification.json')
    for filename in ['train.csv','test.csv']:
        evidence['data/raw/'+filename]=official['files'][filename]['official_sha256']
    verify_hashes(sources);verify_hashes(evidence)
    return dict(sources=sources,evidence=evidence,models=models,protocol_sha256=sha(ROOT/PLAN),
                parent_environment_sha256=hashlib.sha256(environment_lock().encode()).hexdigest(),
                source_commit=history.require_committed(OWN))


def assert_frozen(frozen):
    verify_hashes(frozen['sources']);verify_hashes(frozen['evidence'])
    if sha(ROOT/PLAN)!=frozen['protocol_sha256']:raise RuntimeError('Nested protocol changed.')


def load_inputs(frozen):
    import iterate
    assert_frozen(frozen)
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    # This separate attribute-only read creates the worker file; no target column is parsed.
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c!='Transported')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Transported')
    roster=pd.concat([train,test],ignore_index=True)
    splits=list(StratifiedKFold(3,shuffle=True,random_state=20261112).split(dev,y))
    with np.load(ROOT/'artifacts/run_038/v6_no_id_features_seed20261112.npz',allow_pickle=False) as cache:baseline=cache['prob'].copy()
    probabilities={};evidence={}
    for kind,(run,version,_) in MODELS.items():
        p,_,files=history.read_component(run,version,'mitra' if kind=='raw' else 'kumo',dev,y,splits,baseline)
        probabilities[kind]=p;evidence.update(files)
    evidence['artifacts/run_002/split_indices.npz']=sha(ROOT/'artifacts/run_002/split_indices.npz')
    frozen['evidence'].update(evidence);assert_frozen(frozen)
    history.require_committed([name for name in evidence if name.startswith('reports/')])
    return dict(dev=dev,y=y,roster=roster,features_sha256=frame_hash(matrices[False][0]),
                planets=normalize_planets(train.HomePlanet.iloc[dev]),splits=splits,probabilities=probabilities)


def worker(kind,out,deadline):
    out=Path(out);model=wrapper=None;started=time.monotonic()
    result=dict(status='failed_inconclusive',kind=kind,release_complete=False)
    try:
        frozen=read(out.parent.parent/'freeze.json');job=read(out/'job.json')
        history.require_committed([name for name in frozen['sources'] if not name.startswith('artifacts/')])
        # Parent authenticates raw files/caches. Worker never opens them, even to hash labels.
        verify_hashes(frozen['sources'])
        verify_hashes({name:digest for name,digest in frozen['evidence'].items()
                       if name.startswith('reports/') or name in frozen['models'][kind]['weights']})
        if (kind!=job['kind'] or Path(sys.prefix).resolve()!=(ROOT/MODELS[kind][2]).resolve()
                or hashlib.sha256(environment_lock().encode()).hexdigest()!=frozen['models'][kind]['environment_sha256']
                or job['seed']!=20261117+100*job['outer_fold']+job['inner_fold']
                or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0')):
            raise RuntimeError('Worker native environment/seed/fallback identity differs.')
        if sha(out/'rows.npz')!=job['rows_sha256']:raise RuntimeError('Worker training packet changed.')
        with np.load(out/'rows.npz',allow_pickle=False) as packet:
            if set(packet.files)!={'train_ids','query_ids','outer_train_ids','outer_validation_ids','train_labels'}:
                raise RuntimeError('Worker packet contains unexpected labels/data.')
            a,b=validate_rows(packet['train_ids'],packet['query_ids'],packet['outer_train_ids'],packet['outer_validation_ids'],packet['train_labels'])
            y=packet['train_labels'].copy()
        attributes=out.parent.parent/'attributes.pkl'
        if sha(attributes)!=job['attributes_sha256']:raise RuntimeError('Unlabeled worker attributes changed.')
        roster=pd.read_pickle(attributes)
        if 'Transported' in roster or len(roster)!=12970 or np.max(np.r_[a,b])>=len(roster):
            raise RuntimeError('Unlabeled original full roster required.')
        deadline_check(deadline)
        result.update(current_stage='configure',outer_fold=job['outer_fold'],inner_fold=job['inner_fold'],seed=job['seed'],
                      train_ids_sha256=ids_hash(a),query_ids_sha256=ids_hash(b),train_labels_sha256=ids_hash(y))
        write(out/'progress.json',result)
        if kind=='raw':
            framework,wrapper,fit,_,version=audit.initialize_reference('mitra',out)
        else:
            import kumo_candidate as wrapper
            wrapper.configure_runtime(out);wrapper.configure();framework=wrapper.iterate;fit=wrapper.fit;version=wrapper.NAME
        wrapper.DEADLINE=deadline
        matrices={False:framework.build_features(roster,False)}
        if frame_hash(matrices[False][0])!=job['features_sha256']:raise RuntimeError('Original fullroster feature values changed.')
        result['current_stage']='fit_predict';write(out/'progress.json',result)
        p,model=fit(version,matrices,a,y,b,job['seed']);p=valid_vector(p,len(b))
        history.check_fold_log(model.metadata,'mitra' if kind=='raw' else 'kumo',a,b,y,job['seed'])
        if kind=='kumo' and (model.metadata.get('estimators')!=8 or any(r['rows']!=len(a) for r in model.metadata['last_forward_contexts'])):
            raise RuntimeError('Full eight-view Kumo context required.')
        write(out/'metadata.json',model.metadata)
        np.savez(out/'prediction.npz',prob=p,query_ids=b)
        deadline_check(deadline);verify_hashes(frozen['sources'])
        result.update(status='worker_complete',prediction_rows=len(b),query_labels_read=False,outer_validation_labels_read=False,
            metadata_sha256=sha(out/'metadata.json'),prediction_sha256=sha(out/'prediction.npz'),
            sources_sha256=hashlib.sha256(json.dumps(frozen['sources'],sort_keys=True).encode()).hexdigest())
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
    finally:
        try:
            if model is not None:model.release()
            model=None
            if wrapper is not None:wrapper.release_memory()
            gc.collect();deadline_check(deadline);result['release_complete']=True
        except BaseException as error:
            result.update(status='failed_inconclusive',cleanup_error=str(error),release_complete=False)
            result.setdefault('error_type',type(error).__name__);result.setdefault('error',str(error))
        result['seconds']=time.monotonic()-started;write(out/'worker_result.json',result)
    return result


def supervise(process,deadline):
    while process.poll() is None:
        if time.monotonic()>=deadline:
            process.kill();process.wait(timeout=5)
            raise TimeoutError('Nested experiment parent killed owned worker at7200s deadline.')
        try:process.wait(timeout=min(1,max(.001,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:pass
    if process.returncode!=0:raise RuntimeError('Inner worker process exited abnormally: '+str(process.returncode))
    deadline_check(deadline)


def execute_job(kind,out,deadline):
    command=[str(ROOT/MODELS[kind][2]/'bin/python'),str(Path(__file__).resolve()),'--worker',kind,'--out',str(out),'--deadline',str(deadline)]
    process=None
    try:
        with (out/'native.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            supervise(process,deadline)
        result=read(out/'worker_result.json')
        if result.get('status')!='worker_complete' or result.get('release_complete') is not True:
            raise RuntimeError('Inner '+kind+' failed: '+str(result.get('error_type'))+': '+str(result.get('error')))
        return result
    finally:
        if process is not None and process.poll() is None:process.kill();process.wait(timeout=5)


def verify_prediction(out,record,job,train_ids,query_ids,y):
    if (record.get('status')!='worker_complete' or record.get('release_complete') is not True
            or record.get('kind')!=job['kind'] or record.get('seed')!=job['seed']
            or record.get('outer_fold')!=job['outer_fold'] or record.get('inner_fold')!=job['inner_fold']
            or record.get('train_ids_sha256')!=ids_hash(train_ids) or record.get('query_ids_sha256')!=ids_hash(query_ids)
            or record.get('train_labels_sha256')!=ids_hash(y) or record.get('prediction_rows')!=len(query_ids)
            or record.get('query_labels_read') is not False or record.get('outer_validation_labels_read') is not False
            or record.get('prediction_sha256')!=sha(out/'prediction.npz') or record.get('metadata_sha256')!=sha(out/'metadata.json')):
        raise RuntimeError('Completed inner fit identity/label boundary differs.')
    with np.load(out/'prediction.npz',allow_pickle=False) as cache:
        if not np.array_equal(cache['query_ids'],query_ids):raise RuntimeError('Inner prediction IDs reordered.')
        return valid_vector(cache['prob'],len(query_ids))


def metrics(p,y,splits):
    p=valid_vector(p,len(y))
    return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
        fold_accuracy=[float(accuracy_score(y[b],p[b]>=.5)) for _,b in splits],
        fold_log_loss=[float(log_loss(y[b],p[b],labels=[0,1])) for _,b in splits])
def paired(p,reference,y,splits):
    fixed=(p>=.5)==y;prior=(reference>=.5)==y
    return dict(repaired=int((fixed&~prior).sum()),broken=int((~fixed&prior).sum()),
        fold_net_correct=[int(fixed[b].sum()-prior[b].sum()) for _,b in splits])
def continuation(scores):
    candidate,constant=scores['planet'],scores['constant']
    checks=dict(accuracy_at_least_all=candidate['accuracy']+1e-12>=max(scores[k]['accuracy'] for k in ['raw','kumo','constant']),
        log_loss_not_higher=candidate['log_loss']<=constant['log_loss']+1e-12,
        at_least_two_folds=sum(a+1e-12>=b for a,b in zip(candidate['fold_accuracy'],constant['fold_accuracy']))>=2,
        strict_overall_improvement=candidate['accuracy']>constant['accuracy']+1e-12 or candidate['log_loss']<constant['log_loss']-1e-12)
    return dict(passed=all(checks.values()),checks=checks,automatic_export=False,automatic_upload=False,
                stable_superiority_claim=False,next_if_pass='separate frozen group confirmation review only')


def run():
    if OUTPUT.exists() or any((REPORT/name).exists() for name in ['freeze.json','progress.json','summary.json','failure.json']):
        raise FileExistsError('No overwrite/resume of nested experiment.')
    started=time.monotonic();deadline=started+BUDGET
    frozen=protocol_contract();deadline_check(deadline)
    OUTPUT.mkdir(parents=True);completed=0;current_job=None
    try:
        inputs=load_inputs(frozen);dev,y=inputs['dev'],inputs['y'];splits=inputs['splits'];planets=inputs['planets']
        if len(dev)!=6954 or len(splits)!=3:raise RuntimeError('Fixed original development/outer folds required.')
        attributes=OUTPUT/'attributes.pkl';inputs['roster'].to_pickle(attributes)
        if 'Transported' in inputs['roster']:raise RuntimeError('No target column may be serialized to attributes.')
        attribute_sha=sha(attributes)
        for folder in [OUTPUT,REPORT]:write(folder/'freeze.json',frozen)
        shutil.copy2(ROOT/PLAN,OUTPUT/'protocol.json')
        predictions={**inputs['probabilities'],'constant':np.full(len(dev),np.nan),'planet':np.full(len(dev),np.nan)}
        folds=[]
        for outer,(a,b) in enumerate(splits):
            deadline_check(deadline);assert_frozen(frozen)
            inner_splits=list(StratifiedKFold(3,shuffle=True,random_state=20261117+outer).split(a,y[a]))
            inner_prob={key:np.full(len(a),np.nan) for key in MODELS};coverage={key:np.zeros(len(a),dtype=int) for key in MODELS}
            for inner,(ia,ib) in enumerate(inner_splits):
                training,query=dev[a[ia]],dev[a[ib]];labels=y[a[ia]]
                validate_rows(training,query,dev[a],dev[b],labels)
                for kind in MODELS:
                    deadline_check(deadline);assert_frozen(frozen)
                    out=OUTPUT/'jobs'/f'outer{outer}_inner{inner}_{kind}';out.mkdir(parents=True)
                    current_job=str(out.relative_to(ROOT))
                    np.savez(out/'rows.npz',train_ids=training,query_ids=query,train_labels=labels,
                             outer_train_ids=dev[a],outer_validation_ids=dev[b])
                    job=dict(kind=kind,outer_fold=outer,inner_fold=inner,seed=20261117+100*outer+inner,
                             rows_sha256=sha(out/'rows.npz'),attributes_sha256=attribute_sha,features_sha256=inputs['features_sha256'])
                    write(out/'job.json',job)
                    child=execute_job(kind,out,deadline)
                    p=verify_prediction(out,child,job,training,query,labels)
                    if coverage[kind][ib].any():raise RuntimeError('Duplicate innerOOF rows.')
                    inner_prob[kind][ib]=p;coverage[kind][ib]+=1;completed+=1
                    for name in ['job.json','worker_result.json','metadata.json']:
                        target=REPORT/'jobs'/out.name/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/name,target)
                    progress=dict(status='incomplete',completed_fits=completed,total_fits=18,completed_outer_folds=outer,
                                  current_job=current_job,seconds=time.monotonic()-started)
                    for folder in [OUTPUT,REPORT]:write(folder/'progress.json',progress)
                    print(f'Nested mixture outer{outer+1} inner{inner+1} {kind}: fit{completed}/18 complete',flush=True)
            if any(not np.all(coverage[kind]==1) or not np.isfinite(inner_prob[kind]).all() for kind in MODELS):
                raise RuntimeError('Gate fit requires complete exactly-once innerOOF.')
            constant=fit_gate(inner_prob['raw'],inner_prob['kumo'],y[a],planets[a],'constant')
            gate=fit_gate(inner_prob['raw'],inner_prob['kumo'],y[a],planets[a],'planet',constant['alpha'][0])
            predictions['constant'][b],constant_unknown=apply_gate(constant,predictions['raw'][b],predictions['kumo'][b],planets[b])
            predictions['planet'][b],unknown=apply_gate(gate,predictions['raw'][b],predictions['kumo'][b],planets[b])
            inner_fold_ids=np.empty(len(a),dtype=np.int64)
            for fold,(_,query) in enumerate(inner_splits):inner_fold_ids[query]=fold
            np.savez(OUTPUT/f'outer{outer}_inner_oof.npz',outer_train_ids=dev[a],raw=inner_prob['raw'],kumo=inner_prob['kumo'],
                     inner_fold_id=inner_fold_ids)
            np.savez(OUTPUT/f'outer{outer}_predictions.npz',outer_validation_ids=dev[b],
                     **{name:values[b] for name,values in predictions.items()})
            row=dict(outer_fold=outer,status='complete',outer_train_ids_sha256=ids_hash(dev[a]),outer_validation_ids_sha256=ids_hash(dev[b]),
                     inner_split_seed=20261117+outer,constant=constant,planet=gate,unknown_query_planets=unknown,
                     cached_outer_fit_seed=20261112+outer,inner_oof_sha256=sha(OUTPUT/f'outer{outer}_inner_oof.npz'),
                     outer_predictions_sha256=sha(OUTPUT/f'outer{outer}_predictions.npz'))
            folds.append(row)
            for folder in [OUTPUT,REPORT]:write(folder/f'outer{outer}.json',row)
        if completed!=18 or len(folds)!=3:raise RuntimeError('All18 fits and3 complete outer folds required.')
        scores={key:metrics(p,y,splits) for key,p in predictions.items()}
        assessment=dict(status='complete',completed_fits=18,completed_outer_folds=3,metrics=scores,
            paired={key:paired(predictions['planet'],predictions[key],y,splits) for key in ['raw','kumo','constant']},
            continuation=continuation(scores),old_holdout_labels_read=False,automatic_export=False,automatic_upload=False,
            validation_kind='adaptive_development_nested_cross_validation',budget_seconds=BUDGET,
            disclosure='Repeated adaptive development, not independent validation; gate uses innerOOF only. Original outer fits reuse seed20261112+fold; inner seeds follow frozen protocol.',
            seconds=time.monotonic()-started)
        fold_ids=np.empty(len(dev),dtype=np.int64)
        for fold,(_,query) in enumerate(splits):fold_ids[query]=fold
        np.savez(OUTPUT/'oof.npz',dev_ids=dev,fold_id=fold_ids,**predictions)
        pd.DataFrame([dict(candidate=k,**{name:scores[k][name] for name in ['accuracy','log_loss']}) for k in scores]).to_csv(OUTPUT/'results.csv',index=False)
        assert_frozen(frozen);deadline_check(deadline)
        assessment['result_files_sha256']={name:sha(OUTPUT/name) for name in ['oof.npz','results.csv',*[f'outer{k}.json' for k in range(3)]]}
        for folder in [OUTPUT,REPORT]:write(folder/'summary.json',assessment);write(folder/'progress.json',assessment)
        shutil.copy2(OUTPUT/'results.csv',REPORT/'results.csv')
        return assessment
    except BaseException as error:
        failure=dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),completed_fits=completed,
                     current_job=current_job,budget_seconds=BUDGET,seconds=time.monotonic()-started,
                     no_complete_candidate=True,automatic_export=False,automatic_upload=False)
        if current_job:
            for name,key in [('worker_result.json','worker_failure'),('progress.json','worker_partial')]:
                path=ROOT/current_job/name
                if path.exists():failure[key]=read(path)
        if (OUTPUT/'oof.npz').exists():(OUTPUT/'oof.npz').replace(OUTPUT/'oof.partial.npz')
        for folder in [OUTPUT,REPORT]:
            if (folder/'results.csv').exists():(folder/'results.csv').replace(folder/'results.partial.csv')
            (folder/'summary.json').unlink(missing_ok=True)
            write(folder/'failure.json',failure);write(folder/'progress.json',failure)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=list(MODELS));parser.add_argument('--out',type=Path);parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:
        if args.out is None or args.deadline is None:parser.error('Worker requires fixed job folder and shared deadline.')
        worker(args.worker,args.out,args.deadline)
    else:print(json.dumps(run(),indent=2))
