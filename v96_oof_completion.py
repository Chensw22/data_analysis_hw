"""Complete historical v96 fixed15 OOF: reuse fold0, run folds1/2 only."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):raise RuntimeError('Fresh FALSE process required')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import audit_top5_references as audit
import exaone_validation_data as data
import mitra_candidate as base
import mitra_full_episode as training
import mitra_counts_fixed15_online_first as counts
import error_rescue_feature_probe as old
from sklearn.metrics import log_loss
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/v96_oof_completion'
OUT=ROOT/'artifacts/v96_oof_completion'
PLAN=REPORT/'protocol.json'
HIST=ROOT/'artifacts/error_rescue_20261117/feature_probe_fold0'
SEED=20261112
BUDGET=3600
read=lambda p:json.loads(Path(p).read_text())
write=audit.json_write
sha=audit.file_hash

def historical(inputs,splits):
    result=read(HIST/'result.json');public=read(ROOT/'reports/error_rescue_20261117/feature_probe_result.json')
    if result!=public or result['status']!='complete_adaptive_single_fold_diagnostic':raise RuntimeError('Historical result mismatch')
    for name,digest in result['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Historical source changed '+name)
    a,b=splits[0];dev=inputs['dev'];y=inputs['y'];path=HIST/(counts.NAME+'.npz');log=read(HIST/(counts.NAME+'.json'))
    if sha(path)!=result['evidence_files'][str(path.relative_to(ROOT))]:raise RuntimeError('Fold0 cache SHA mismatch')
    with np.load(path,allow_pickle=False) as z:
        for k,want in [('train_ids',dev[a]),('query_ids',dev[b]),('query_y',y[b])]:np.testing.assert_array_equal(z[k],want)
        p=z['prob'].copy()
    if (log['status']!='complete' or log['seed']!=SEED or log['optimizer_steps']!=15 or log['fixed_steps']!=15
        or log['training_rows']!=4636 or log['query_rows']!=2318 or log['input_columns']!=counts.COLUMNS
        or log['checkpoint_sha256']!=base.WEIGHT_SHA or log['training_ids_sha256']!=audit.row_hash(dev[a])
        or log['query_ids_sha256']!=audit.row_hash(dev[b]) or log['training_labels_sha256']!=audit.row_hash(y[a])
        or log['prediction_sha256']!=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()):raise RuntimeError('Fold0 recipe/rows/log identity failed')
    updates=log['updates']
    if len(updates)!=15 or any(u['step']!=i or u['cumulative_optimizer_steps']!=i or u['support_rows']!=1024 or u['query_rows']!=256 or not u['disjoint_ids'] for i,u in enumerate(updates,1)):raise RuntimeError('Historical15 complete updates required')
    metrics=result['metrics'][counts.NAME]
    if abs(np.mean((p>=.5)==y[b])-metrics['accuracy'])>1e-12 or abs(log_loss(y[b],p)-metrics['log_loss'])>1e-9:raise RuntimeError('Historical fold score mismatch')
    files=[path,HIST/(counts.NAME+'.json'),HIST/'result.json',HIST/'freeze.json',ROOT/'reports/error_rescue_20261117/feature_probe_result.json']
    return p,log,{str(f.relative_to(ROOT)):sha(f) for f in files}

def fit_fold(fold,features,train_ids,y_train,query_ids,roster):
    """Same historical fixed15 mathematics; never receives query targets."""
    train_ids=np.asarray(train_ids,dtype=np.int64);query_ids=np.asarray(query_ids,dtype=np.int64)
    if len(train_ids)!=4636 or len(query_ids)!=2318 or np.intersect1d(train_ids,query_ids).size:raise ValueError('Disjoint original fold required')
    y_train=base.binary_labels(y_train,len(train_ids));x=features.loc[train_ids];query=features.loc[query_ids]
    base.seed_everything(SEED,'mps');prep=counts.CountsPreprocessor(roster).fit(x,y_train,SEED);support=prep.transform(x)
    if support.shape!=(4636,15) or prep.active_columns!=counts.COLUMNS:raise RuntimeError('15 actual inputs required')
    log=dict(status='incomplete',version=counts.NAME,seed=SEED,fold=fold,fixed_steps=15,training_rows=4636,query_rows=2318,
        training_ids_sha256=audit.row_hash(train_ids),query_ids_sha256=audit.row_hash(query_ids),training_labels_sha256=audit.row_hash(y_train),
        input_columns=list(counts.COLUMNS),inner_selection_performed=False,checkpoint_sha256=base.WEIGHT_SHA,updates=[])
    path=OUT/f'fold{fold}_training.json';start=time.monotonic();model=optimizer=train_tensor=targets=predictor=None
    try:
        model,config=training.fresh_model('mps');train_tensor=torch.as_tensor(support,device='mps');targets=torch.as_tensor(y_train,device='mps')
        optimizer=base.optimizer_for(model);rng=np.random.default_rng(SEED)
        for step in range(1,16):
            counts.require_false();log['updates'].append(training.episode_step(model,optimizer,train_tensor,targets,train_ids,rng,step))
            base.save_training_log(path,log);print(f'v96 completion fold{fold} update{step}/15',flush=True)
        optimizer=train_tensor=targets=None;base.release_memory()
        predictor=counts.CountsPredictor(prep,config,model.state_dict(),support,y_train,'mps',log,query_chunk=2318)
        model=None;base.release_memory();p=np.asarray(predictor.predict_proba(query))[:,1]
        if p.shape!=(2318,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('Complete finite prediction required')
        log.update(status='complete',optimizer_steps=15,seconds=time.monotonic()-start,prediction_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest())
        base.save_training_log(path,log);return p,log
    except BaseException as exc:
        log.update(status='failed_incomplete',error_type=type(exc).__name__,error=str(exc));base.save_training_log(path,log);raise
    finally:
        if predictor is not None:predictor.release()
        model=optimizer=train_tensor=targets=predictor=None;base.release_memory()

def load():
    inputs=data.load_data();splits,refs,evidence=data.random_references(inputs)
    p,log,files=historical(inputs,splits);evidence.update(inputs['evidence']);evidence.update(files)
    return inputs,splits,refs,evidence,p,log

def run(check=False):
    plan=read(PLAN)
    if plan['seed_each_fold']!=SEED or plan['new_folds']!=[1,2] or plan['fixed_updates']!=15 or plan['budget_seconds']!=BUDGET:raise RuntimeError('Plan mismatch')
    previous,lock=counts.verify_reference();inputs,splits,refs,evidence,p0,log0=load()
    if check:
        print(json.dumps(dict(status='historical_fold0_authenticated_no_model_execution',rows=len(p0),seed=log0['seed'],updates=log0['optimizer_steps'],accuracy=float(np.mean((p0>=.5)==inputs['y'][splits[0][1]])))))
        return
    if OUT.exists() or (REPORT/'result.json').exists():raise RuntimeError('No overwrite/resume')
    names=set(old.OWN)|{Path(__file__).name,str(PLAN.relative_to(ROOT))}
    names.update(n for n in previous['source_hashes'] if n.startswith('third_party/mitra/'))
    commit=old.committed(names);sources={n:sha(ROOT/n) for n in names};sources.update(evidence)
    catalog_path=ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv';catalog=read(ROOT/'reports/error_rescue_20261117/catalog.json')
    relative=str(catalog_path.relative_to(ROOT))
    if sha(catalog_path)!=plan['catalog_rows_sha256']:raise RuntimeError('Catalog SHA mismatch')
    sources[relative]=sha(catalog_path)
    frozen=dict(source_commit=commit,sources=sources,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),protocol=plan)
    OUT.mkdir(parents=True);write(OUT/'freeze.json',frozen);write(REPORT/'freeze.json',frozen)
    start=time.monotonic();base.DEADLINE=start+BUDGET
    def timeout(*_):raise TimeoutError('Twofoldcompletion3600s exceeded')
    handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    try:
        if not torch.backends.mps.is_available():raise RuntimeError('OriginalMPS required')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        dev,y=inputs['dev'],inputs['y'];features=inputs['features'];roster=counts.load_roster()
        pd.testing.assert_frame_equal(features,roster.plain,check_exact=True)
        p=np.full(len(y),np.nan);folds=np.full(len(y),-1);p[splits[0][1]]=p0;logs={'0':log0}
        for i,(_,b) in enumerate(splits):folds[b]=i
        for i in [1,2]:
            old.verify_sources(frozen);a,b=splits[i];q,log=fit_fold(i,features,dev[a],y[a].copy(),dev[b],roster)
            p[b]=q;logs[str(i)]=log
            np.savez_compressed(OUT/f'fold{i}.npz',prob=q,train_ids=dev[a],query_ids=dev[b],passenger_ids=np.asarray(inputs['passengers'][b],dtype=str))
        np.testing.assert_array_equal(p[splits[0][1]],p0)
        if not np.isfinite(p).all():raise RuntimeError('PartialOOF forbidden')
        # Outcome-defined hard sets are reporting only after prediction completion.
        rows=pd.read_csv(catalog_path,dtype={'PassengerId':str})
        for key,want in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[key],want)
        raw=refs['v71_Mitra2_zeroshot'];hard=rows.all_recipes_still_wrong.to_numpy(bool)
        if hard.sum()!=546:raise RuntimeError('Historical546 changed')
        def stats(mask):return dict(rows=int(mask.sum()),accuracy=float(np.mean(((p>=.5)==y)[mask])),raw_accuracy=float(np.mean(((raw>=.5)==y)[mask])),
            correct=int((((p>=.5)==y)&mask).sum()),paired_raw=old.paired(p,raw,y,mask))
        result=dict(status='complete_v96_fixed15_original3fold_oof',version=counts.NAME,seed_each_fold=SEED,fixed_updates=15,
            historical_fold0_reused=True,new_folds=[1,2],accuracy=float(np.mean((p>=.5)==y)),log_loss=float(log_loss(y,p)),
            raw_accuracy=float(np.mean((raw>=.5)==y)),raw_log_loss=float(log_loss(y,raw)),all=stats(np.ones(len(y),bool)),hard546=stats(hard),
            folds={str(i):dict(all=stats(folds==i),hard546=stats((folds==i)&hard)) for i in range(3)},
            limitations=plan['limitations'],seconds=time.monotonic()-start,source_commit=commit,source_hashes=sources)
        np.savez_compressed(OUT/'oof.npz',prob=p,raw=raw,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        result['oof_sha256']=sha(OUT/'oof.npz');old.verify_sources(frozen);base.check_deadline()
        write(OUT/'result.json',result);write(REPORT/'result.json',result)
        print(json.dumps({k:result[k] for k in ['accuracy','log_loss','all','hard546','seconds']},indent=2))
    except BaseException as exc:
        failure=dict(status='failed_incomplete_no_complete_oof',type=type(exc).__name__,error=str(exc));write(OUT/'failure.json',failure);write(REPORT/'failure.json',failure);raise
    finally:base.release_memory();signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');run(parser.parse_args().check)
