"""One adaptive historical fold: fresh fixed15 compact13/counts15/peer17.

No other folds, tuning, export, submission, or holdout-label access. Cached v81
fold0 used47 updates and is only a historical reference; new compact13 is the
matched15-step control. All three fits receive outer-training labels only.
"""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
import audit_top5_references as audit
import exaone_validation_data as data
import mitra_candidate as base
import mitra_full_episode as training
import mitra_counts_fixed15_online_first as counts
import mitra_peercryo_fixed15_ready as peer

ROOT=Path(__file__).resolve().parent
PLAN='reports/error_rescue_20261117/feature_probe_protocol.json'
OUT=ROOT/'artifacts/error_rescue_20261117/feature_probe_fold0'
REPORT=ROOT/'reports/error_rescue_20261117/feature_probe_result.json'
CONTROL='compact13_fixed15_fold0'
RECIPES=[(CONTROL,13),(counts.NAME,15),(peer.NAME,17)]
SEED,STEPS,BUDGET=20261112,15,7200
OWN=[Path(__file__).name,PLAN,'mitra_counts_fixed15_online_first.py','mitra_peercryo_fixed15_ready.py',
     'mitra_candidate.py','mitra_full_episode.py','mitra_full_episode_probe.py','prepare_v84_online_first.py',
     'audit_top5_references.py','exaone_validation_data.py','three_family_mean_screen.py','prior_adapt_mean_screen.py',
     'iterate.py','experiment.py','requirements-mitra.lock.txt','reports/error_rescue_20261117/catalog.json']


def committed(names):
    names=sorted(set(names))
    tracked=subprocess.check_output(['git','ls-files','-z','--',*names],cwd=ROOT).decode()
    if set(filter(None,tracked.split('\0')))!=set(names) or subprocess.check_output(['git','status','--porcelain','--',*names],cwd=ROOT).strip():
        raise RuntimeError('Commit diagnostic recipe and all source dependencies before execution.')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def verify_sources(frozen):
    counts.require_false()
    for name,digest in frozen['sources'].items():
        if audit.file_hash(ROOT/name)!=digest:raise RuntimeError('Source/evidence changed: '+name)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=frozen['environment_sha256']:
        raise RuntimeError('Environment changed.')


def fit_predict(name,features,train_ids,y_train,query_ids,rosters):
    """This function never receives any query/outer-validation targets."""
    train_ids=np.asarray(train_ids,dtype=np.int64);query_ids=np.asarray(query_ids,dtype=np.int64)
    if (len(train_ids)!=4636 or len(query_ids)!=2318 or len(np.unique(train_ids))!=4636
            or len(np.unique(query_ids))!=2318 or np.intersect1d(train_ids,query_ids).size):
        raise ValueError('Exact disjoint fold0 membership required.')
    y_train=base.binary_labels(y_train,len(train_ids))
    x=features.loc[train_ids];query=features.loc[query_ids]
    base.seed_everything(SEED,'mps')
    if name==CONTROL:
        prep=base.MitraPreprocessor().fit(x,y_train,SEED);predictor_type=base.MitraPredictor;expected_columns=base.COMPACT
    elif name==counts.NAME:
        prep=counts.CountsPreprocessor(rosters['counts']).fit(x,y_train,SEED)
        predictor_type=counts.CountsPredictor;expected_columns=counts.COLUMNS
    elif name==peer.NAME:
        prep=peer.PeerCryoPreprocessor(rosters['peer']).fit(x,y_train,SEED)
        predictor_type=peer.PeerCryoPredictor;expected_columns=peer.COLUMNS
    else:raise ValueError(name)
    support=prep.transform(x)
    if support.shape!=(4636,len(expected_columns)):
        raise RuntimeError('Actual model feature count changed.')
    log=dict(status='incomplete',version=name,seed=SEED,fold=0,fixed_steps=15,training_rows=4636,query_rows=2318,
        training_ids_sha256=audit.row_hash(train_ids),query_ids_sha256=audit.row_hash(query_ids),
        training_labels_sha256=audit.row_hash(y_train),input_columns=list(expected_columns),
        inner_selection_performed=False,checkpoint_sha256=base.WEIGHT_SHA,updates=[])
    path=OUT/(name+'.json');started=time.monotonic()
    model=optimizer=train_tensor=targets=predictor=None
    try:
        model,config=training.fresh_model('mps')
        train_tensor=torch.as_tensor(support,device='mps');targets=torch.as_tensor(y_train,device='mps')
        optimizer=base.optimizer_for(model);rng=np.random.default_rng(SEED)
        for step in range(1,16):
            counts.require_false()
            log['updates'].append(training.episode_step(model,optimizer,train_tensor,targets,train_ids,rng,step))
            base.save_training_log(path,log)
            print(f'{name} fold0 update{step}/15',flush=True)
        optimizer=train_tensor=targets=None;base.release_memory()
        predictor=predictor_type(prep,config,model.state_dict(),support,y_train,'mps',log,query_chunk=2318)
        model=None;base.release_memory()
        p=np.asarray(predictor.predict_proba(query))[:,1]
        if p.shape!=(2318,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise RuntimeError('Complete finite2318-query probabilities required.')
        log.update(status='complete',optimizer_steps=15,seconds=time.monotonic()-started,
            prediction_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest())
        base.save_training_log(path,log)
        return p,log
    except BaseException as error:
        log.update(status='failed_incomplete',error_type=type(error).__name__,error=str(error))
        base.save_training_log(path,log);raise
    finally:
        if predictor is not None:predictor.release()
        model=optimizer=train_tensor=targets=predictor=None;base.release_memory()


def paired(p,reference,y,mask):
    a=(p>=.5)==y;b=(reference>=.5)==y
    return dict(rows=int(mask.sum()),repairs=int((a&~b&mask).sum()),breaks=int((~a&b&mask).sum()),
                net_correct=int(((a.astype(int)-b.astype(int))*mask).sum()))


def run():
    if OUT.exists() or REPORT.exists():raise RuntimeError('No overwrite/resume of fixed diagnostic.')
    plan=json.loads((ROOT/PLAN).read_text())
    if (plan['seed']!=SEED or plan['fold_index']!=0 or plan['fixed_steps']!=15
            or plan['recipes']!=[n for n,_ in RECIPES] or plan['budget_seconds']!=BUDGET):
        raise RuntimeError('Diagnostic scope differs from fixed protocol.')
    historical=ROOT/plan['historical_fold0_log']
    old=json.loads(historical.read_text())
    if (audit.file_hash(historical)!=plan['historical_fold0_log_sha256'] or old.get('selected_steps')!=47
            or old.get('refit_optimizer_steps')!=47 or old.get('seed')!=SEED):
        raise RuntimeError('Historical47step reference identity changed.')
    previous,lock=counts.verify_reference()
    names=OWN+[n for n in previous['source_hashes'] if n.startswith('third_party/mitra/')]
    commit=committed(names)
    frozen=dict(source_commit=commit,sources={n:audit.file_hash(ROOT/n) for n in names},
                environment_sha256=hashlib.sha256(lock.encode()).hexdigest())
    frozen['sources'][plan['historical_fold0_log']]=audit.file_hash(historical)
    OUT.mkdir(parents=True);started=time.monotonic()
    def timeout(*_):raise TimeoutError('Whole single-fold three-fit diagnostic exceeds7200s.')
    old_handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    base.DEADLINE=started+BUDGET
    try:
        inputs=data.load_data()  # Dev-only targets; never read old1739 targets.
        splits,refs,reference_evidence=data.random_references(inputs)
        a,b=splits[0];train_ids=inputs['dev'][a];query_ids=inputs['dev'][b]
        folds=np.full(6954,-1,dtype=int)
        for i,(_,v) in enumerate(splits):folds[v]=i
        anchor_path=ROOT/'artifacts/data_error_diagnosis_20261117/aligned_rows.csv'
        catalog=json.loads((ROOT/'reports/error_rescue_20261117/catalog.json').read_text())
        if audit.file_hash(anchor_path)!=catalog['evidence_sha256'][str(anchor_path.relative_to(ROOT))]:
            raise RuntimeError('914-case anchor source changed.')
        anchor=pd.read_csv(anchor_path,dtype={'PassengerId':str})
        if (not np.array_equal(anchor.row_id,inputs['dev']) or not np.array_equal(anchor.y,inputs['y'])
                or not np.array_equal(anchor.fold,folds) or not np.array_equal(anchor.PassengerId,inputs['passengers'])):
            raise RuntimeError('Anchor dev/labels/folds/PassengerIds differ.')
        cached_v81=refs['v81_Mitra_full_episode50']
        if not np.allclose(anchor.p_v81,cached_v81,atol=1e-15,rtol=0):raise RuntimeError('Cached v81 reference differs.')
        masks={'all':np.ones(len(b),dtype=bool),'five_common_wrong':anchor.family_wrong_count.to_numpy()[b]==5,
               'five_common_correct':anchor.family_wrong_count.to_numpy()[b]==0}
        if (anchor.family_wrong_count==5).sum()!=914 or (anchor.family_wrong_count==0).sum()!=5263:
            raise RuntimeError('Original common-error sets differ.')
        rosters={'counts':counts.load_roster(),'peer':peer.load_roster()}
        features=inputs['features']
        for roster in rosters.values():pd.testing.assert_frame_equal(features,roster.plain,check_exact=True)
        frozen['sources'].update(inputs['evidence']);frozen['sources'].update(reference_evidence)
        frozen['sources'][str(anchor_path.relative_to(ROOT))]=audit.file_hash(anchor_path)
        frozen.update(training_ids_sha256=audit.row_hash(train_ids),query_ids_sha256=audit.row_hash(query_ids),
            fold_assignment_sha256=audit.row_hash(folds),protocol=plan)
        audit.json_write(OUT/'freeze.json',frozen);(OUT/'environment.lock.txt').write_text(lock)
        if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        predictions={};logs={}
        for name,_ in RECIPES:
            verify_sources(frozen)
            p,log=fit_predict(name,features,train_ids,inputs['y'][a].copy(),query_ids,rosters)
            predictions[name]=p;logs[name]=log
            np.savez(OUT/(name+'.npz'),prob=p,train_ids=train_ids,query_ids=query_ids,
                     query_passenger_ids=inputs['passengers'][b],query_y=inputs['y'][b],
                     common_wrong_mask=masks['five_common_wrong'],common_correct_mask=masks['five_common_correct'])
        y=inputs['y'][b];all_p={'historical_v81_fold0_47steps':cached_v81[b],**predictions}
        metrics={name:dict(accuracy=float(np.mean((p>=.5)==y)),log_loss=float(log_loss(y,p,labels=[0,1])),
            common_wrong_correct=int((((p>=.5)==y)&masks['five_common_wrong']).sum()),
            common_correct_damaged=int((((p>=.5)!=y)&masks['five_common_correct']).sum())) for name,p in all_p.items()}
        comparisons={}
        for left,right in [(counts.NAME,CONTROL),(peer.NAME,counts.NAME),(peer.NAME,CONTROL),
                (counts.NAME,'historical_v81_fold0_47steps'),(peer.NAME,'historical_v81_fold0_47steps'),
                (CONTROL,'historical_v81_fold0_47steps')]:
            comparisons[left+' vs '+right]={key:paired(all_p[left],all_p[right],y,mask) for key,mask in masks.items()}
        # The exact per-episode row draw must agree across all three15-step fits.
        draws=lambda name:[(r['support_ids_sha256'],r['query_ids_sha256']) for r in logs[name]['updates']]
        if any(draws(n)!=draws(CONTROL) for n,_ in RECIPES):raise RuntimeError('Matched episode sampling changed.')
        verify_sources(frozen);base.check_deadline()
        result=dict(status='complete_adaptive_single_fold_diagnostic',fold=0,seed=SEED,training_rows=4636,query_rows=2318,
            fixed_steps=15,matched_episode_ids=True,mask_sizes={k:int(v.sum()) for k,v in masks.items()},
            metrics=metrics,paired=comparisons,source_commit=commit,source_hashes=frozen['sources'],
            train_ids_sha256=audit.row_hash(train_ids),query_ids_sha256=audit.row_hash(query_ids),
            seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
            interpretation=plan['interpretation'],evidence_files={str(p.relative_to(ROOT)):audit.file_hash(p) for p in OUT.glob('*.npz')})
        audit.json_write(OUT/'result.json',result);audit.json_write(REPORT,result)
        print(json.dumps(dict(metrics=metrics,paired=comparisons),indent=2),flush=True)
        return result
    except BaseException as error:
        failure=dict(status='failed_incomplete_no_three_fit_comparison',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
        audit.json_write(OUT/'failure.json',failure);audit.json_write(REPORT.with_name('feature_probe_failure.json'),failure)
        raise
    finally:
        base.release_memory();signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();run()
