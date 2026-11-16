"""One fixed development-only missing-flags screen; no export or upload."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import StratifiedKFold

import iterate
import mitra_missing_flags as core
import three_family_mean_screen as common

base,audit=core.base,common.audit
ROOT=Path(__file__).resolve().parent
RUN,NAME,CONTROL,SEED='run_052',core.NAME,'v71_Mitra2_zeroshot',20261112
REVIEW='reports/top5_missing_flags_review.json'
REVIEW_SHA='462a75daa17d50e8a908a755a3461d35dae20ef566fc54fd06d068fd55929d1e'
PROBE='artifacts/probes/mitra_missing_flags/result.json'
PROBE_REPORT='reports/mitra_missing_flags_resources/result.json'
OWN=['mitra_missing_flags.py','test_mitra_missing_flags.py',
    'mitra_missing_flags_candidate.py','test_mitra_missing_flags_candidate.py',
    'mitra_missing_flags_probe.py','test_mitra_missing_flags_probe.py',
    'three_family_mean_screen.py','test_three_family_mean_screen.py',
    'audit_top5_references.py','test_audit_top5_references.py','quota_guard.py',
    REVIEW,'reports/official_data_verification.json']
sha=base.sha256


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);audit.json_write(path,value)


def check_quota(session_file):
    from quota_guard import latest_weekly,assess
    try:allowed=assess(latest_weekly(session_file),reserve=30)['can_start_more_work']
    except (OSError,ValueError,TypeError,KeyError):allowed=False
    if not allowed:raise RuntimeError('Fresh weekly reserve guard did not permit launching this stage.')


def freeze_contract():
    if sha(ROOT/REVIEW)!=REVIEW_SHA:raise RuntimeError('Fixed missing-flags review changed.')
    review=json.loads((ROOT/REVIEW).read_text())
    evidence=review['source_sha256']
    for name,digest in evidence.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Reviewed evidence changed: '+name)
    original=json.loads((ROOT/'artifacts/run_038/protocol.json').read_text())
    audit.check_historical_sources(original)
    lock=base.environment_lock()
    if hashlib.sha256(lock.encode()).hexdigest()!=original['mitra_environment_sha256']:
        raise RuntimeError('Original raw-Mitra environment changed.')
    if original['configs'][CONTROL]['query_chunk']!=2318:raise RuntimeError('Original context recipe changed.')
    base.verify_checkpoint(base.CHECKPOINT_DIR)
    sources={name:sha(ROOT/name) for name in sorted(set(original['sources'])|set(OWN))}
    commit=common.require_committed([*sources,*[name for name in evidence if name.startswith('reports/')]])
    return dict(sources=sources,evidence_files=dict(evidence),environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        source_commit=commit,settings=core.SETTINGS,review_sha256=REVIEW_SHA),lock


def assert_frozen(contract):
    for mapping in [contract['sources'],contract['evidence_files']]:
        for name,digest in mapping.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Frozen source/evidence changed: '+name)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=contract['environment_sha256']:
        raise RuntimeError('Frozen environment changed.')
    if contract['settings']!=core.SETTINGS:raise RuntimeError('Fixed inference settings changed.')
    base.verify_checkpoint(base.CHECKPOINT_DIR)


def runtime():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('Automatic MPS fallback must be disabled.')
    if not base.torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
    base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)


@contextmanager
def deadline(seconds):
    old_deadline=base.DEADLINE;old_handler=signal.getsignal(signal.SIGALRM)
    def expired(*_):raise TimeoutError(f'Fixed{seconds}second complete-stage budget exceeded.')
    base.DEADLINE=time.monotonic()+seconds
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        base.DEADLINE=old_deadline


def require_probe(contract):
    common.require_committed([PROBE_REPORT])
    if sha(ROOT/PROBE_REPORT)!=sha(ROOT/PROBE):raise RuntimeError('Resource report/artifact differs.')
    result=json.loads((ROOT/PROBE).read_text())
    if (result.get('status')!='complete' or result.get('device')!='mps'
            or result.get('settings')!=core.SETTINGS or result.get('sources')!=contract['sources']
            or result.get('environment_sha256')!=contract['environment_sha256']
            or result.get('checkpoint_sha256')!=base.WEIGHT_SHA
            or result.get('budget_seconds')!=1200 or result.get('no_accuracy_computed') is not True):
        raise RuntimeError('Complete matching full resource probe required before any new score.')
    cases=result.get('cases',[])
    if len(cases)!=2:raise RuntimeError('Both full resource shapes required.')
    for case,rows in zip(cases,[4636,8693]):
        if (case.get('status')!='complete' or case.get('support_rows')!=rows or case.get('query_rows')!=2318
                or case.get('actual_support_rows')!=rows or case.get('actual_retained_features')!=20
                or any(case.get(key) is not True for key in ['subset_passed','reload_passed','cpu_mps_input_exact','schema_reload_exact','full_support'])
                or any(not isinstance(case.get(key),(int,float)) or not np.isfinite(case[key]) or case[key]>2e-6
                       or case[key]<0 for key in ['subset_max_difference','reload_max_difference'])):
            raise RuntimeError('Full resource shape, precision or persistence proof failed.')
    return {PROBE:sha(ROOT/PROBE),PROBE_REPORT:sha(ROOT/PROBE_REPORT)}


def load_inputs():
    evidence={};official=json.loads((ROOT/'reports/official_data_verification.json').read_text())
    if official.get('competition')!='spaceship-titanic':raise RuntimeError('Unexpected dataset identity.')
    for filename in ['train.csv','test.csv']:
        name='data/raw/'+filename;evidence[name]=sha(ROOT/name)
        if evidence[name]!=official['files'][filename]['official_sha256']:raise RuntimeError('Official data changed.')
    split='artifacts/run_002/split_indices.npz';evidence[split]=sha(ROOT/split)
    with np.load(ROOT/split,allow_pickle=False) as saved:dev=np.sort(saved['development']).astype(np.int64)
    if len(dev)!=6954 or audit.row_hash(dev)!=audit.DEV_HASH:raise RuntimeError('Original dev IDs required before reading labels.')
    train,test,y=audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev)
    splits=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,y))
    with np.load(ROOT/f'artifacts/run_038/v6_no_id_features_seed{SEED}.npz',allow_pickle=False) as saved:baseline=saved['prob'].copy()
    raw,record,files=common.read_component('run_038',CONTROL,'mitra',dev,y,splits,baseline)
    evidence.update(files)
    lock='reports/run_038/requirements-mitra.lock.txt'
    common.require_committed([lock])
    if (ROOT/lock).read_text()!=base.environment_lock():raise RuntimeError('Cached raw-Mitra environment differs.')
    evidence[lock]=sha(ROOT/lock)
    # Context references only: no rerun, blending, ranking gate, or new result access.
    references={}
    for run,version in [('run_046','v81_Mitra_full_episode50'),('run_019','v44_seed_ensemble')]:
        name=f'reports/{run}/screen_results.csv';common.require_committed([name]);evidence[name]=sha(ROOT/name)
        row=pd.read_csv(ROOT/name).query('version == @version')
        if len(row)!=1:raise RuntimeError('Unique historical contextual reference required.')
        references[version]={key:float(row.iloc[0][key]) for key in ['accuracy','log_loss']}
    roster=pd.concat([train,test],ignore_index=True)
    x=iterate.build_features(roster,False)[0]
    if len(x.columns)!=30:raise RuntimeError('Original engineered feature schema changed.')
    passengers=train.PassengerId.iloc[dev].astype(str).to_numpy()
    return dict(x=x,dev=dev,y=y,splits=splits,raw=raw,raw_record=record,references=references,
        passenger_ids=passengers,evidence=evidence)


def scores(p,y,splits):
    p=np.asarray(p)
    if p.shape!=y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Complete aligned OOF required.')
    return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
        fold_accuracy=[float(accuracy_score(y[b],p[b]>=.5)) for _,b in splits],
        fold_log_loss=[float(log_loss(y[b],p[b],labels=[0,1])) for _,b in splits])


def continue_gate(candidate,raw,completed):
    valid=completed==[0,1,2] and all(len(m.get('fold_accuracy',[]))==3 and
        np.isfinite([m.get('accuracy',np.nan),m.get('log_loss',np.nan),*m.get('fold_accuracy',[])]).all() for m in [candidate,raw])
    checks=dict(all_three_folds_complete=bool(valid),accuracy_gain_at_least_0_001=bool(valid and candidate['accuracy']-raw['accuracy']>=.001-1e-12),
        logloss_not_worse=bool(valid and candidate['log_loss']<=raw['log_loss']+1e-12),
        two_folds_not_worse=bool(valid and sum(a>=b-1e-12 for a,b in zip(candidate['fold_accuracy'],raw['fold_accuracy']))>=2))
    return dict(passed=all(checks.values()),checks=checks,tolerance=1e-12,
        interpretation='Prospective resource continuation only; no A/S qualification or upload permission.',automatic_fullfit_or_upload=False)


def paired(p,raw,y,mask=None):
    mask=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if not mask.any():return dict(rows=0,accuracy=None,raw_accuracy=None,corrections=0,regressions=0)
    good=(p[mask]>=.5)==y[mask];old=(raw[mask]>=.5)==y[mask]
    return dict(rows=int(mask.sum()),accuracy=float(good.mean()),raw_accuracy=float(old.mean()),
        corrections=int((good&~old).sum()),regressions=int((~good&old).sum()))


def fit_fold(data,fold,out):
    a,b=data['splits'][fold];dev,y=data['dev'],data['y'];model=None;started=time.monotonic()
    path=out/f'{NAME}_seed{SEED+fold}_{audit.row_hash(dev[a])[:12]}.json'
    if path.exists():raise RuntimeError('Existing fit record; no resume/overwrite.')
    try:
        model=core.fit_missing_flags(data['x'].iloc[dev[a]],y[a],dev[a],SEED+fold,'mps')
        if len(model.support)!=len(a):raise RuntimeError('Full context required.')
        p=core.valid_probabilities(model.predict_proba(data['x'].iloc[dev[b]]),len(b))
        model.metadata.update(status='complete',fold=fold,outer_split_seed=SEED,prediction_rows=len(b),
            prediction_ids_sha256=audit.row_hash(dev[b]),training_passenger_ids_sha256=audit.string_hash(data['passenger_ids'][a]),
            prediction_passenger_ids_sha256=audit.string_hash(data['passenger_ids'][b]),
            probabilities_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
            seconds=time.monotonic()-started,resources=base.snapshot('mps'))
        write(path,model.metadata);return p[:,1].copy(),model.metadata
    except BaseException as error:
        write(path,dict(status='failed_inconclusive',fold=fold,error_type=type(error).__name__,error=str(error)));raise
    finally:
        if model is not None:model.release()
        model=None;base.release_memory()


def screen(session_file):
    out=ROOT/'artifacts'/RUN;report=ROOT/'reports'/RUN
    if out.exists() or report.exists():raise RuntimeError('run052 exists; no overwrite/resume.')
    check_quota(session_file)
    with deadline(900):
        contract,lock=freeze_contract();resources=require_probe(contract)  # Before any labels/new accuracy.
        data=load_inputs();contract['evidence_files'].update({**resources,**data['evidence']})
        common.require_committed([p for p in contract['evidence_files'] if p.startswith('reports/')])
        assert_frozen(contract);runtime()
        out.mkdir(parents=True);report.mkdir(parents=True)
        protocol=dict(contract,run=RUN,seed=SEED,folds=3,threshold=.5,development_rows=6954,
            candidate=NAME,configs={NAME:core.SETTINGS,CONTROL:dict(role='cached_matched_reference')},
            development_ids_sha256=audit.row_hash(data['dev']),passenger_ids_sha256=audit.string_hash(data['passenger_ids']),
            development_targets_sha256=audit.row_hash(data['y']),budget_seconds=900,
            disclosure='Matched representation experiment on reused development CV; no independent holdout, group comparison or causal claim.')
        write(out/'protocol.json',protocol);write(report/'protocol.json',protocol)
        (out/'environment.lock.txt').write_text(lock);(report/'environment.lock.txt').write_text(lock)
        for name in contract['sources']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        assignment=np.full(len(data['y']),-1,dtype=np.int8)
        for fold,(_,b) in enumerate(data['splits']):
            if np.any(assignment[b]!=-1):raise RuntimeError('Repeated outer query row.')
            assignment[b]=fold
        if (assignment<0).any():raise RuntimeError('Incomplete outer split coverage.')
        np.savez(out/'folds.npz',dev_ids=data['dev'],fold_id=assignment,passenger_ids=data['passenger_ids'].astype(str),seed=SEED)
        p=np.full(len(data['y']),np.nan);rows=[];started=time.monotonic()
        try:
            for fold,(_,b) in enumerate(data['splits']):
                assert_frozen(contract);base.check_deadline()
                prediction,record=fit_fold(data,fold,out);p[b]=prediction
                row=dict(fold=fold,accuracy=float(accuracy_score(data['y'][b],prediction>=.5)),
                    log_loss=float(log_loss(data['y'][b],prediction,labels=[0,1])),
                    paired=paired(prediction,data['raw'][b],data['y'][b]),schema=record['schema'],
                    seconds=record['seconds'],seed=SEED+fold)
                rows.append(row);write(out/f'fold{fold}.json',row);write(report/f'fold{fold}.json',row)
                np.savez(out/'partial_oof.npz',prob=p,dev_ids=data['dev'],fold_id=assignment)
                progress=dict(status='incomplete',completed_folds=rows,pooled_available=False)
                write(out/'progress.json',progress);write(report/'progress.json',progress)
                print(json.dumps(row),flush=True);assert_frozen(contract)
            result=scores(p,data['y'],data['splits']);raw=scores(data['raw'],data['y'],data['splits'])
            gate=continue_gate(result,raw,[row['fold'] for row in rows])
            missing=core.missing_flags(data['x'].iloc[data['dev']]).any(axis=1)
            assessment=dict(status='complete',candidate=NAME,metrics=result,raw_metrics=raw,
                paired_vs_raw=paired(p,data['raw'],data['y']),missing_subset=paired(p,data['raw'],data['y'],missing),
                complete_subset=paired(p,data['raw'],data['y'],~missing),context_references=data['references'],
                continuation=gate,automatic_eligibility=False,S_supported=False,seconds=time.monotonic()-started)
            np.savez(out/f'{NAME}_seed{SEED}.npz',prob=p,fold_accuracy=result['fold_accuracy'],
                dev_ids=data['dev'],fold_id=assignment,passenger_ids=data['passenger_ids'].astype(str))
            assessment['oof_sha256']=sha(out/f'{NAME}_seed{SEED}.npz')
            assessment['folds_sha256']=sha(out/'folds.npz')
            pd.DataFrame([dict(version=name,role=role,accuracy=m['accuracy'],log_loss=m['log_loss'],
                fold_std=float(np.std(m['fold_accuracy']))) for name,role,m in [(NAME,'candidate',result),(CONTROL,'cached_reference',raw)]]).to_csv(out/'screen_results.csv',index=False)
            write(out/'assessment.json',assessment)
            write(out/'selection.json',dict(version=NAME,screen_accuracy=result['accuracy'],continuation_passed=gate['passed'],
                stage='pending_separate_fullfit_decision' if gate['passed'] else 'archived_no_fullfit',automatic_eligibility=False))
            write(out/'progress.json',dict(status='complete',completed_folds=rows,pooled_available=True))
            assert_frozen(contract);base.check_deadline()
            for name in ['assessment.json','selection.json','screen_results.csv','progress.json']:shutil.copy2(out/name,report/name)
            (out/'partial_oof.npz').unlink();print(json.dumps(assessment),flush=True);return assessment
        except BaseException as error:
            for folder in [out,report]:
                for name in ['assessment.json','selection.json','screen_results.csv',f'{NAME}_seed{SEED}.npz']:(folder/name).unlink(missing_ok=True)
                write(folder/'screen_failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),completed_folds=rows))
                write(folder/'progress.json',dict(status='failed_inconclusive',completed_folds=rows,pooled_available=False))
            raise
        finally:
            base.release_memory()
            for path in out.glob(f'{NAME}_seed*.json'):shutil.copy2(path,report/path.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    screen(parser.parse_args().session_file)
