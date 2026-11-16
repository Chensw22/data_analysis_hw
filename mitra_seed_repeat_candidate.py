"""Conditional fixed +100 training-seed repeat and its sole equal-mean candidate.

No export or upload. The outer split stays20261112; only the training program's
seed changes. A committed activation after v85 completion is mandatory.
"""
import argparse
from datetime import datetime,timezone
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

import prior_adapt_mean_screen as cached
import mitra_full_episode_candidate as original
import mitra_full_episode as core

ROOT=Path(__file__).resolve().parent
common,audit,base=cached.common,cached.audit,core.base
RUN='run_051'
NAME='v88_Mitra_full_episode50_two_seed_mean'
SOLO='v87_Mitra_full_episode50_seed100'
OLD='v81_Mitra_full_episode50'
OUTER_SEED=20261112
TRAINING_SEEDS=[20261212,20261213,20261214]
REVIEW='reports/top5_seed_repeat_review.json'
REVIEW_SHA='2f7e1052e2678f9b08ad6d9b6434af9fa96ffae7d0e384ca3bd3b5bf8159f8e5'
ACTIVATION='reports/online_iteration_20261116/seed_repeat_activation.json'
V85_SUMMARY='reports/group_episode_20261116/summary.json'
V85_NAME='v85_Mitra_group_episode50'
V85_ATTEMPTS='reports/proposal_nov16_group_episode/online_attempts'
SETTINGS=dict(outer_split_seed=OUTER_SEED,outer_folds=3,development_rows=6954,
    new_fit_seeds=TRAINING_SEEDS,seed_offset=100,final_new_fit_seed=20261212,
    mean_weights=[.5,.5],threshold=.5,primary_candidate=NAME,diagnostic_solo=SOLO,
    screen_budget_seconds=14400,final_export_budget_seconds=7200,
    mps_memory_fraction=.65,threads=4,query_chunk=2318,weekly_reserve_percent=30,
    solo_upload_allowed=False,automatic_fullfit_or_upload=False)
OWN_SOURCES=['mitra_seed_repeat_candidate.py','test_mitra_seed_repeat_candidate.py',
    'prior_adapt_mean_screen.py','test_prior_adapt_mean_screen.py',
    'three_family_mean_screen.py','test_three_family_mean_screen.py',
    'audit_top5_references.py','test_audit_top5_references.py','quota_guard.py',
    'reports/official_data_verification.json',REVIEW,ACTIVATION]
sha=base.sha256


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);audit.json_write(path,value)


def relative_file(value):
    path=Path(value)
    if (path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file()
            or not (ROOT/path).resolve().is_relative_to(ROOT.resolve())):
        raise RuntimeError('Existing repository-relative evidence file required.')
    return path.as_posix()


def verify_review():
    if sha(ROOT/REVIEW)!=REVIEW_SHA:
        raise RuntimeError('The entire prospectively reviewed seed-repeat plan is fixed.')
    review=json.loads((ROOT/REVIEW).read_text())
    design=review['fixed_design'];decision=review['root_decision']
    if (design['new_training_seeds']!=TRAINING_SEEDS
            or design['old_training_seeds']!=[20261112,20261113,20261114]
            or design['final_fullfit_new_seed']!=20261212
            or decision['candidate_names']!={'diagnostic_solo':SOLO,'primary_mean':NAME}
            or decision['reserved_run']!=RUN
            or review['cost']['fixed_screen_budget_seconds']!=14400
            or review['cost']['fixed_final_export_budget_seconds']!=7200):
        raise RuntimeError('Fixed seed, identity or resource declaration differs.')
    for relative,digest in review['provenance_sha256'].items():
        if sha(ROOT/relative_file(relative))!=digest:
            raise RuntimeError('Reviewed original component evidence changed: '+relative)
    return review


def utc_now():
    return datetime.now(timezone.utc)


def aware_timestamp(value):
    try:
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:raise ValueError('Naive timestamp')
        return parsed.astimezone(timezone.utc)
    except (AttributeError,ValueError,TypeError,OverflowError) as error:
        raise RuntimeError('Valid timezone-aware evidence timestamp required.') from error


def activation_evidence():
    """Read final v85 outcome only after root explicitly creates this activation."""
    activation=json.loads((ROOT/ACTIVATION).read_text())
    if (activation.get('status')!='activate_fixed_fallback' or activation.get('run')!=RUN
            or activation.get('primary_candidate')!=NAME or activation.get('diagnostic_solo')!=SOLO
            or activation.get('review_sha256')!=REVIEW_SHA
            or activation.get('v85_process_complete') is not True
            or activation.get('top5_goal_met') is not False
            or activation.get('mps_exclusive_confirmed') is not True
            or not isinstance(activation.get('rationale'),str) or not activation['rationale'].strip()):
        raise RuntimeError('Explicit root activation after v85 completion and unmet goal is required.')
    files=activation.get('evidence_files')
    if not isinstance(files,dict) or V85_SUMMARY not in files:
        raise RuntimeError('Activation must bind completed v85 and ranking evidence.')
    for relative,digest in files.items():
        if not relative.startswith('reports/') or sha(ROOT/relative_file(relative))!=digest:
            raise RuntimeError('Activation evidence changed or is not a public report.')
    common.require_committed([ACTIVATION,*files])
    summary=json.loads((ROOT/V85_SUMMARY).read_text())
    if (summary.get('status')!='complete' or summary.get('candidate')!=V85_NAME
            or summary.get('kind')!='group' or [r.get('fold') for r in summary.get('folds',[])]!=[0,1,2]):
        raise RuntimeError('All fixed v85 group folds must finish before the fallback.')
    attempts=list((ROOT/V85_ATTEMPTS).glob('*.json'))
    disposition=activation.get('v85_online_disposition')
    if disposition=='online_complete':
        if not attempts:raise RuntimeError('Actual completed v85 online attempt required.')
        completed_times=[]
        for path in attempts:
            relative=str(path.relative_to(ROOT))
            record=json.loads(path.read_text())
            if (relative not in files or record.get('version')!=V85_NAME
                    or record.get('evaluation',{}).get('Status')!='COMPLETE'):
                raise RuntimeError('Every actual v85 upload must be resolved COMPLETE and bound.')
            completed_times.append(aware_timestamp(record.get('last_checked_at_utc')))
            snapshot=record.get('leaderboard_snapshot',{})
            if type(snapshot.get('rank_fraction')) in (int,float) and snapshot['rank_fraction']<=.05:
                raise RuntimeError('v85 already reached top5; do not launch the fallback.')
        completed_at=max(completed_times)
    elif disposition=='not_submitted':
        path=activation.get('v85_disposition_decision_path')
        if attempts or path not in files:
            raise RuntimeError('No-attempt disposition needs a bound explicit archive/failure decision.')
        decision=json.loads((ROOT/path).read_text())
        if (decision.get('candidate')!=V85_NAME
                or decision.get('status') not in ['archived_no_upload','export_failed_no_upload']
                or decision.get('v85_process_complete') is not True
                or decision.get('submission_attempted') is not False
                or not isinstance(decision.get('rationale'),str) or not decision['rationale'].strip()):
            raise RuntimeError('Pending v85 work is not an explicit completed no-upload decision.')
        completed_at=aware_timestamp(decision.get('completed_at_utc'))
    else:raise RuntimeError('Declare the completed v85 online disposition.')
    rank_path=activation.get('ranking_evidence_path')
    if rank_path not in files:raise RuntimeError('Committed ranking evidence must be bound.')
    rank_record=json.loads((ROOT/rank_path).read_text())
    rank_record=rank_record.get('leaderboard_snapshot',rank_record)
    rank=rank_record.get('account_best_rank');teams=rank_record.get('team_count',rank_record.get('teams'))
    ratio=rank_record.get('rank_fraction')
    if (type(rank) is not int or type(teams) is not int or not 1<=rank<=teams
            or type(ratio) not in (float,int) or not np.isfinite(ratio)
            or abs(ratio-rank/teams)>1e-12 or ratio<=.05
            or not rank_record.get('observed_at_utc')):
        raise RuntimeError('Valid official account ranking must show the top5 goal remains unmet.')
    observed_at=aware_timestamp(rank_record.get('observed_at_utc'))
    age=(utc_now()-observed_at).total_seconds()
    if age>900 or age < -60:
        raise RuntimeError('Ranking snapshot must be at most900seconds old and at most60seconds in the future.')
    if observed_at<completed_at:
        raise RuntimeError('Ranking snapshot predates completed v85 disposition.')
    return activation,dict(files)


def check_quota(session_file):
    # Do not persist the path, usage percentages, or account metadata anywhere.
    from quota_guard import latest_weekly,assess
    try:allowed=assess(latest_weekly(session_file),reserve=30)['can_start_more_work']
    except (OSError,ValueError,KeyError,TypeError):allowed=False
    if not allowed:raise RuntimeError('Fresh weekly reserve guard did not permit launching this fallback.')


def verify_runtime():
    _,lock=original.verify_reference()  # Original checkpoint/environment/resource proof, no configure/fit.
    protocol=json.loads((ROOT/'artifacts/run_046/protocol.json').read_text())
    audit.check_historical_sources(protocol)
    settings=protocol['full_episode_policy']['settings']
    if (any(settings.get(k)!=v for k,v in core.SETTINGS.items())
            or settings['mps_memory_fraction']!=.65 or settings['query_chunk']!=2318
            or settings['threads']!=4 or settings['device']!='mps'
            or protocol['full_episode_policy']['environment_sha256']!=hashlib.sha256(lock.encode()).hexdigest()):
        raise RuntimeError('Original v81 core/runtime environment changed.')
    return protocol,lock


def fixed_splits(dev,y):
    if len(dev)!=6954 or audit.row_hash(dev)!=audit.DEV_HASH:
        raise RuntimeError('Original6954development membership required.')
    return list(StratifiedKFold(3,shuffle=True,random_state=OUTER_SEED).split(dev,y))


def load_cached_inputs():
    evidence={}
    verified=json.loads((ROOT/'reports/official_data_verification.json').read_text())
    if verified.get('competition')!='spaceship-titanic':raise RuntimeError('Unexpected official dataset identity.')
    for name in ['train.csv','test.csv']:
        path=ROOT/'data/raw'/name
        if sha(path)!=verified['files'][name]['official_sha256']:raise RuntimeError('Official data changed.')
        evidence[str(path.relative_to(ROOT))]=sha(path)
    split_path=ROOT/'artifacts/run_002/split_indices.npz'
    with np.load(split_path,allow_pickle=False) as saved:dev=np.sort(saved['development']).astype(np.int64)
    evidence[str(split_path.relative_to(ROOT))]=sha(split_path)
    if len(dev)!=6954 or audit.row_hash(dev)!=audit.DEV_HASH:
        raise RuntimeError('Original development membership required before reading any target rows.')
    train,test,y=audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev)
    splits=fixed_splits(dev,y)
    with np.load(ROOT/f'artifacts/run_038/v6_no_id_features_seed{OUTER_SEED}.npz',allow_pickle=False) as saved:
        baseline=saved['prob'].copy()
    raw,_,files=common.read_component('run_038','v71_Mitra2_zeroshot','mitra',dev,y,splits,baseline)
    evidence.update(files)
    old,old_record,files=cached.read_adapt(dev,y,splits,baseline,raw);evidence.update(files)
    cached.verified_report('run_038','requirements-mitra.lock.txt',evidence)
    lock=cached.verified_report('run_046','full_episode_environment.lock.txt',evidence)
    if lock.read_text()!=base.environment_lock():raise RuntimeError('Current and cached v81 environments differ.')
    old_logs=[]
    for fold,(a,b) in enumerate(splits):
        path=ROOT/f'artifacts/run_046/{OLD}_seed{OUTER_SEED+fold}_{audit.row_hash(dev[a])[:12]}.json'
        record=json.loads(path.read_text())
        if record.get('recipe')!=core.SETTINGS:raise RuntimeError('Cached training is not the unchanged original full50 core.')
        old_logs.append(dict(fold=fold,fit_seed=OUTER_SEED+fold,selected_steps=record['selected_steps'],
            inner_train_ids_sha256=record['inner_train_ids_sha256'],inner_validation_ids_sha256=record['inner_validation_ids_sha256']))
    roster=pd.concat([train,test],ignore_index=True)
    features,_=original.iterate.build_features(roster,False)
    if len(features.columns)!=30:raise RuntimeError('Original feature matrix changed.')
    return dict(features=features,dev=dev,y=y,splits=splits,old=old,old_record=old_record,
        old_logs=old_logs,evidence=evidence)


def make_fold_assignment(dev,splits):
    assignment=np.full(len(dev),-1,dtype=np.int8);covered=np.zeros(len(dev),dtype=int)
    if len(splits)!=3:raise RuntimeError('Exactly three outer folds required.')
    for fold,(a,b) in enumerate(splits):
        if (not len(a) or not len(b) or np.intersect1d(a,b).size
                or not np.array_equal(np.sort(np.concatenate([a,b])),np.arange(len(dev)))):
            raise RuntimeError('Outer fold is not a full disjoint development partition.')
        assignment[b]=fold;covered[b]+=1
    if not np.all(covered==1):raise RuntimeError('Each development row must be predicted exactly once.')
    return assignment


def metrics(p,y,splits):
    p=np.asarray(p,dtype=float)
    if p.shape!=y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise RuntimeError('Complete finite OOF required before pooled metrics.')
    return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
        fold_accuracy=[float(accuracy_score(y[b],p[b]>=.5)) for _,b in splits],
        fold_log_loss=[float(log_loss(y[b],p[b],labels=[0,1])) for _,b in splits])


def paired(p,reference,y,splits):
    def one(index):
        q,r,t=p[index],reference[index],y[index];correct=(q>=.5)==t;old_correct=(r>=.5)==t
        e1,e0=(~correct).astype(float),(~old_correct).astype(float)
        correlation=float(np.corrcoef(e1,e0)[0,1]) if np.std(e1)>0 and np.std(e0)>0 else None
        return dict(corrections=int(np.sum(correct&~old_correct)),regressions=int(np.sum(~correct&old_correct)),
            disagreements=int(np.sum(correct!=old_correct)),error_correlation=correlation,
            correlation_note='undefined for a constant error vector' if correlation is None else 'descriptive Pearson correlation of actual binary prediction errors',
            mean_absolute_probability_change=float(np.mean(np.abs(q-r))),
            maximum_absolute_probability_change=float(np.max(np.abs(q-r))),
            mean_signed_probability_change=float(np.mean(q-r)))
    return dict(overall=one(np.arange(len(y))),folds=[one(b) for _,b in splits])


def continuation(mean,old,completed_folds):
    finite=lambda values:len(values)==3 and bool(np.isfinite(values).all())
    whole=all(type(row.get(k)) in (int,float) and np.isfinite(row[k])
              for row in [mean,old] for k in ['accuracy','log_loss'])
    folds_ok=finite(mean.get('fold_accuracy',[])) and finite(old.get('fold_accuracy',[]))
    nondecreasing=sum(a>=b-1e-12 for a,b in zip(mean.get('fold_accuracy',[]),old.get('fold_accuracy',[]))) if folds_ok else 0
    checks=dict(all_three_folds_complete=completed_folds==[0,1,2] and whole and folds_ok,
        mean_accuracy_not_below_old=whole and mean['accuracy']>=old['accuracy']-1e-12,
        mean_logloss_not_above_old=whole and mean['log_loss']<=old['log_loss']+1e-12,
        at_least_two_folds_not_below_old=folds_ok and nondecreasing>=2)
    return dict(passed=all(checks.values()),checks=checks,nondecreasing_fold_count=nondecreasing,
        numerical_tolerance=1e-12,sole_primary_candidate=NAME,diagnostic_solo_upload_allowed=False,
        automatic_fullfit_or_upload=False,
        interpretation='Prospective resource/exploration heuristic only; no stable superiority, S-tier or across-split confirmation.')


def assert_frozen(protocol):
    for mapping in [protocol['sources'],protocol['evidence_files']]:
        for relative,digest in mapping.items():
            if sha(ROOT/relative)!=digest:raise RuntimeError('Frozen source/evidence changed: '+relative)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest()!=protocol['environment_sha256']:
        raise RuntimeError('Frozen environment changed.')
    base.verify_checkpoint(base.CHECKPOINT_DIR)


def fit_fold(inputs,fold,out):
    a,b=inputs['splits'][fold];dev,y=inputs['dev'],inputs['y'];seed=TRAINING_SEEDS[fold]
    path=out/f'{SOLO}_seed{seed}_{audit.row_hash(dev[a])[:12]}.json'
    if path.exists():raise RuntimeError('Fit log exists; no resume or overwrite.')
    model=None
    try:
        base.check_deadline()
        model=core.fit_full_episode(inputs['features'].iloc[dev[a]],y[a],dev[a],seed,
            device='mps',log_path=path,max_steps=50)
        p=np.asarray(model.predict_proba(inputs['features'].iloc[dev[b]]))
        if (p.shape!=(len(b),2) or not np.isfinite(p).all() or np.any((p<0)|(p>1))):
            raise RuntimeError('Incomplete fold probability matrix.')
        base.check_deadline()
        model.metadata.update(outer_split_seed=OUTER_SEED,outer_fold=fold,actual_training_seed=seed,
            prediction_ids_sha256=audit.row_hash(dev[b]),prediction_rows=len(b),
            query_chunk=2318,training_program='unchanged_original_v81_full50',diagnostic_component_only=True)
        cached.check_adaptation_log(model.metadata,dev[a],dev[b],y[a],seed)
        if model.metadata['recipe']!=core.SETTINGS:raise RuntimeError('Training core recipe changed.')
        write(path,model.metadata)
        row=dict(fold=fold,outer_split_seed=OUTER_SEED,actual_training_seed=seed,
            train_ids_sha256=audit.row_hash(dev[a]),query_ids_sha256=audit.row_hash(dev[b]),
            training_rows=len(a),query_rows=len(b),selected_steps=model.metadata['selected_steps'],
            inner_train_ids_sha256=model.metadata['inner_train_ids_sha256'],
            inner_validation_ids_sha256=model.metadata['inner_validation_ids_sha256'])
        return p[:,1].copy(),row
    finally:
        if model is not None:model.release()
        model=None;base.release_memory()


def screen(session_file):
    out=ROOT/'artifacts'/RUN;report=ROOT/'reports'/RUN
    if out.exists() or report.exists():raise RuntimeError('Reserved run051 exists; no overwrite/resume.')
    review=verify_review();activation,activation_files=activation_evidence()
    common.require_committed(OWN_SOURCES)
    check_quota(session_file)  # Ephemeral input only; never serialize path or quota metadata.
    original_protocol,lock=verify_runtime();inputs=load_cached_inputs()
    sources=dict(original_protocol['sources'])
    sources.update({name:sha(ROOT/name) for name in OWN_SOURCES})
    evidence={**inputs['evidence'],**activation_files,**review['provenance_sha256']}
    committed=common.require_committed([*sources,*[p for p in evidence if p.startswith('reports/')]])
    for relative,digest in {**sources,**evidence}.items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Source/evidence changed before freezing.')
    assignment=make_fold_assignment(inputs['dev'],inputs['splits'])
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('Disable automatic MPS fallback.')
    if not base.torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
    base.torch.set_num_threads(4);base.torch.mps.set_per_process_memory_fraction(.65)
    out.mkdir(parents=True);report.mkdir(parents=True)
    protocol=dict(seed=OUTER_SEED,folds=3,threshold=.5,development_rows=6954,run=RUN,
        candidate=NAME,diagnostic_solo=SOLO,configs={OLD:dict(role='cached_reference'),
            SOLO:dict(role='diagnostic_only',core_recipe=core.SETTINGS,fit_seeds=TRAINING_SEEDS),
            NAME:dict(role='sole_primary',components=[OLD,SOLO],weights=[.5,.5])},
        settings=SETTINGS,source_commit=committed,sources=sources,evidence_files=evidence,
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        development_ids_sha256=audit.row_hash(inputs['dev']),development_targets_sha256=audit.row_hash(inputs['y']),
        old_component=inputs['old_record'],old_inner_selections=inputs['old_logs'],
        automatic_eligibility=False,online_submission_authorized=False,
        disclosure=review['fixed_design']['scoring_disclosure'],
        seed_disclosure=review['fixed_design']['randomness_disclosure'],
        cache_alignment='Original frozen CV source/dev IDs/fold metrics/complete training logs and bitwise v6/raw controls. Legacy old NPZ lacks IDs; every new NPZ saves them.',
        continuation_rule=review['fixed_continue_to_fullfit_heuristic'])
    write(out/'protocol.json',protocol);write(report/'protocol.json',protocol)
    (out/'environment.lock.txt').write_text(lock);(report/'environment.lock.txt').write_text(lock)
    np.savez(out/'folds.npz',dev_ids=inputs['dev'],fold_id=assignment,
        actual_training_seeds=np.asarray(TRAINING_SEEDS,dtype=np.int64),outer_split_seed=OUTER_SEED)
    for relative in {**sources,**evidence}:
        if relative.startswith('data/raw/') or relative.endswith('split_indices.npz'):continue
        target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    started=time.monotonic();base.DEADLINE=started+14400
    def timeout(*_):raise TimeoutError('Fixed14400second seed-repeat screen budget exceeded.')
    old_handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,14400)
    p=np.full(len(inputs['y']),np.nan);covered=np.zeros(len(p),dtype=bool);fold_rows=[]
    try:
        for fold,(a,b) in enumerate(inputs['splits']):
            assert_frozen(protocol);base.check_deadline()
            predicted,row=fit_fold(inputs,fold,out);p[b]=predicted;covered[b]=True
            mean_fold=cached.fixed_mean(inputs['old'][b],predicted)
            row.update(diagnostic_solo_accuracy=float(accuracy_score(inputs['y'][b],predicted>=.5)),
                diagnostic_solo_log_loss=float(log_loss(inputs['y'][b],predicted,labels=[0,1])),
                fixed_mean_accuracy=float(accuracy_score(inputs['y'][b],mean_fold>=.5)),
                fixed_mean_log_loss=float(log_loss(inputs['y'][b],mean_fold,labels=[0,1])),
                cached_old_accuracy=float(accuracy_score(inputs['y'][b],inputs['old'][b]>=.5)))
            fold_rows.append(row);write(out/f'fold{fold}.json',row);write(report/f'fold{fold}.json',row)
            np.savez(out/'partial_solo_oof.npz',prob=p,covered=covered,dev_ids=inputs['dev'],fold_id=assignment)
            progress=dict(status='incomplete',completed_folds=fold_rows,pooled_mean_available=False)
            write(out/'progress.json',progress);write(report/'progress.json',progress)
            print(json.dumps(row),flush=True);assert_frozen(protocol)
        assert_frozen(protocol)
        if not np.all(covered) or len(fold_rows)!=3 or not np.isfinite(p).all():
            raise RuntimeError('All three new folds required before mean/result/selection.')
        mean=cached.fixed_mean(inputs['old'],p)
        old_metrics=metrics(inputs['old'],inputs['y'],inputs['splits'])
        solo_metrics=metrics(p,inputs['y'],inputs['splits']);mean_metrics=metrics(mean,inputs['y'],inputs['splits'])
        decision=continuation(mean_metrics,old_metrics,[r['fold'] for r in fold_rows])
        seconds=time.monotonic()-started
        rows=[dict(version=name,role=role,accuracy=m['accuracy'],log_loss=m['log_loss'],
                   fold_std=float(np.std(m['fold_accuracy'])),seconds=0. if name==OLD else seconds)
              for name,role,m in [(OLD,'cached_reference',old_metrics),(SOLO,'diagnostic_only',solo_metrics),(NAME,'sole_primary',mean_metrics)]]
        assessment=dict(status='complete',primary_candidate=NAME,diagnostic_solo=SOLO,
            metrics={OLD:old_metrics,SOLO:solo_metrics,NAME:mean_metrics},new_folds=fold_rows,
            old_inner_selections=inputs['old_logs'],paired_vs_cached_old={
                SOLO:paired(p,inputs['old'],inputs['y'],inputs['splits']),
                NAME:paired(mean,inputs['old'],inputs['y'],inputs['splits'])},
            continuation=decision,automatic_eligibility=False,S_supported=False,
            sole_possible_future_export=NAME,solo_upload_allowed=False,
            interpretation='Single training-program repeat on reused fixed partitions; within1pp remains practical parity. Not stable superiority or independent validation.')
        for name,values,m in [(OLD,inputs['old'],old_metrics),(SOLO,p,solo_metrics),(NAME,mean,mean_metrics)]:
            np.savez(out/f'{name}_seed{OUTER_SEED}.npz',prob=values,fold_accuracy=m['fold_accuracy'],seconds=seconds,
                dev_ids=inputs['dev'],fold_id=assignment,actual_training_seeds=TRAINING_SEEDS if name!=OLD else np.arange(20261112,20261115))
        selection=dict(version=NAME,components=[OLD,SOLO],weights=[.5,.5],threshold=.5,
            screen_accuracy=mean_metrics['accuracy'],continuation_passed=decision['passed'],
            stage='screen_only_pending_separate_decision' if decision['passed'] else 'archived_no_fullfit',
            diagnostic_solo_upload_allowed=False,automatic_eligibility=False)
        pd.DataFrame(rows).to_csv(out/'screen_results.csv',index=False)
        write(out/'assessment.json',assessment);write(out/'selection.json',selection)
        progress=dict(status='complete',completed_folds=fold_rows,pooled_mean_available=True,
            primary_candidate=NAME,continuation_passed=decision['passed'])
        write(out/'progress.json',progress)
        assert_frozen(protocol);base.check_deadline()
        for name in ['screen_results.csv','assessment.json','selection.json','progress.json']:
            shutil.copy2(out/name,report/name)
        (out/'partial_solo_oof.npz').unlink()
        print(json.dumps(dict(primary_candidate=NAME,metrics=assessment['metrics'],continuation=decision)),flush=True)
        return assessment
    except BaseException as error:
        for folder in [out,report]:
            for name in ['screen_results.csv','assessment.json','selection.json']:(folder/name).unlink(missing_ok=True)
            for name in [OLD,SOLO,NAME]:(folder/f'{name}_seed{OUTER_SEED}.npz').unlink(missing_ok=True)
        failure=dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),
            completed_folds=fold_rows,pooled_mean_available=False,seconds=time.monotonic()-started)
        write(out/'screen_failure.json',failure);write(report/'screen_failure.json',failure)
        write(out/'progress.json',failure);write(report/'progress.json',failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler);base.DEADLINE=None
        base.release_memory()
        for path in out.glob(f'{SOLO}_seed*.json'):shutil.copy2(path,report/path.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-file',type=Path,required=True)
    screen(parser.parse_args().session_file)
