"""Prepare one stability-qualified top-five proposal; never upload to Kaggle.

The reused-development A decision authorizes final full-label fitting, not a
claim of statistical superiority. Historical gates and runs remain untouched.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPECS = {
    'mitra_mirror_ensemble': ('v74_Mitra_mirror4', 'stability', 3600),
    'mitra_lastblock_candidate': ('v75_Mitra_lastblock50', 'export', 7200),
    'mitra_lastblock100_candidate': ('v76_Mitra_lastblock100', 'export', 10800),
    'mitra_context_candidate': ('v77_Mitra_context20', 'export', 3600),
}
POLICY = 'reports/top5_validation_review.json'
AUTH = 'reports/top5_authorization.json'
EXPECTED_TRAIN_ROWS, EXPECTED_TEST_ROWS = 8693, 4277
RELOAD_ATOL = 2e-6
GROUP_CANDIDATES = {
    'mitra_mirror_ensemble': ('mirror', 'run_040'),
    'mitra_context_candidate': ('context', 'run_043'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('Run and proposal must be single directory names.')
    return value


def eligibility_evidence(run, name):
    """Read metadata only. No train data, labels, or fitted model is accessed."""
    run_out = ROOT/'artifacts'/safe_name(run)
    eligibility_path = run_out/'stability/eligibility.json'
    eligibility = read(eligibility_path)
    if (eligibility.get('status') != 'complete' or eligibility.get('candidate') != name
            or eligibility.get('A_eligible') is not True):
        raise RuntimeError('Complete, matching, A-eligible validation is required.')
    if not eligibility.get('A_checks') or not all(value is True for value in eligibility['A_checks'].values()):
        raise RuntimeError('Every frozen A condition must pass.')
    paths = {'original_screen_protocol_sha256':run_out/'protocol.json',
             'validation_policy_sha256':ROOT/POLICY, 'authorization_sha256':ROOT/AUTH}
    stability = read(run_out/'stability/protocol.json')
    if stability.get('candidate') != name:
        raise RuntimeError('Stability protocol candidate differs.')
    for key, path in paths.items():
        if eligibility.get(key) != sha(path) or stability.get(key) != sha(path):
            raise RuntimeError(f'Eligibility/stability binding changed: {key}')
    authorization = read(ROOT/AUTH)
    if (authorization.get('eligibility_policy') != POLICY
            or authorization.get('eligibility_policy_sha256') != sha(ROOT/POLICY)
            or authorization.get('no_duplicate_or_blind_retry_upload') is not True):
        raise RuntimeError('Current authorization does not bind the frozen policy.')
    source = read(run_out/'protocol.json')
    required = [f'{module}.py' for module, spec in SPECS.items() if spec[0] == name]
    required += ['top5_validate.py', 'audit_top5_references.py', POLICY, AUTH]
    for relative in required:
        if source.get('sources', {}).get(relative) != sha(ROOT/relative):
            raise RuntimeError(f'Required source not frozen/matching: {relative}')
    if stability.get('own_source_sha256') != source['sources']['top5_validate.py']:
        raise RuntimeError('Stability implementation differs from the screen freeze.')
    return dict(eligibility=eligibility, authorization=authorization, source_protocol=source,
        validation_track='random', validation_sources={},
        evidence_files={'source_protocol.json':str(run_out/'protocol.json'),
            'eligibility.json':str(eligibility_path),
            'stability_protocol.json':str(run_out/'stability/protocol.json'),
            'validation_policy.json':str(ROOT/POLICY),'source_authorization.json':str(ROOT/AUTH)},
        eligibility_sha256=sha(eligibility_path), stability_protocol_sha256=sha(run_out/'stability/protocol.json'),
        **{key:sha(path) for key,path in paths.items()})


def group_eligibility_evidence(module_name, run, name):
    """Metadata/hash gate only: no labels, model, or eligibility-writing helper."""
    if module_name not in GROUP_CANDIDATES or GROUP_CANDIDATES[module_name][1] != run:
        raise RuntimeError('Group validation only covers the fixed module/run queue.')
    key = GROUP_CANDIDATES[module_name][0]
    group = importlib.import_module('top5_group_evaluate')
    expected = group.CANDIDATES[key]
    if expected != dict(version=name,module=module_name,run=run):
        raise RuntimeError('Group candidate identity differs.')
    record = group.assert_phase('mitra')  # Source/history/policy and exact current environment.
    summaries = group.complete_queue(record)
    components = {('refs','mitra'),('refs','v44'),('candidates','mirror'),('candidates','context')}
    if set(summaries) != components or group.SEEDS != [20261112,20261115,20261116]:
        raise RuntimeError('Every fixed component and group split is required.')
    phase, path = group.PHASE, group.PHASE/'eligibility.json'
    if sha(path) != (phase/'eligibility.sha256').read_text().strip():
        raise RuntimeError('Group eligibility hash changed.')
    overall = read(path)
    if (overall.get('status')!='complete' or overall.get('fixed_queue_complete') is not True
            or set(overall.get('candidates',{}))!={'mirror','context'}
            or overall.get('phase_sha256')!=sha(phase/'protocol.json')
            or overall.get('policy_sha256')!=sha(ROOT/group.POLICY)
            or overall.get('authorization_sha256')!=sha(ROOT/group.AUTH)
            or set(overall.get('component_summary_sha256',{}))!={f'{kind}/{component}' for kind,component in components}):
        raise RuntimeError('Complete group queue/policy/authorization binding required.')
    files = {'source_protocol.json':str(ROOT/'artifacts'/run/'protocol.json'),
        'eligibility.json':str(path),'group_phase_protocol.json':str(phase/'protocol.json'),
        'validation_policy.json':str(ROOT/group.POLICY),'source_authorization.json':str(ROOT/group.AUTH)}
    for (kind,component), summary in summaries.items():
        summary_path=phase/kind/component/'summary.json'
        if (sha(summary_path)!=overall['component_summary_sha256'][f'{kind}/{component}']
                or read(summary_path)!=summary or [row['seed'] for row in summary['results']]!=group.SEEDS):
            raise RuntimeError('Completed group summary changed or split queue is incomplete.')
        files[f'group_evidence/{kind}/{component}/summary.json']=str(summary_path)
        for row in summary['results']:
            if sha(ROOT/row['oof_path'])!=row['oof_sha256']:
                raise RuntimeError('Completed group OOF changed.')
            if not np.isfinite(row['accuracy']) or not 0<=row['accuracy']<=1:
                raise RuntimeError('Invalid completed-group accuracy.')
    # Recompute both decisions from bound metrics, even if only one will export.
    for component,spec in group.CANDIDATES.items():
        decision=overall['candidates'][component]
        if decision.get('candidate')!=spec['version'] or [r['seed'] for r in decision['results']]!=group.SEEDS:
            raise RuntimeError('Group decision candidate/splits differ.')
        screen_path=ROOT/'artifacts'/spec['run']/'screen_results.csv'
        if record['historical_files'].get(str(screen_path.relative_to(ROOT)))!=sha(screen_path):
            raise RuntimeError('Original screen metric was not phase-bound.')
        screen_table=pd.read_csv(screen_path)
        screen_rows=screen_table.loc[screen_table.version.eq(spec['version']),'accuracy']
        frozen_screen=record['screen_evidence'][component]
        if (len(screen_rows)!=1 or abs(float(screen_rows.iloc[0])-decision['original_screen_accuracy'])>1e-12
                or frozen_screen['version']!=spec['version']
                or abs(frozen_screen['accuracy']-decision['original_screen_accuracy'])>1e-12):
            raise RuntimeError('Group decision original screen metric differs.')
        for index,row in enumerate(decision['results']):
            accuracy=summaries[('candidates',component)]['results'][index]['accuracy']
            if not np.isfinite(row['accuracy']) or abs(row['accuracy']-accuracy)>1e-12:
                raise RuntimeError('Group decision accuracy differs from bound summary.')
            for ref in ['mitra','v44']:
                reference=summaries[('refs',ref)]['results'][index]['accuracy']
                comparison=row[f'vs_{ref}']
                values=[comparison['reference_accuracy'],comparison['gain']]
                if (not np.isfinite(values).all() or abs(values[0]-reference)>1e-12
                        or abs(values[1]-(accuracy-reference))>1e-12):
                    raise RuntimeError('Group decision paired gain differs from bound summaries.')
        recomputed=group.make_decision(decision['original_screen_accuracy'],
            decision['results'],decision['paired_cluster_sensitivity'])
        for field in ['A_eligible','S_supported','A_checks','S_checks']:
            if decision.get(field)!=recomputed[field]:
                raise RuntimeError(f'Group numerical eligibility differs: {field}')
    selected=overall['candidates'][key]
    if (selected.get('A_eligible') is not True or not selected.get('A_checks')
            or not all(value is True for value in selected['A_checks'].values())):
        raise RuntimeError('Candidate is not A-eligible under the group-primary policy.')
    source_path=ROOT/'artifacts'/run/'protocol.json'
    if record['historical_files'].get(str(source_path.relative_to(ROOT)))!=sha(source_path):
        raise RuntimeError('Candidate screen protocol was not phase-bound.')
    required=['top5_group_evaluate.py','test_top5_group_evaluate.py',group.POLICY,group.AUTH]
    for relative in required:
        if record['sources'].get(relative)!=sha(ROOT/relative):
            raise RuntimeError('Group validation source was not phase-frozen.')
    authorization=read(ROOT/group.AUTH)
    if (authorization.get('no_duplicate_or_blind_retry_upload') is not True
            or authorization.get('all_fixed_queue_must_complete_before_submission') is not True):
        raise RuntimeError('Group upload authorization lacks complete-queue/no-retry scope.')
    previous=ROOT/'artifacts'/run/'stability/eligibility.json'
    legacy = None if not previous.exists() else dict(path=str(previous.relative_to(ROOT)),
        sha256=sha(previous),A_eligible=read(previous).get('A_eligible'),status=read(previous).get('status'))
    return dict(validation_track='group',eligibility={**selected,'screen_accuracy':selected['original_screen_accuracy']},
        authorization=authorization,source_protocol=read(source_path),validation_sources=record['sources'],
        evidence_files=files,eligibility_sha256=sha(path),stability_protocol_sha256=sha(phase/'protocol.json'),
        original_screen_protocol_sha256=sha(source_path),validation_policy_sha256=sha(ROOT/group.POLICY),
        authorization_sha256=sha(ROOT/group.AUTH),component_summary_sha256=overall['component_summary_sha256'],
        prior_random_eligibility=legacy,disclosure=authorization['revision_disclosure'])


def validation_evidence(module_name,run,name,validation):
    if validation=='random':
        return eligibility_evidence(run,name)
    if validation=='group':
        return group_eligibility_evidence(module_name,run,name)
    raise ValueError('Validation must be random or group.')


def duplicate_attempts(file_sha256):
    # Fail closed on unreadable attempt metadata, including nested batch reports.
    return [str(path.relative_to(ROOT)) for path in sorted((ROOT/'reports').rglob('online_attempts/*.json'))
            if read(path).get('file_sha256') == file_sha256]


def probability_vector(value, rows):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (rows,) or not np.isfinite(value).all() or not ((value>=0)&(value<=1)).all():
        raise RuntimeError('Incomplete or invalid probabilities; no partial export.')
    return value


def check_reload(first, second):
    second = probability_vector(second, len(first))
    difference = float(np.max(np.abs(first-second)))
    if difference > RELOAD_ATOL or not np.array_equal(first>=.5, second>=.5):
        raise RuntimeError('Full reload inference differs in probability or threshold decisions.')
    return difference


def release(candidate, model):
    if model is not None and hasattr(model, 'release'):
        model.release()
    candidate.mitra.release_memory()


def validate_csv(path):
    from kaggle_workflow import validate_submission
    return validate_submission(path, ROOT/'data/raw/test.csv', expected_rows=EXPECTED_TEST_ROWS)


def prepare(module_name, run, proposal, validation='random'):
    if module_name not in SPECS:
        raise ValueError('Unsupported candidate module.')
    name, stage, budget = SPECS[module_name]
    run, proposal = safe_name(run), safe_name(proposal)
    out, report = ROOT/'artifacts'/proposal, ROOT/'reports'/proposal
    if out.exists() or report.exists():
        raise RuntimeError('Proposal exists; no overwrite, implicit resume, or duplicate retry.')
    out.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = restored = candidate = None
    previous_handler = None
    provenance = None
    try:
        evidence = validation_evidence(module_name,run,name,validation)  # Before full labels.
        candidate = importlib.import_module(module_name)
        if candidate.NAME != name:
            raise RuntimeError('Module candidate identity differs from the authorized recipe.')
        candidate.configure_runtime(out/'fit_logs', stage=stage)
        candidate.assert_frozen(ROOT/'artifacts'/run)
        def timeout(*_):
            raise TimeoutError('Complete export/reload budget exceeded; no partial authorization.')
        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, budget)
        own_sources = ['prepare_top5_submission.py', 'test_prepare_top5_submission.py', 'kaggle_workflow.py']
        sources = dict(evidence['source_protocol']['sources'])
        sources.update(evidence['validation_sources'])
        sources.update({relative:sha(ROOT/relative) for relative in own_sources})
        provenance = dict(candidate=name, module=module_name, source_run=run, proposal=proposal,
            validation_track=validation,evidence_paths=evidence['evidence_files'],
            component_summary_sha256=evidence.get('component_summary_sha256'),
            prior_random_eligibility=evidence.get('prior_random_eligibility'),
            validation_disclosure=evidence.get('disclosure','Original random-primary protocol.'),
            source_hashes=sources, sources_frozen_before_full_label_read=True,
            prepared_from_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            eligibility_sha256=evidence['eligibility_sha256'],
            stability_protocol_sha256=evidence['stability_protocol_sha256'],
            source_protocol_sha256=evidence['original_screen_protocol_sha256'],
            validation_policy_sha256=evidence['validation_policy_sha256'],
            authorization_sha256=evidence['authorization_sha256'],
            runtime_stage=stage, budget_seconds=budget, device='mps', reload_atol=RELOAD_ATOL,
            threshold=.5, interpretation='Reused-development A eligibility; no new holdout or S claim.')
        for relative in sources:
            target = out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative,target)
        for filename, source in evidence['evidence_files'].items():
            (out/filename).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,out/filename)
        environment = candidate.mitra.environment_lock()
        (out/'environment.lock.txt').write_text(environment)
        provenance['environment_sha256'] = sha(out/'environment.lock.txt')
        write(out/'preparation_protocol.json',provenance)
        import iterate
        train, test, matrices, _ = iterate.dataset()  # First full-label access, final fit only.
        if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
            raise RuntimeError('Final full-data shape changed.')
        a, b = np.arange(len(train)), np.arange(len(train),len(train)+len(test))
        write(out/'progress.json',dict(stage='full_fit_started',train_rows=len(a),query_rows=len(b)))
        first, model = candidate.fit(name,matrices,a,train.Transported.astype(int).to_numpy(),b,iterate.SEEDS[0])
        first = probability_vector(first,len(test))
        if len(model.labels) != len(train):
            raise RuntimeError('Serialized model lacks the complete training context.')
        joblib.dump(model,out/'model.joblib')
        write(out/'fit_metadata.json',model.metadata)
        release(candidate, model)
        model = None
        gc.collect()
        candidate.mitra.release_memory()
        write(out/'progress.json',dict(stage='full_reload_inference_started',query_rows=len(b)))
        restored = joblib.load(out/'model.joblib')
        restored.device = 'mps'  # MitraPredictor intentionally reloads as CPU by default.
        if module_name == 'mitra_mirror_ensemble':
            result = restored.predict_proba(matrices[False][0].iloc[b],prediction_ids=b,
                log_path=out/'reload_log.json',prediction_path=out/'reload_views.npz')
        else:
            result = restored.predict_proba(matrices[False][0].iloc[b])
        result = np.asarray(result)
        if result.shape != (len(test),2):
            raise RuntimeError('Reload must predict both classes for every test row.')
        max_difference = check_reload(first,result[:,1])
        write(out/'reload_metadata.json',restored.metadata)
        release(candidate,restored)
        restored = None
        gc.collect()
        candidate.mitra.release_memory()
        pending = out/'pending'
        pending.mkdir()
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=first>=.5)).to_csv(pending/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=first,
            reload_probability=result[:,1])).to_csv(pending/'test_probabilities.csv',index=False)
        check = validate_csv(pending/'submission.csv')
        attempts = duplicate_attempts(check['sha256'])
        if attempts:
            raise RuntimeError(f'Identical CSV already has an upload attempt: {attempts}')
        candidate.assert_frozen(ROOT/'artifacts'/run)
        if validation_evidence(module_name,run,name,validation) != evidence:
            raise RuntimeError('Eligibility evidence changed during preparation.')
        if any(sha(ROOT/path) != digest for path,digest in sources.items()):
            raise RuntimeError('Preparation source changed during execution.')
        if hashlib.sha256(candidate.mitra.environment_lock().encode()).hexdigest() != provenance['environment_sha256']:
            raise RuntimeError('Runtime environment changed during preparation.')
        selection = dict(version=name,components=[name],source_run=run,validation_track=validation,
            stage='top5_stability_qualified_final_fit',A_eligible=True,
            S_supported=evidence['eligibility'].get('S_supported',False),
            screen_accuracy=evidence['eligibility']['screen_accuracy'])
        bound = dict(status='bound_single_upload_authorization',competition='spaceship-titanic',
            proposal=proposal,version=name,validation_track=validation,
            scope='One upload of this exact complete proposal CSV only',
            file_sha256=check['sha256'], maximum_upload_attempts_for_this_file=1,
            source_authorization_sha256=evidence['authorization_sha256'],
            eligibility_sha256=evidence['eligibility_sha256'],
            validation_protocol_sha256=evidence['stability_protocol_sha256'],
            fixed_queue_component_summary_sha256=evidence.get('component_summary_sha256'),
            source_protocol_sha256=provenance['source_protocol_sha256'],
            validation_policy_sha256=provenance['validation_policy_sha256'],
            prepared_from_commit=provenance['prepared_from_commit'],
            model_sha256=sha(out/'model.joblib'),desktop_reminder_required=True,
            check_remaining_attempts_before_upload=True,no_duplicate_or_blind_retry_upload=True,
            duplicate_attempts_checked=True,upload_attempted=False)
        manifest = dict(**check,status='complete',validation_track=validation,selection=selection,provenance=provenance,
            fit_rows=len(train),threshold=.5,complete_test_predictions=True,
            export_reload_inference_verified=True,reload_rows=len(test),reload_device='mps',
            reload_max_absolute_difference=max_difference,reload_threshold_decisions_equal=True,
            model_sha256=bound['model_sha256'],upload_attempted=False,upload_eligible=True,
            legacy_gate_unchanged=True,seconds=time.monotonic()-started,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        for filename in ['submission.csv','test_probabilities.csv']:
            (pending/filename).replace(out/filename)
        pending.rmdir()
        for filename,value in [('selection.json',selection),('authorization.json',bound),('manifest.json',manifest)]:
            write(out/filename,value)
            write(report/filename,value)
        print(json.dumps(manifest),flush=True)
        return manifest
    except BaseException as error:
        # No complete manifest, bound upload authorization, or public CSV survives failure.
        for directory in [out,report]:
            for filename in ['manifest.json','authorization.json','submission.csv','selection.json']:
                (directory/filename).unlink(missing_ok=True)
            write(directory/'failure.json',dict(status='failed_incomplete_not_upload_eligible',
                candidate=name,module=module_name,source_run=run,proposal=proposal,
                error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started))
        raise
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous_handler)
        if candidate is not None:
            release(candidate,model)
            release(candidate,restored)
            candidate.mitra.DEADLINE = None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',choices=SPECS,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--proposal',required=True)
    parser.add_argument('--validation',choices=['random','group'],default='random')
    args=parser.parse_args()
    prepare(args.module,args.run,args.proposal,validation=args.validation)


if __name__=='__main__':
    main()
