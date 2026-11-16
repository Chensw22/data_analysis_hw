"""Conditional v93 export, never upload. Only new committed decisions open full targets.

Collecting completed development evidence does not create or imply approval.
A spawned worker owns MPS; its parent enforces the complete7200s export deadline.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import weakref

import joblib
import numpy as np
import pandas as pd

import mitra_candidate as base
import mitra_support3072 as core
import mitra_support3072_screen as screen
import mitra_long_context_probe as probe
import exaone_validation_data as data
import kaggle_workflow as workflow
import submit_iteration as submission

ROOT = Path(__file__).resolve().parent
NAME, RUN = 'v93_Mitra_support3072', 'run_056'
BATCH, PROPOSAL = 'online_iteration_20261116', 'proposal_nov16_support3072'
AUTH = f'reports/{BATCH}/authorization.json'
DECISION = f'reports/{BATCH}/support3072_decision.json'
OLD_FULL = 'proposal_nov16_full_episode'
BUDGET, SEED = 7200, 20261112
OWN = ['prepare_support3072_submission.py', 'test_prepare_support3072_submission.py',
       'submit_iteration.py', 'kaggle_workflow.py', 'quota_guard.py']
read, write, sha = probe.read, screen.write, base.sha256


def verify_hashes(mapping):
    for name, digest in mapping.items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file()
                or not (ROOT/path).resolve().is_relative_to(ROOT.resolve()) or sha(ROOT/path) != digest):
            raise RuntimeError('Bound source/evidence changed: '+name)


def bind(path, evidence):
    evidence[str(path.relative_to(ROOT))] = sha(path)


def old_full_evidence(evidence, lock):
    """Authenticate the old full-fit logs and portable-file hashes; never unpickle."""
    out = ROOT/'artifacts'/OLD_FULL
    for name in ['manifest.json', 'authorization.json', 'selection.json', 'fit_metadata.json', 'reload_metadata.json']:
        data.report_pair(OLD_FULL, name, evidence)
    manifest, log = read(out/'manifest.json'), read(out/'fit_metadata.json')
    if (manifest.get('status') != 'complete' or manifest['selection']['version'] != screen.REFERENCE
            or manifest.get('fit_rows') != 8693 or manifest.get('rows') != 4277 or manifest.get('reload_rows') != 4277
            or manifest.get('export_reload_inference_verified') is not True or manifest.get('reload_threshold_decisions_equal') is not True
            or not 0 <= manifest.get('reload_max_absolute_difference', float('inf')) <= 2e-6
            or manifest.get('model_sha256') != sha(out/'model.joblib')
            or manifest.get('environment_sha256') != hashlib.sha256(lock.encode()).hexdigest()
            or read(out/'reload_metadata.json') != log or log.get('recipe') != core.original.SETTINGS
            or log.get('status') != 'fit_complete' or log.get('seed') != SEED
            or log.get('inner_training_rows') != 6954 or log.get('inner_validation_rows') != 1739):
        raise RuntimeError('Original complete full v81 resource and paired-log evidence required.')
    for name in ['model.joblib', 'preparation_protocol.json', 'environment.lock.txt']:
        bind(out/name, evidence)
    if (out/'environment.lock.txt').read_text() != lock:
        raise RuntimeError('Old full-fit environment differs.')
    for name, digest in manifest['provenance']['source_hashes'].items():
        path = out/'source'/name
        if sha(path) != digest:
            raise RuntimeError('Original full-fit source snapshot changed.')
        bind(path, evidence)
    return log


def collect_evidence():
    """Only complete passed screens; development-only target replay, no full labels."""
    out = ROOT/'artifacts'/RUN
    # Refuse incomplete phases before opening any development targets.
    summary = read(out/'summary.json')
    if (summary.get('status') != 'complete' or summary.get('candidate') != NAME
            or summary.get('completed_folds') != 3 or summary.get('continuation', {}).get('passed') is not True):
        raise RuntimeError('All3 complete folds and the fixed continuation gate must pass.')
    frozen, lock = screen.resource_contract()
    evidence = {}
    for name in ['protocol.json', 'summary.json', 'progress.json', 'assessment.json', 'selection.json',
                 'screen_results.csv', 'requirements.lock', *[f'fold{k}.json' for k in range(3)]]:
        data.report_pair(RUN, name, evidence)
    protocol, child = read(out/'protocol.json'), read(out/'worker_result.json')
    expected = dict(candidate=NAME, seed=SEED, folds=3, development_rows=6954, threshold=.5,
                    settings=screen.SETTINGS, budget_seconds=18000, externally_supervised=True,
                    plan_sha256=screen.PLAN_SHA, old_holdout_labels_read=False,
                    automatic_export=False, automatic_upload=False)
    if (any(protocol.get(k) != v for k, v in expected.items()) or read(out/'progress.json') != summary
            or summary.get('budget_seconds') != 18000 or summary.get('externally_supervised') is not True
            or summary.get('old_holdout_labels_read') is not False or summary.get('S_supported') is not False
            or summary.get('automatic_export') is not False or summary.get('automatic_upload') is not False
            or child.get('status') != 'worker_complete' or child.get('completed_folds') != 3 or child.get('release_complete') is not True):
        raise RuntimeError('Completed fixed5h screen/protocol/release identity differs.')
    screen.assert_frozen(protocol['frozen'])
    frozen['sources'] = {name:sha(ROOT/name) for name in screen.OWN}
    inputs, splits, reference, old_logs = screen.reference_inputs(frozen)  #6954 labels only.
    saved, current = copy.deepcopy(protocol['frozen']), copy.deepcopy(frozen)
    for value in [saved, current]:value['resource'].pop('source_commit', None)
    if saved != current or (out/'requirements.lock').read_text() != lock:
        raise RuntimeError('Current sources/reference/environment differ from screened identity.')
    dev, y = inputs['dev'], inputs['y']
    if protocol['development_ids_sha256'] != base.ids_hash(dev) or protocol['development_labels_sha256'] != base.ids_hash(y):
        raise RuntimeError('Development identity changed.')
    filename = f'{NAME}_seed{SEED}.npz'
    required = {filename, 'screen_results.csv', 'assessment.json', 'selection.json', 'worker_result.json'}
    if set(summary['result_files_sha256']) != required:
        raise RuntimeError('Complete immutable screen result manifest required.')
    for name, digest in summary['result_files_sha256'].items():
        if sha(out/name) != digest:raise RuntimeError('Completed result changed: '+name)
        bind(out/name, evidence)
    with np.load(out/filename, allow_pickle=False) as cache:
        p, folds = cache['prob'].copy(), cache['fold_id'].copy()
        if (not np.array_equal(cache['dev_ids'], dev)
                or any(not np.array_equal(np.flatnonzero(folds == k), b) for k, (_, b) in enumerate(splits))):
            raise RuntimeError('Complete OOF fold membership differs.')
        score = data.metrics(p, y, splits)
        if not np.array_equal(cache['fold_accuracy'], score['fold_accuracy']):
            raise RuntimeError('OOF fold metrics differ.')
    with np.load(out/'partial_oof.npz', allow_pickle=False) as partial:
        if any(not np.array_equal(partial[k], v) for k, v in [('prob', p), ('dev_ids', dev), ('fold_id', folds)]):
            raise RuntimeError('Final worker OOF differs from published OOF.')
    bind(out/'partial_oof.npz', evidence)
    ref = data.metrics(reference, y, splits)
    gate = screen.continuation(score, ref)
    assessment = read(out/'assessment.json')
    if (score != summary['metrics'] or ref != summary['reference'] or gate != summary['continuation'] or not gate['passed']
            or child['metrics'] != score or child['reference'] != ref
            or assessment != {k:v for k,v in summary.items() if k != 'result_files_sha256'}
            or summary['paired'] != screen.paired(p, reference, y, splits)):
        raise RuntimeError('Recomputed complete screen/paired metrics or gate differs.')
    table = pd.read_csv(out/'screen_results.csv')
    selection = read(out/'selection.json')
    if (len(table) != 1 or table.iloc[0].version != NAME
            or any(abs(float(table.iloc[0][k])-score[k]) > 1e-12 for k in ['accuracy', 'log_loss'])
            or selection.get('version') != NAME or selection.get('continuation_passed') is not True
            or selection.get('screen_accuracy') != score['accuracy'] or selection.get('automatic_eligibility') is not False):
        raise RuntimeError('Selection/CSV identity differs.')
    for fold, (a, b) in enumerate(splits):
        row = read(out/f'fold{fold}.json')
        name = f'{NAME}_seed{SEED+fold}_{base.ids_hash(dev[a])[:12]}.json'
        path = data.report_pair(RUN, name, evidence); log = read(path)
        screen.check_log(log, old_logs[fold], dev[a], dev[b], y[a], SEED+fold, True)
        expected_sources = {name:{**frozen['resource']['sources'], **frozen['sources']}[name] for name in core.SOURCES}
        if log.get('source_hashes') != expected_sources or log.get('environment_sha256') != frozen['resource']['environment_sha256']:
            raise RuntimeError('Actual fold source/environment differs.')
        with np.load(out/f'fold{fold}_rows.npz', allow_pickle=False) as ids:
            if not np.array_equal(ids['train_ids'], dev[a]) or not np.array_equal(ids['validation_ids'], dev[b]):
                raise RuntimeError('Fold row cache differs.')
        bind(out/f'fold{fold}_rows.npz', evidence)
        single = [(np.array([], dtype=int), np.arange(len(b)))]
        if (row.get('status') != 'complete' or row.get('fold') != fold or row.get('seed') != SEED+fold
                or row.get('training_log_sha256') != sha(path) or row.get('train_ids_sha256') != base.ids_hash(dev[a])
                or row.get('validation_ids_sha256') != base.ids_hash(dev[b]) or row.get('selected_steps') != log['selected_steps']
                or row.get('paired_inner_steps') != 50 or row.get('paired_common_refit_steps') != min(log['selected_steps'], old_logs[fold]['selected_steps'])
                or row['metrics'] != data.metrics(p[b], y[b], single) or row['reference'] != data.metrics(reference[b], y[b], single)
                or row['paired'] != screen.paired(p[b], reference[b], y[b], single) or child['folds'][fold] != row):
            raise RuntimeError('Complete paired fold evidence differs.')
    evidence.update(frozen['evidence'])
    old_full_evidence(evidence, lock)
    sources = {**frozen['resource']['sources'], **frozen['sources']}
    verify_hashes(sources); verify_hashes(evidence)
    probe.require_committed([*sources, *[name for name in evidence if name.startswith('reports/')]])
    return dict(evidence_files=dict(sorted(evidence.items())), sources=dict(sorted(sources.items())),
                screen_accuracy=score['accuracy'], screen_log_loss=score['log_loss'],
                source_protocol_sha256=sha(out/'protocol.json'), environment_sha256=hashlib.sha256(lock.encode()).hexdigest())


def decision_evidence():
    probe.require_committed([DECISION, AUTH, *OWN])  #Before any full-label reader.
    decision, auth = read(ROOT/DECISION), read(ROOT/AUTH)
    if (auth.get('batch_id') != BATCH or auth.get('competition') != 'spaceship-titanic'
            or auth.get('maximum_upload_attempts') != 10 or auth.get('weekly_reserve_percent') != 30
            or auth.get('no_duplicate_or_blind_retry_upload') is not True):
        raise RuntimeError('Current bounded batch authorization required.')
    expected = dict(status='approved_conditional_exploratory_export', batch_id=BATCH,
                    batch_authorization_sha256=sha(ROOT/AUTH), version=NAME, module='mitra_support3072', run=RUN,
                    proposal=PROPOSAL, validation_kind='random_development', exploratory=True,
                    prior_A_eligible=None, prior_S_supported=False, threshold=.5, maximum_attempts=1,
                    export_budget_seconds=BUDGET, fit_seed=SEED, max_steps=50)
    if (any(decision.get(k) != v or type(decision.get(k)) is not type(v) for k, v in expected.items())
            or not isinstance(decision.get('rationale'), str) or not decision['rationale'].strip()):
        raise RuntimeError('Separate root decision identity/scope differs.')
    facts = collect_evidence()
    for key in ['evidence_files', 'screen_accuracy', 'screen_log_loss', 'source_protocol_sha256']:
        if decision.get(key) != facts[key]:raise RuntimeError('Decision differs from complete screen: '+key)
    sources = {**facts['sources'], **{name:sha(ROOT/name) for name in [*OWN, DECISION, AUTH]}}
    probe.require_committed(sources)
    return dict(decision=decision, decision_path=DECISION, decision_sha256=sha(ROOT/DECISION),
                authorization_sha256=sha(ROOT/AUTH), sources=sources, evidence_files=facts['evidence_files'],
                environment_sha256=facts['environment_sha256'],
                committed_at=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def assert_evidence(evidence):
    verify_hashes(evidence['sources']); verify_hashes(evidence['evidence_files'])
    if hashlib.sha256(base.environment_lock().encode()).hexdigest() != evidence['environment_sha256']:
        raise RuntimeError('Qualified environment changed.')
    base.verify_checkpoint()


def load_full_data(evidence):
    assert_evidence(evidence)
    train = pd.read_csv(ROOT/'data/raw/train.csv', dtype={'PassengerId':str})
    test = pd.read_csv(ROOT/'data/raw/test.csv', dtype={'PassengerId':str})
    if (len(train) != 8693 or len(test) != 4277 or 'Transported' in test
            or not pd.api.types.is_bool_dtype(train.Transported) or train.Transported.isna().any()):
        raise RuntimeError('Complete official binary training/test roster required.')
    features = data.compact_features(pd.concat([train.drop(columns='Transported'), test], ignore_index=True))
    if data.feature_hash(features) != read(ROOT/data.PROOF)['compact_values_sha256']:
        raise RuntimeError('Full compact13 attributes differ from original v81 proof.')
    return features.iloc[:8693], train.Transported.astype(np.int64).to_numpy(), features.iloc[8693:], test.PassengerId


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()


def context_identity(model, x, y, query):
    if (not isinstance(model, base.MitraPredictor) or model.query_chunk != 2318
            or model.support.shape != (8693, 13) or not np.array_equal(model.labels, y)
            or model.preprocessor.columns != base.COMPACT
            or not np.array_equal(model.support, model.preprocessor.transform(x), equal_nan=True)
            or any(value.device.type != 'cpu' for value in model.state.values())):
        raise RuntimeError('Original full-support CPU-portable Mitra context/schema required.')
    return dict(support_sha256=array_hash(model.support), labels_sha256=base.ids_hash(model.labels),
                transformed_query_sha256=array_hash(model.preprocessor.transform(query)),
                columns=list(model.preprocessor.columns), config=model.config,
                full_support_rows=8693, query_rows=4277, query_chunk=2318)


def valid_probabilities(p):
    p = np.asarray(p)
    if (p.shape != (4277, 2) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1., atol=2e-6, rtol=0)):
        raise RuntimeError('All4277 binary class-order probabilities required.')
    return p


def predict_full(model, query):
    records = []
    def before(module, args):
        support, labels, queries = args[:3]
        records.append(dict(support_rows=int(support.shape[1]), label_rows=int(labels.shape[1]),
                            query_rows=int(queries.shape[1])))
    engine = model.engine(); hook = engine.register_forward_pre_hook(before)
    try:probability = valid_probabilities(model.predict_proba(query))
    finally:hook.remove()
    expected = [dict(support_rows=8693, label_rows=8693, query_rows=n) for n in [2318, 1959]]
    if records != expected:raise RuntimeError('Actual full-support forward/chunk coverage differs.')
    return probability.copy(), records


def compare(first, second):
    first, second = valid_probabilities(first), valid_probabilities(second)
    delta = float(np.max(np.abs(first-second)))
    if delta > 2e-6 or not np.array_equal(first[:, 1] >= .5, second[:, 1] >= .5):
        raise RuntimeError('Full portable reload probability/threshold gate failed.')
    return delta


def configure_runtime():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None, '0') or not base.torch.backends.mps.is_available():
        raise RuntimeError('Qualified MPS runtime without fallback required.')
    base.torch.set_num_threads(4); base.torch.mps.set_per_process_memory_fraction(.65)


def full_export(out, evidence, deadline):
    """Worker-side fit/reload. Full targets exist only in this authorized stage."""
    assert_evidence(evidence); probe.check_deadline(deadline)
    configure_runtime()
    x, y, query, passengers = load_full_data(evidence)
    ids, query_ids = np.arange(8693, dtype=np.int64), np.arange(8693, 12970, dtype=np.int64)
    old = read(ROOT/'artifacts'/OLD_FULL/'fit_metadata.json')
    screen.check_log(old, old, ids, query_ids, y, SEED, False)
    model = restored = None
    try:
        model = core.fit_support3072(x, y, ids, SEED, device='mps', log_path=out/'training_log.json', max_steps=50)
        first, first_coverage = predict_full(model, query)
        metadata = dict(model.metadata, prediction_rows=4277, prediction_ids_sha256=base.ids_hash(query_ids),
                        final_prediction_path='original_MitraPredictor', automatic_further_extension_allowed=False)
        screen.check_log(metadata, old, ids, query_ids, y, SEED, True)
        if (metadata.get('source_hashes') != {name:evidence['sources'][name] for name in core.SOURCES}
                or metadata.get('environment_sha256') != evidence['environment_sha256']):
            raise RuntimeError('Full fit source/environment differs from approved recipe.')
        identity = context_identity(model, x, y, query)
        model.metadata = metadata
        write(out/'fit_metadata.json', metadata)
        joblib.dump(model, out/'model.joblib')
        previous = weakref.ref(model)
        model.release(); model = None; base.release_memory(); gc.collect()
        if previous() is not None:raise RuntimeError('Original model retained across portable reload.')
        probe.check_deadline(deadline)
        restored = joblib.load(out/'model.joblib')
        if restored.device != 'cpu' or restored._engine is not None:
            raise RuntimeError('Saved artifact must reload as CPU state with no engine.')
        if context_identity(restored, x, y, query) != identity or restored.metadata != metadata:
            raise RuntimeError('Portable preprocessing/context/log identity changed.')
        restored.device = 'mps'
        second, second_coverage = predict_full(restored, query)
        difference = compare(first, second)
        if context_identity(restored, x, y, query) != identity:
            raise RuntimeError('Reload inference mutated context/preprocessing.')
        write(out/'reload_metadata.json', restored.metadata)
        restored.release(); restored = None; base.release_memory(); gc.collect()
        assert_evidence(evidence); probe.check_deadline(deadline)
        np.savez(out/'predictions.npz', first=first, reloaded=second, passenger_ids=np.asarray(passengers, dtype=str))
        return dict(status='worker_complete', fit_rows=8693, prediction_rows=4277, reload_rows=4277,
                    reload_max_absolute_difference=difference, reload_threshold_decisions_equal=True,
                    original_model_released_before_reload=True, restored_cpu_state=True, preprocessing_identity=identity,
                    first_forward_records=first_coverage, reload_forward_records=second_coverage,
                    paired_inner_steps=50, paired_common_refit_steps=min(metadata['selected_steps'], old['selected_steps']),
                    selected_steps=metadata['selected_steps'],
                    files_sha256={name:sha(out/name) for name in ['model.joblib', 'fit_metadata.json', 'reload_metadata.json', 'training_log.json', 'predictions.npz']})
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error = None
        for obj in [model, restored]:
            try:
                if obj is not None:obj.release()
            except BaseException as error:
                cleanup_error = cleanup_error or error
        model = restored = None
        try:base.release_memory()
        except BaseException as error:cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            write(out/'cleanup_failure.json', dict(error_type=type(cleanup_error).__name__, error=str(cleanup_error)))
            if active_error is None:raise cleanup_error


def worker(out, evidence, deadline, session_file):
    out = Path(out); base.DEADLINE = deadline
    result = dict(status='failed_inconclusive', release_complete=False)
    authorized = False
    try:
        # Recheck committed authorization in the new process before full targets.
        current = decision_evidence()
        if {k:v for k,v in current.items() if k != 'committed_at'} != {k:v for k,v in evidence.items() if k != 'committed_at'}:
            raise RuntimeError('Decision/source changed before child full-label read.')
        probe.quota_guard(session_file)
        authorized = True
        result = full_export(out, evidence, deadline)
    except BaseException as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        if (out/'training_log.json').exists():
            try:result['current_stage'] = read(out/'training_log.json').get('current_stage')
            except (OSError, ValueError):pass
    finally:
        try:
            if authorized:
                base.release_memory(); result['resources_after_release'] = probe.synchronized_snapshot('mps')
            probe.check_deadline(deadline); result['release_complete'] = True
        except BaseException as error:
            result.update(status='failed_inconclusive', release_complete=False,
                          cleanup_error_type=type(error).__name__, cleanup_error=str(error))
            if 'error_type' not in result:
                result.update(error_type=type(error).__name__, error=str(error))
        write(out/'worker_result.json', result)


def supervise(process, deadline):
    while process.is_alive() and time.monotonic() < deadline:
        process.join(timeout=min(5, max(0, deadline-time.monotonic())))
    if process.is_alive():
        process.kill(); process.join(timeout=5)
        if process.is_alive():raise RuntimeError('Export worker could not be reaped after kill.')
        raise TimeoutError('External supervisor killed export worker at fixed7200s deadline.')
    if process.exitcode != 0:raise RuntimeError('Export worker exited abnormally: '+str(process.exitcode))


def duplicate_attempts(digest):
    return [str(p.relative_to(ROOT)) for p in (ROOT/'reports').rglob('online_attempts/*.json')
            if read(p).get('file_sha256') == digest]


def publish(out, report, evidence, child, started, deadline):
    if child.get('status') != 'worker_complete' or child.get('release_complete') is not True:
        raise RuntimeError('Incomplete export worker: '+str(child.get('error_type'))+': '+str(child.get('error')))
    expected = dict(fit_rows=8693, prediction_rows=4277, reload_rows=4277, paired_inner_steps=50,
                    original_model_released_before_reload=True, restored_cpu_state=True, reload_threshold_decisions_equal=True)
    if any(child.get(k) != v for k,v in expected.items()):raise RuntimeError('Complete fit/reload coverage proof required.')
    for name, digest in child['files_sha256'].items():
        if name not in ['model.joblib', 'fit_metadata.json', 'reload_metadata.json', 'training_log.json', 'predictions.npz'] or sha(out/name) != digest:
            raise RuntimeError('Worker output changed.')
    if len(child['files_sha256']) != 5:raise RuntimeError('All full export records required.')
    coverage = [dict(support_rows=8693, label_rows=8693, query_rows=n) for n in [2318, 1959]]
    if child['first_forward_records'] != coverage or child['reload_forward_records'] != coverage:
        raise RuntimeError('Full first/reload actual forward coverage required.')
    metadata = read(out/'fit_metadata.json')
    if (metadata != read(out/'reload_metadata.json') or metadata.get('status') != 'fit_complete'
            or child['selected_steps'] != metadata['selected_steps']
            or child['paired_common_refit_steps'] != min(metadata['selected_steps'], read(ROOT/'artifacts'/OLD_FULL/'fit_metadata.json')['selected_steps'])):
        raise RuntimeError('Full training/reload metadata differs.')
    with np.load(out/'predictions.npz', allow_pickle=False) as cache:
        first, second, passengers = cache['first'], cache['reloaded'], cache['passenger_ids']
        difference = compare(first, second)
        if child['reload_max_absolute_difference'] != difference:raise RuntimeError('Worker reload difference differs.')
        pending = out/'pending'; pending.mkdir()
        pd.DataFrame(dict(PassengerId=passengers, Transported=first[:,1] >= .5)).to_csv(pending/'submission.csv', index=False)
        pd.DataFrame(dict(PassengerId=passengers, probability=first[:,1], reload_probability=second[:,1])).to_csv(pending/'test_probabilities.csv', index=False)
    check = workflow.validate_submission(pending/'submission.csv', ROOT/'data/raw/test.csv')
    if duplicate_attempts(check['sha256']):raise RuntimeError('Identical CSV already attempted; no repeated upload artifact.')
    if submission.count_attempts(ROOT/'reports', BATCH) >= 10:raise RuntimeError('All batch attempts consumed.')
    assert_evidence(evidence)
    current = decision_evidence()
    if {k:v for k,v in current.items() if k != 'committed_at'} != {k:v for k,v in evidence.items() if k != 'committed_at'}:
        raise RuntimeError('Committed decision/evidence changed during export.')
    decision = evidence['decision']
    selection = dict(version=NAME, source_run=RUN, components=[NAME], stage='explicit_autonomous_exploration',
                     A_eligible=None, S_supported=False, validation_kind='random_development',
                     screen_accuracy=decision['screen_accuracy'], threshold=.5)
    bound = dict(status='bound_single_upload_authorization', competition='spaceship-titanic', batch_id=BATCH,
                 source_authorization_sha256=evidence['authorization_sha256'], proposal=PROPOSAL, version=NAME,
                 file_sha256=check['sha256'], model_sha256=sha(out/'model.joblib'), maximum_upload_attempts_for_this_file=1,
                 desktop_reminder_required=True, decision_sha256=evidence['decision_sha256'], decision_path=DECISION,
                 source_protocol_sha256=decision['source_protocol_sha256'], prepared_from_commit=evidence['committed_at'],
                 no_duplicate_or_blind_retry_upload=True)
    manifest = dict(**check, status='complete', batch_id=BATCH, selection=selection, fit_rows=8693,
                    export_reload_inference_verified=True, reload_rows=4277, reload_max_absolute_difference=difference,
                    reload_threshold_decisions_equal=True, model_sha256=bound['model_sha256'], authorization=bound,
                    budget_seconds=BUDGET, externally_supervised=True, threshold=.5, fit_seed=SEED,
                    source_hashes=evidence['sources'], environment_sha256=evidence['environment_sha256'],
                    paired_inner_steps=50, paired_common_refit_steps=child['paired_common_refit_steps'],
                    provenance=dict(source_hashes=evidence['sources'], decision_path=DECISION,
                                    decision_sha256=evidence['decision_sha256'], evidence_files=evidence['evidence_files'],
                                    source_protocol_sha256=decision['source_protocol_sha256'], prepared_from_commit=evidence['committed_at']),
                    seconds=time.monotonic()-started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    probe.check_deadline(deadline)
    for name in ['submission.csv', 'test_probabilities.csv']:(pending/name).replace(out/name)
    pending.rmdir()
    for name, value in [('selection.json',selection), ('authorization.json',bound), ('manifest.json',manifest)]:
        write(out/name,value); write(report/name,value)
    for name in ['fit_metadata.json', 'reload_metadata.json', 'training_log.json', 'worker_result.json']:
        shutil.copy2(out/name, report/name)
    probe.check_deadline(deadline)
    return manifest


def prepare(session_file):
    out, report = ROOT/'artifacts'/PROPOSAL, ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():raise FileExistsError('No overwrite/resume of prior export evidence.')
    evidence = decision_evidence()
    probe.quota_guard(session_file)
    if submission.count_attempts(ROOT/'reports', BATCH) >= 10:raise RuntimeError('All batch attempts consumed.')
    started = time.monotonic(); deadline = started+BUDGET
    out.mkdir(parents=True); report.mkdir(parents=True); process = None
    try:
        assert_evidence(evidence)
        for folder in [out,report]:(folder/'environment.lock.txt').write_text(base.environment_lock())
        for name in evidence['sources']:
            target = out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,target)
        write(out/'preparation_protocol.json', dict(**evidence, budget_seconds=BUDGET, externally_supervised=True,
              fit_seed=SEED, max_steps=50, sources_frozen_before_full_labels=True, full_targets_only_final_fit=True))
        process = multiprocessing.get_context('spawn').Process(target=worker,args=(str(out),evidence,deadline,session_file))
        process.start(); supervise(process,deadline); probe.check_deadline(deadline)
        child = read(out/'worker_result.json')
        return publish(out,report,evidence,child,started,deadline)
    except BaseException as error:
        if process is not None and process.is_alive():process.kill(); process.join(timeout=5)
        failure = dict(status='failed_incomplete_not_upload_eligible', candidate=NAME, error_type=type(error).__name__,
                       error=str(error), seconds=time.monotonic()-started, budget_seconds=BUDGET, externally_supervised=True)
        if (out/'worker_result.json').exists():
            child = read(out/'worker_result.json')
            failure.update(worker_error_type=child.get('error_type'), worker_error=child.get('error'), worker_current_stage=child.get('current_stage'))
        if isinstance(error,TimeoutError):
            write(out/'external_timeout.json', dict(status='external_deadline_exceeded',budget_seconds=BUDGET,
                  child_reaped=process is None or not process.is_alive(),partial_evidence_preserved=True))
        for folder in [out,report]:
            for name in ['submission.csv','test_probabilities.csv','manifest.json','selection.json','authorization.json']:
                (folder/name).unlink(missing_ok=True)
            write(folder/'failure.json',failure)
        if (out/'pending').exists():shutil.rmtree(out/'pending')
        raise
    finally:
        if process is not None and not process.is_alive():process.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--collect-evidence',action='store_true');modes.add_argument('--prepare',action='store_true')
    parser.add_argument('--session-file',type=Path)
    args=parser.parse_args()
    if args.prepare and args.session_file is None:parser.error('--prepare requires --session-file for fresh reserve30 guard.')
    print(json.dumps(prepare(args.session_file) if args.prepare else collect_evidence(),indent=2))
