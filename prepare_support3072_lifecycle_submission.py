"""Conditional lifecycle-only v93 export, never upload. Only new committed decisions open full targets.

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
import mitra_support3072_lifecycle as core
import prepare_support3072_submission as previous
import mitra_support3072_gc_probe as diagnostic
import mitra_support3072_screen as screen
import mitra_long_context_probe as probe
import exaone_validation_data as data
import kaggle_workflow as workflow
import submit_iteration as submission

ROOT = Path(__file__).resolve().parent
NAME, RUN = 'v93_Mitra_support3072_lifecycle', 'run_056'
BATCH, PROPOSAL = 'online_iteration_20261116', 'proposal_nov17_support3072_lifecycle'
AUTH = f'reports/{BATCH}/authorization.json'
DECISION = f'reports/{BATCH}/support3072_lifecycle_decision.json'
OLD_FULL = 'proposal_nov16_full_episode'
BUDGET, SEED = 7200, 20261112
OWN = ['prepare_support3072_lifecycle_submission.py', 'test_prepare_support3072_lifecycle_submission.py',
       *previous.OWN, *core.SOURCES, 'test_mitra_support3072_lifecycle.py']
STATIC = 'reports/support3072_lifecycle_devicefix_static_validation.json'
DESIGN = 'reports/support3072_lifecycle_repair_design.json'
FAILURE_REVIEW = 'reports/support3072_export_failure_review.json'
INHERITANCE = dict(inherited_algorithm='v93_Mitra_support3072',
    inherited_algorithm_evidence='reports/run_056/summary.json', runtime_three_fold_reproduced=False)
read, write, sha = probe.read, screen.write, base.sha256


def verify_hashes(mapping):
    for name, digest in mapping.items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file()
                or not (ROOT/path).resolve().is_relative_to(ROOT.resolve()) or sha(ROOT/path) != digest):
            raise RuntimeError('Bound source/evidence changed: '+name)


def bind(path, evidence):
    evidence[str(path.relative_to(ROOT))] = sha(path)


def collect_evidence():
    """Authenticate original CV unchanged; add independently qualified runtime facts.

    No approval is created, no complete-label reader or model engine is called.
    The previous module's core/SOURCES are deliberately never reassigned.
    """
    facts = previous.collect_evidence()
    evidence = dict(facts['evidence_files'])
    current, lock = diagnostic.contract()
    out, report = diagnostic.OUTPUT, diagnostic.REPORT
    frozen = read(out/'freeze.json')
    if ({k:v for k,v in current.items() if k != 'source_commit'} !=
            {k:v for k,v in frozen.items() if k != 'source_commit'}):
        raise RuntimeError('Current lifecycle sources differ from passed devicefix resource.')
    diagnostic.check_frozen(frozen)
    if facts['environment_sha256'] != frozen['environment_sha256']:
        raise RuntimeError('Original CV and lifecycle environment differ.')
    for name in ['freeze.json', 'result.json', 'requirements.lock']:
        if (out/name).read_bytes() != (report/name).read_bytes():
            raise RuntimeError('Lifecycle resource report differs: '+name)
        bind(out/name, evidence); bind(report/name, evidence)
    if (out/'requirements.lock').read_text() != lock:
        raise RuntimeError('Lifecycle resource lock differs.')
    result = read(out/'result.json')
    expected = dict(status='complete', diagnostic_qualification=True, budget_seconds=1200,
                    externally_supervised=True, no_real_targets=True, no_accuracy_computed=True,
                    no_automatic_fullfit=True)
    if (any(result.get(k) != v for k,v in expected.items()) or
            not 0 < result.get('seconds', float('inf')) <= 1200 or
            [r.get('stage') for r in result.get('stages', [])] != ['small', 'full']):
        raise RuntimeError('Complete bounded devicefix diagnostic required.')
    for kind, row in zip(['small', 'full'], result['stages']):
        directory = out/kind
        worker = read(directory/'worker_result.json')
        if row.get('worker_result_sha256') != sha(directory/'worker_result.json') or row.get('release_complete') is not True:
            raise RuntimeError('Lifecycle resource worker identity differs.')
        diagnostic.qualify_stage(worker, directory, kind)  # Includes saved small NPZ recomputation.
        for path in sorted(directory.iterdir()):
            if path.is_file():
                bind(path, evidence)
                if path.suffix != '.npz':
                    public = report/kind/path.name
                    if path.read_bytes() != public.read_bytes():raise RuntimeError('Lifecycle stage report differs.')
                    bind(public, evidence)
    audit_path = report/'root_audit.json'
    audit = read(audit_path)
    if (audit.get('status') != 'root_verified_complete_bounded_diagnostic'
            or audit.get('small_saved_probabilities_bitwise_equal') is not True
            or audit.get('no_real_accuracy') is not True
            or audit.get('full_optimizer_steps') != [0,1,2]):
        raise RuntimeError('Root devicefix diagnostic audit required.')
    expected_proofs = [(kind, step, sha(out/kind/f'barrier_{step}.json'))
                       for kind, steps in [('small',[1]),('full',[1,2])] for step in steps]
    if [(r['stage'],r['step'],r['proof_sha256']) for r in audit['state_proofs']] != expected_proofs:
        raise RuntimeError('Root audit state proof identities differ.')
    bind(audit_path, evidence)
    for name,digest in frozen['sources'].items():
        path = out/'source'/name
        if sha(path) != digest:raise RuntimeError('Diagnostic source snapshot changed.')
        bind(path, evidence)
    evidence.update(frozen['evidence_files'])
    static = read(ROOT/STATIC)
    if (static.get('status') != 'passed_cpu_and_static_after_device_dispatch_fix' or static.get('tests_run') != 17
            or static.get('cpu_two_update_tab2d_exact') is not True
            or static.get('mps_index_zero_dispatch_regression_mock') is not True
            or static.get('full_runtime_CV_reproduced') is not False
            or static.get('ast', {}).get('numerical_program_ast_identical_after_whitelist') is not True
            or static['ast'].get('step0_or_refit_barrier_added') is not False
            or set(static.get('source_sha256', {})) != {'mitra_lifecycle_guard.py', 'test_mitra_lifecycle_guard.py',
                'mitra_support3072_gc_probe.py', 'test_mitra_support3072_gc_probe.py',
                'mitra_support3072_lifecycle.py', 'test_mitra_support3072_lifecycle.py'}):
        raise RuntimeError('Current17-test/AST/CPU lifecycle evidence required.')
    verify_hashes(static['source_sha256'])
    design = read(ROOT/DESIGN)
    if (design.get('new_fullfit_budget_seconds') != BUDGET or design.get('maximum_new_fullfit_runs') != 1
            or design.get('no_overwrite_resume_or_checkpoint_reuse') is not True
            or design.get('no_automatic_fullfit_or_upload') is not True):
        raise RuntimeError('Single conditional lifecycle repair scope differs.')
    evidence.update(design['evidence'])
    evidence.update(read(ROOT/FAILURE_REVIEW)['evidence_sha256'])
    # Preserve and bind the complete first failed proposal, including source snapshots.
    for folder in [ROOT/'artifacts/proposal_nov16_support3072', ROOT/'reports/proposal_nov16_support3072']:
        for path in sorted(folder.rglob('*')):
            if path.is_file():bind(path, evidence)
    failure = read(ROOT/'artifacts/proposal_nov16_support3072/failure.json')
    if failure.get('status') != 'failed_incomplete_not_upload_eligible':
        raise RuntimeError('Original failed export evidence must remain failed.')
    for name in [STATIC, DESIGN, FAILURE_REVIEW]:bind(ROOT/name, evidence)
    sources = {**facts['sources'], **frozen['sources'], **static['source_sha256'],
               **{name:sha(ROOT/name) for name in OWN}}
    verify_hashes(sources); verify_hashes(evidence)
    probe.require_committed([*sources, *[p for p in evidence if p.startswith('reports/')]])
    runtime = dict(**INHERITANCE, diagnostic_result_sha256=sha(out/'result.json'),
                   diagnostic_freeze_sha256=sha(out/'freeze.json'), static_validation_sha256=sha(ROOT/STATIC),
                   repair_design_sha256=sha(ROOT/DESIGN), devicefix_plan_sha256=diagnostic.PLAN_SHA,
                   diagnostic_does_not_prove_root_cause=True, complete_fullfit_not_yet_verified=True)
    return {**facts, 'evidence_files':dict(sorted(evidence.items())), 'sources':dict(sorted(sources.items())),
            'runtime_evidence':runtime}


def decision_evidence():
    probe.require_committed([DECISION, AUTH, *OWN])  #Before any full-label reader.
    decision, auth = read(ROOT/DECISION), read(ROOT/AUTH)
    if (auth.get('batch_id') != BATCH or auth.get('competition') != 'spaceship-titanic'
            or auth.get('maximum_upload_attempts') != 10 or auth.get('weekly_reserve_percent') != 30
            or auth.get('no_duplicate_or_blind_retry_upload') is not True):
        raise RuntimeError('Current bounded batch authorization required.')
    expected = dict(status='approved_conditional_exploratory_export', batch_id=BATCH,
                    batch_authorization_sha256=sha(ROOT/AUTH), version=NAME, module='mitra_support3072_lifecycle', run=RUN,
                    proposal=PROPOSAL, validation_kind='random_development', exploratory=True,
                    prior_A_eligible=None, prior_S_supported=False, threshold=.5, maximum_attempts=1,
                    export_budget_seconds=BUDGET, fit_seed=SEED, max_steps=50,
                    maximum_new_fullfit_runs=1, **INHERITANCE)
    if (any(decision.get(k) != v or type(decision.get(k)) is not type(v) for k, v in expected.items())
            or not isinstance(decision.get('rationale'), str) or not decision['rationale'].strip()):
        raise RuntimeError('Separate root decision identity/scope differs.')
    facts = collect_evidence()
    for key in ['evidence_files', 'sources', 'environment_sha256', 'screen_accuracy', 'screen_log_loss',
                'source_protocol_sha256', 'runtime_evidence']:
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


def lifecycle_proofs(metadata, out, evidence):
    """Verify all50 persisted exact-state audits; return a compact public index."""
    if (metadata.get('runtime_provenance') != core.RUNTIME_PROVENANCE
            or metadata.get('source_hashes') != {name:evidence['sources'][name] for name in core.SOURCES}
            or metadata.get('environment_sha256') != evidence['environment_sha256']):
        raise RuntimeError('Lifecycle training provenance/source/environment differs.')
    inner, refit = metadata.get('inner_history', []), metadata.get('refit_history', [])
    if (len(inner) != 51 or [r.get('step') for r in inner] != list(range(51))
            or any(r.get('stage') != 'inner' for r in inner)
            or 'lifecycle_barrier' in inner[0] or any('lifecycle_barrier' in r for r in refit)):
        raise RuntimeError('Exactly50 inner barriers and none at step0/refit required.')
    expected_files = {f'training_log.lifecycle_step{step:02d}.json' for step in range(1,51)}
    if {p.name for p in out.glob('training_log.lifecycle_step*.json')} != expected_files:
        raise RuntimeError('Exactly50 complete lifecycle proof files required.')
    template = read(diagnostic.OUTPUT/'full/barrier_1.json')['before']
    observed_stages = ['before_pre_hash','after_pre_hash','barrier_after_synchronize1',
        'barrier_after_gc_collect','barrier_after_empty_cache','barrier_after_synchronize2',
        'before_post_hash','after_post_hash']
    tensor_keys = {'object_id','storage_id','storage_data_ptr','storage_bytes','storage_offset',
                   'shape','stride','dtype','device','requires_grad','bytes_sha256'}
    def tensor_schema(value):
        if (set(value) != tensor_keys or not isinstance(value['bytes_sha256'],str)
                or len(value['bytes_sha256']) != 64 or any(c not in '0123456789abcdef' for c in value['bytes_sha256'])
                or any(type(value[k]) is not int for k in ['object_id','storage_id','storage_data_ptr','storage_bytes','storage_offset'])
                or len(value['shape']) != len(value['stride'])):
            raise RuntimeError('Incomplete tensor bytes/object/storage schema.')
    records, persistent = [], None
    for step, row in enumerate(inner[1:], 1):
        item = row.get('lifecycle_barrier', {})
        name = f'training_log.lifecycle_step{step:02d}.json'; path = out/name
        if (item.get('proof_file') != name or item.get('proof_sha256') != sha(path)
                or item.get('step') != step or item.get('proof_persisted') is not True
                or item.get('status') != 'complete' or item.get('state_preserved') is not True):
            raise RuntimeError('Lifecycle proof row/path/hash/status differs at step'+str(step))
        proof = read(path); state = proof.get('before', {})
        if (proof.get('status') != 'complete' or proof.get('state_preserved') is not True
                or state != proof.get('after') or set(state) != set(template)
                or proof.get('barrier') != core.lifecycle_guard.BARRIER or proof.get('device') != 'mps'
                or type(proof.get('gc_collected')) is not int or proof['gc_collected'] < 0
                or [v['stage'] for v in proof.get('observations', [])] != observed_stages
                or any(item.get(k) != proof.get(k) for k in ['barrier','gc_collected','observations','observation_caveat'])):
            raise RuntimeError('Complete unchanged lifecycle state required at step'+str(step))
        if (state['all_gradients_none'] is not True or state['all_input_gradients_none'] is not True
                or state['model_training'] is not True or not state['module_training']
                or set(state['module_training']) != set(template['module_training'])
                or set(state['module_training'].values()) != {True}
                or state['optimizer_type'] != 'AdamW' or not state['optimizer_groups']
                or set(state['optimizer_steps']) != set(state['parameters'])
                or set(state['optimizer_steps'].values()) != {float(step)}
                or set(state['optimizer_state']) != set(state['parameters'])
                or set(state['rng']) != set(template['rng'])
                or state['rng']['sampler'] != state['rng']['counterfactual_sampler']):
            raise RuntimeError('Complete persistent Adam/model/mode/RNG proof required.')
        # Shapes/names/dtypes must match the actual qualified original full model.
        for section in ['parameters','buffers']:
            if set(state[section]) != set(template[section]):raise RuntimeError('Model state names differ.')
            for key, value in state[section].items():
                tensor_schema(value)
                if any(value[k] != template[section][key][k] for k in ['shape','stride','dtype','device','requires_grad']):
                    raise RuntimeError('Model tensor schema differs from qualified resource.')
        for name, shape, dtype in [('data',[6954,13],'torch.float32'),('targets',[6954],'torch.int64'),('query',[1739,13],'torch.float32')]:
            value = state['inputs'][name]; tensor_schema(value)
            if value['shape'] != shape or value['dtype'] != dtype or value['device'] != 'mps:0' or value['requires_grad'] is not False:
                raise RuntimeError('Full inner input schema differs.')
        if set(state['inputs']) != {'data','targets','query'}:raise RuntimeError('Unexpected input state.')
        for name, value in state['optimizer_state'].items():
            if value['parameter_object_id'] != state['parameters'][name]['object_id'] or set(value['values']) != {'step','exp_avg','exp_avg_sq'}:
                raise RuntimeError('Adam moment/parameter association differs.')
            for part, entry in value['values'].items():
                tensor_schema(entry['tensor'])
                original = template['optimizer_state'][name]['values'][part]['tensor']
                if any(entry['tensor'][k] != original[k] for k in ['shape','dtype','device','requires_grad']):
                    raise RuntimeError('Adam tensor schema differs.')
        for group in state['optimizer_groups']:
            if group['options']['lr'] != row['learning_rate'] or group['options']['weight_decay'] != .3:
                raise RuntimeError('Adam schedule or weight decay differs.')
        stable = dict(model=state['model_object_id'], optimizer=state['optimizer_object_id'],
                      inputs=state['inputs'], groups=[g['object_id'] for g in state['optimizer_groups']],
                      parameter_ids={k:v['object_id'] for k,v in state['parameters'].items()},
                      moment_ids={k:(v['state_object_id'], {n:t['tensor']['object_id'] for n,t in v['values'].items()})
                                  for k,v in state['optimizer_state'].items()})
        if persistent is not None and stable != persistent:raise RuntimeError('Persistent inner optimizer/model/input identity changed.')
        persistent = stable
        records.append(dict(step=step, file=path.name, sha256=sha(path), state_preserved=True,
                            gc_collected=proof['gc_collected']))
    return dict(status='complete', verified_inner_barriers=50, step0_barriers=0, refit_barriers=0,
                source_hashes={name:evidence['sources'][name] for name in core.SOURCES},
                runtime_provenance=core.RUNTIME_PROVENANCE, proofs=records)


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
        proof_index = lifecycle_proofs(metadata, out, evidence)
        write(out/'lifecycle_proofs.json', proof_index)
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
                    selected_steps=metadata['selected_steps'], lifecycle_proofs_sha256=sha(out/'lifecycle_proofs.json'),
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
    proof_index = lifecycle_proofs(metadata, out, evidence)
    if (proof_index != read(out/'lifecycle_proofs.json')
            or child.get('lifecycle_proofs_sha256') != sha(out/'lifecycle_proofs.json')):
        raise RuntimeError('Persisted public lifecycle proof index differs.')
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
                     screen_accuracy=decision['screen_accuracy'], threshold=.5, **INHERITANCE)
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
                    lifecycle_proofs=proof_index, lifecycle_proofs_sha256=sha(out/'lifecycle_proofs.json'), **INHERITANCE,
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
    for name in ['fit_metadata.json', 'reload_metadata.json', 'training_log.json', 'worker_result.json', 'lifecycle_proofs.json']:
        shutil.copy2(out/name, report/name)
    probe.check_deadline(deadline)
    return manifest


def prepare(session_file):
    out, report = ROOT/'artifacts'/PROPOSAL, ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():raise FileExistsError('No overwrite/resume of prior export evidence.')
    probe.quota_guard(session_file)
    evidence = decision_evidence()
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
