"""Prepare one reviewed exploratory candidate under the November 16 batch.

This never uploads. Prior A/S decisions are evidence, not overwritten gates.
Only an exact, committed decision can authorize full-data fitting here.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
from pathlib import Path
import shutil
import signal
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

import prepare_top5_submission as common

ROOT = Path(__file__).resolve().parent
AUTH = 'reports/online_iteration_20261116/authorization.json'
BATCH = 'online_iteration_20261116'
SPECS = {
    'mitra_mirror_ensemble': ('v74_Mitra_mirror4', 'run_040', 'stability', 3600),
    'mitra_lastblock100_candidate': ('v76_Mitra_lastblock100', 'run_042', 'export', 10800),
    'mitra_full_episode_candidate': ('v81_Mitra_full_episode50', 'run_046', 'export', 10800),
}
OWN_SOURCES = ['prepare_exploratory_iteration.py', 'test_prepare_exploratory_iteration.py',
               'prepare_top5_submission.py', 'kaggle_workflow.py']
sha, read, write = common.sha, common.read, common.write


def relative_file(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file():
        raise ValueError('Expected an existing repository-relative file.')
    if not (ROOT/path).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('File resolves outside repository.')
    return path.as_posix()


def require_committed(paths):
    paths = sorted(paths)
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *paths], cwd=ROOT).decode()
    if set(filter(None, tracked.split('\0'))) != set(paths):
        raise RuntimeError('Commit every preparation source, decision and authorization first.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=ROOT).strip():
        raise RuntimeError('Preparation sources differ from their committed versions.')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def decision_evidence(decision_path):
    """Metadata and hashes only, before any full-label read or model creation."""
    decision_path = relative_file(decision_path)
    decision, auth = read(ROOT/decision_path), read(ROOT/AUTH)
    if (auth.get('batch_id') != BATCH or auth.get('competition') != 'spaceship-titanic'
            or auth.get('maximum_upload_attempts') != 10
            or auth.get('weekly_reserve_percent') != 30
            or auth.get('no_duplicate_or_blind_retry_upload') is not True):
        raise RuntimeError('Unexpected current batch authorization.')
    module = decision.get('module')
    if module not in SPECS:
        raise ValueError('Unsupported candidate module.')
    name, run, _, _ = SPECS[module]
    if (decision.get('batch_id') != BATCH
            or decision.get('batch_authorization_sha256') != sha(ROOT/AUTH)
            or decision.get('version') != name or decision.get('run') != run
            or decision.get('exploratory') is not True
            or decision.get('threshold') != .5 or decision.get('maximum_attempts') != 1
            or decision.get('prior_A_eligible') not in [False, None]
            or decision.get('prior_S_supported') is not False
            or not isinstance(decision.get('rationale'), str) or not decision['rationale'].strip()):
        raise RuntimeError('Decision identity/scope does not match this exploratory batch.')
    # bool is not an acceptable accuracy value or historical eligibility surrogate.
    if type(decision.get('prior_A_eligible')) not in [bool, type(None)]:
        raise RuntimeError('Historical A eligibility must be false or unknown.')
    accuracy = decision.get('screen_accuracy')
    if isinstance(accuracy, bool) or not isinstance(accuracy, (float, int)) or not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise RuntimeError('Finite completed screen accuracy required.')
    common.safe_name(decision.get('proposal', ''))
    protocol_path = f'artifacts/{run}/protocol.json'
    if decision.get('source_protocol_sha256') != sha(ROOT/protocol_path):
        raise RuntimeError('Original screen protocol changed.')
    protocol = read(ROOT/protocol_path)
    sources = dict(protocol['sources'])
    if module+'.py' not in sources:
        raise RuntimeError('Candidate implementation was not screen-frozen.')
    evidence = decision.get('evidence_files', {})
    screen_path = f'artifacts/{run}/screen_results.csv'
    if not isinstance(evidence, dict) or screen_path not in evidence:
        raise RuntimeError('Decision must bind the complete screen CSV evidence.')
    for mapping in [sources, evidence]:
        for filename, digest in mapping.items():
            if sha(ROOT/relative_file(filename)) != digest:
                raise RuntimeError('Bound source/evidence changed: '+filename)
    table = pd.read_csv(ROOT/screen_path)
    row = table.loc[table.version.eq(name), 'accuracy']
    if len(row) != 1 or abs(float(row.iloc[0])-accuracy) > 1e-12:
        raise RuntimeError('Decision screen accuracy differs from completed evidence.')
    sources.update({filename: sha(ROOT/filename) for filename in OWN_SOURCES+[AUTH, decision_path]})
    commit = require_committed(sources)
    return dict(decision=decision, decision_path=decision_path, decision_sha256=sha(ROOT/decision_path),
        authorization_sha256=sha(ROOT/AUTH), source_protocol=protocol,
        sources=sources, evidence_files=evidence, committed_at=commit)


def duplicate_attempts(digest):
    return [str(path.relative_to(ROOT)) for path in (ROOT/'reports').rglob('online_attempts/*.json')
            if read(path).get('file_sha256') == digest]


def validate_csv(path):
    from kaggle_workflow import validate_submission
    return validate_submission(path, ROOT/'data/raw/test.csv', expected_rows=4277)


def prepare(decision_path):
    evidence = decision_evidence(decision_path)
    decision = evidence['decision']
    module, proposal = decision['module'], decision['proposal']
    name, run, stage, budget = SPECS[module]
    out, report = ROOT/'artifacts'/proposal, ROOT/'reports'/proposal
    if out.exists() or report.exists():
        raise RuntimeError('Proposal exists; no overwrite or implicit resume.')
    out.mkdir(parents=True); report.mkdir(parents=True)
    started = time.monotonic()
    candidate = model = restored = None
    previous_handler = None
    try:
        candidate = importlib.import_module(module)
        if candidate.NAME != name:
            raise RuntimeError('Candidate module identity changed.')
        candidate.configure_runtime(out/'fit_logs', stage=stage)
        candidate.assert_frozen(ROOT/'artifacts'/run)
        def timeout(*_):
            raise TimeoutError('Complete fit/export/reload time budget exceeded.')
        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, budget)
        environment = candidate.mitra.environment_lock()
        (out/'environment.lock.txt').write_text(environment)
        environment_sha = sha(out/'environment.lock.txt')
        for filename in evidence['sources']:
            target = out/'source'/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/filename, target)
        write(out/'preparation_protocol.json', dict(**evidence,
            environment_sha256=environment_sha, budget_seconds=budget, stage=stage,
            original_failed_gates_unchanged=True, sources_frozen_before_full_labels=True))
        import iterate
        train, test, matrices, _ = iterate.dataset()
        if len(train) != 8693 or len(test) != 4277:
            raise RuntimeError('Full-data row counts differ.')
        a, b = np.arange(8693), np.arange(8693, 12970)
        first, model = candidate.fit(name, matrices, a, train.Transported.astype(int).to_numpy(), b, iterate.SEEDS[0])
        first = common.probability_vector(first, 4277)
        if len(model.labels) != 8693:
            raise RuntimeError('Complete training context missing.')
        joblib.dump(model, out/'model.joblib')
        write(out/'fit_metadata.json', model.metadata)
        common.release(candidate, model)
        model = None
        gc.collect(); candidate.mitra.release_memory()
        restored = joblib.load(out/'model.joblib')
        restored.device = 'mps'
        kwargs = dict(prediction_ids=b, log_path=out/'reload_log.json',
                      prediction_path=out/'reload_views.npz') if module == 'mitra_mirror_ensemble' else {}
        result = np.asarray(restored.predict_proba(matrices[False][0].iloc[b], **kwargs))
        if result.shape != (4277, 2):
            raise RuntimeError('Full reload class-probability shape differs.')
        difference = common.check_reload(first, result[:, 1])
        write(out/'reload_metadata.json', restored.metadata)
        common.release(candidate, restored)
        restored = None
        gc.collect(); candidate.mitra.release_memory()
        pending = out/'pending'; pending.mkdir()
        pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=first >= .5)).to_csv(pending/'submission.csv', index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId, probability=first,
                          reload_probability=result[:, 1])).to_csv(pending/'test_probabilities.csv', index=False)
        check = validate_csv(pending/'submission.csv')
        if duplicate_attempts(check['sha256']):
            raise RuntimeError('Identical predictions already have an online attempt.')
        candidate.assert_frozen(ROOT/'artifacts'/run)
        after = decision_evidence(decision_path)
        # Unrelated root commits may occur during a long fit. Every bound file
        # must remain unchanged; provenance retains the initial commit.
        if {k:v for k,v in after.items() if k != 'committed_at'} != {k:v for k,v in evidence.items() if k != 'committed_at'}:
            raise RuntimeError('Decision or source evidence changed during preparation.')
        if hashlib.sha256(candidate.mitra.environment_lock().encode()).hexdigest() != environment_sha:
            raise RuntimeError('Runtime environment changed.')
        selection = dict(version=name, source_run=run, components=[name],
            stage='explicit_autonomous_exploration', A_eligible=decision['prior_A_eligible'],
            S_supported=False, screen_accuracy=decision['screen_accuracy'], threshold=.5)
        bound = dict(status='bound_single_upload_authorization', competition='spaceship-titanic',
            batch_id=BATCH, source_authorization_sha256=evidence['authorization_sha256'],
            proposal=proposal, version=name, file_sha256=check['sha256'],
            model_sha256=sha(out/'model.joblib'), maximum_upload_attempts_for_this_file=1,
            desktop_reminder_required=True, decision_sha256=evidence['decision_sha256'],
            decision_path=evidence['decision_path'], source_protocol_sha256=decision['source_protocol_sha256'],
            prepared_from_commit=evidence['committed_at'], no_duplicate_or_blind_retry_upload=True)
        manifest = dict(**check, status='complete', batch_id=BATCH, selection=selection,
            fit_rows=8693, threshold=.5, export_reload_inference_verified=True,
            reload_rows=4277, reload_max_absolute_difference=difference,
            reload_threshold_decisions_equal=True, model_sha256=bound['model_sha256'],
            authorization=bound, original_failed_gates_unchanged=True,
            source_hashes=evidence['sources'], environment_sha256=environment_sha,
            provenance=dict(source_hashes=evidence['sources'], decision_path=evidence['decision_path'],
                decision_sha256=evidence['decision_sha256'], evidence_files=evidence['evidence_files'],
                source_protocol_sha256=decision['source_protocol_sha256'],
                prepared_from_commit=evidence['committed_at']),
            seconds=time.monotonic()-started, completed_at_utc=datetime.now(timezone.utc).isoformat())
        for filename in ['submission.csv', 'test_probabilities.csv']:
            (pending/filename).replace(out/filename)
        pending.rmdir()
        for filename, value in [('selection.json',selection), ('authorization.json',bound), ('manifest.json',manifest)]:
            write(out/filename,value); write(report/filename,value)
        print(f'Prepared {proposal}: {check["rows"]} rows; full reload difference {difference}.', flush=True)
        return manifest
    except BaseException as error:
        for directory in [out, report]:
            for filename in ['manifest.json', 'authorization.json', 'selection.json', 'submission.csv']:
                (directory/filename).unlink(missing_ok=True)
            write(directory/'failure.json', dict(status='failed_incomplete_not_upload_eligible',
                candidate=name, error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started))
        raise
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous_handler)
        if candidate is not None:
            common.release(candidate,model); common.release(candidate,restored)
            candidate.mitra.DEADLINE = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', required=True)
    prepare(parser.parse_args().decision)
