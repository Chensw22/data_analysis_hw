"""Export the unchanged EXAONE solo recipe under explicit online-first authority.

No CV, quota gate, upload, alternate recipe, or retry. Run in .venv-exaone.
"""
import os
import sys

# This must precede every numerical import, including transitive sklearn imports.
ENTRY_FLAG = os.environ.get('KMP_DUPLICATE_LIB_OK')
if ENTRY_FLAG not in (None, 'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required; existing override rejected.')
if any(name in sys.modules for name in ('numpy', 'torch', 'sklearn')):
    raise RuntimeError('Start this exporter in a fresh process before numerical imports.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
VERSION = 'EXAONE_v91_CAST_online_first'
PROPOSAL = 'proposal_online_first_exaone_fixed'
POLICY = 'reports/online_iteration_20261117/online_first_policy.json'
RESOURCE = 'reports/exaone_resources/mps/result.json'
BUDGET = 3600


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def require_false():
    if os.environ.get('KMP_DUPLICATE_LIB_OK') != 'FALSE':
        raise RuntimeError('Explicit FALSE runtime changed.')


def committed(names):
    names = sorted(set(names))
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *names], cwd=ROOT).decode()
    if set(filter(None, tracked.split('\0'))) != set(names):
        raise RuntimeError('Commit every exporter/policy/source dependency first.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *names], cwd=ROOT).strip():
        raise RuntimeError('Export dependencies differ from their committed versions.')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def coverage(metadata):
    from collections import Counter
    record = metadata['last_prediction']
    expected = Counter({1024: 32, 181: 8})
    records = record['forward_records']
    if (record['query_rows'] != 4277 or record['full_support_rows'] != 8693 or record['views'] != 8
            or Counter(r['query_rows'] for r in records) != expected
            or any(r['support_rows'] != 8693 or r['views'] != 1 or r['input_dtype'] != 'float32' for r in records)):
        raise RuntimeError('Full 8693 support / 4277 query / eight views required.')


def check_context(model, y, ids, core, np):
    if (model.seed != 0 or model.support.shape != (8693, 13)
            or not np.array_equal(model.ids, ids) or not np.array_equal(model.labels, y)
            or model.metadata['settings'] != core.SETTINGS
            or model.metadata['checkpoint_sha256'] != core.WEIGHT_SHA
            or model.metadata['training_rows'] != 8693):
        raise RuntimeError('Portable context, settings or checkpoint differs.')


def prepare():
    out, report = ROOT/'artifacts'/PROPOSAL, ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():
        raise RuntimeError('Existing proposal is immutable; no overwrite or resume.')
    out.mkdir(parents=True); report.mkdir(parents=True)
    started = time.monotonic()
    model = restored = None
    def timeout(*_):
        raise TimeoutError('Complete export and reload exceeded 3600 seconds.')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    try:
        require_false()
        import numpy as np
        import pandas as pd
        import joblib
        import exaone_candidate as core
        import exaone_resource_probe as resource
        import exaone_validation_data as data
        import kaggle_workflow
        require_false()
        policy = read(ROOT/POLICY)
        if (policy.get('competition') != 'spaceship-titanic'
                or policy.get('allow_full_export') is not True
                or VERSION not in policy.get('versions', [])):
            raise RuntimeError('Committed online-first policy must explicitly authorize this solo export.')
        contract, lock = resource.contract(require_committed=True)  # No quota or CV functions called.
        previous = read(ROOT/RESOURCE)
        if (previous.get('status') != 'complete' or previous.get('resource_qualification') is not True
                or any(previous.get(k) != contract[k] for k in ('sources', 'environment_sha256', 'checkpoint_sha256', 'checkpoint_bytes', 'settings'))):
            raise RuntimeError('Existing complete resource recipe/environment changed.')
        own = [Path(__file__).name, 'exaone_validation_data.py', 'audit_top5_references.py',
               'three_family_mean_screen.py', 'prior_adapt_mean_screen.py', data.PROOF,
               'kaggle_workflow.py', POLICY, RESOURCE]
        source_commit = committed(list(contract['sources']) + own)
        sources = {**contract['sources'], **{name: sha(ROOT/name) for name in own}}
        raw_hashes = {f'data/raw/{name}.csv': sha(ROOT/f'data/raw/{name}.csv') for name in ('train', 'test', 'sample_submission')}
        frozen = dict(version=VERSION, source_commit=source_commit, source_hashes=sources,
            resource_contract=contract, raw_hashes=raw_hashes, policy_sha256=sources[POLICY],
            settings=core.SETTINGS, budget_seconds=BUDGET, entry_flag=ENTRY_FLAG,
            runtime_flag='FALSE', full_labels_authorized=True, original_failed_gates_unchanged=True,
            interpretation='First solo EXAONE online export; v92 mixture failure remains unchanged; no local eligibility gate.')
        write(out/'protocol.json', frozen); write(report/'protocol.json', frozen)
        (out/'environment.lock.txt').write_text(lock)
        (report/'environment.lock.txt').write_text(lock)
        for name in sources:
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        core.DEADLINE = started + BUDGET
        if not core.torch.backends.mps.is_available():
            raise RuntimeError('Original MPS backend required; no fallback.')
        core.torch.set_num_threads(4)
        core.torch.mps.set_per_process_memory_fraction(.65)
        # No development split or historical holdout evaluation is performed.
        proof = read(ROOT/data.PROOF)
        for name, expected in proof['source_data_sha256'].items():
            if sha(ROOT/name) != expected:
                raise RuntimeError('Original feature-proof input changed: ' + name)
        train = pd.read_csv(ROOT/'data/raw/train.csv')
        test = pd.read_csv(ROOT/'data/raw/test.csv')
        sample = pd.read_csv(ROOT/'data/raw/sample_submission.csv')
        frame = data.compact_features(pd.concat([train.drop(columns='Transported'), test], ignore_index=True))
        if len(train) != 8693 or data.feature_hash(frame) != proof['compact_values_sha256']:
            raise RuntimeError('Exact original compact13 values required.')
        x, q = frame.iloc[:8693][core.COMPACT], frame.iloc[8693:][core.COMPACT]
        y, ids = train.Transported.astype(np.int64).to_numpy(), np.arange(8693, dtype=np.int64)
        if len(q) != 4277 or not test.PassengerId.equals(sample.PassengerId):
            raise RuntimeError('Test size/order differs.')
        require_false()
        model = core.fit_context(x, y, ids, seed=0, device='mps')
        check_context(model, y, ids, core, np)
        first = core.valid_probabilities(model.predict_proba(q), 4277).copy()
        coverage(model.metadata); require_false()
        write(out/'initial_metadata.json', model.metadata)
        joblib.dump(model, out/'model.joblib')
        model.release(); model = None; gc.collect(); core.torch.mps.empty_cache()
        restored = joblib.load(out/'model.joblib')
        if restored.device != 'cpu' or restored._engine is not None:
            raise RuntimeError('Saved artifact must have CPU context and no live engine.')
        check_context(restored, y, ids, core, np)
        restored.device = 'mps'
        again = core.valid_probabilities(restored.predict_proba(q), 4277)
        difference = resource.compare(first, again, 2e-6)
        coverage(restored.metadata); require_false()
        write(out/'reload_metadata.json', restored.metadata)
        restored.release(); restored = None; gc.collect(); core.torch.mps.empty_cache()
        require_false()
        pending = out/'pending'
        pending.mkdir()
        pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': first[:, 1] >= .5}).to_csv(pending/'submission.csv', index=False)
        pd.DataFrame({'PassengerId': test.PassengerId, 'probability': first[:, 1], 'reload_probability': again[:, 1]}).to_csv(pending/'test_probabilities.csv', index=False)
        check = kaggle_workflow.validate_submission(pending/'submission.csv', ROOT/'data/raw/test.csv')
        for attempt in (ROOT/'reports').rglob('online_attempts/*.json'):
            if read(attempt).get('file_sha256') == check['sha256']:
                raise RuntimeError('Identical CSV already has an online attempt.')
        core.check_deadline()
        resource.assert_frozen(contract)
        committed(sources)
        for name, digest in {**sources, **raw_hashes}.items():
            if sha(ROOT/name) != digest:
                raise RuntimeError('Export source/evidence/data changed: ' + name)
        selection = dict(version=VERSION, model_recipe=core.NAME, seed=0, threshold=.5,
            stage='explicit_online_first', A_eligible=None, S_supported=False,
            first_solo_online=True, old_v92_failure_unchanged=True)
        manifest = dict(**check, status='complete', version=VERSION, proposal=PROPOSAL,
            training_rows=8693, fit_rows=8693, threshold=.5, model_sha256=sha(out/'model.joblib'),
            export_reload_inference_verified=True, reload_rows=4277,
            reload_max_absolute_difference=difference, reload_threshold_decisions_equal=True,
            source_commit=source_commit, source_hashes=sources, selection=selection,
            policy_path=POLICY, policy_sha256=sources[POLICY], environment_sha256=contract['environment_sha256'],
            checkpoint_sha256=core.WEIGHT_SHA, runtime_flag='FALSE', seconds=time.monotonic()-started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), original_failed_gates_unchanged=True)
        for name in ('submission.csv', 'test_probabilities.csv'):
            (pending/name).replace(out/name)
        pending.rmdir()
        for name, value in [('selection.json', selection), ('manifest.json', manifest)]:
            write(out/name, value); write(report/name, value)
        print(json.dumps(dict(proposal=PROPOSAL, status='complete', rows=4277, reload_max_absolute_difference=difference)), flush=True)
        return manifest
    except BaseException as error:
        for directory in (out, report):
            for name in ('manifest.json', 'selection.json', 'submission.csv'):
                (directory/name).unlink(missing_ok=True)
            write(directory/'failure.json', dict(status='failed_not_upload_eligible', version=VERSION,
                error_type=type(error).__name__, error=str(error), seconds=time.monotonic()-started))
        if (out/'pending').exists():
            shutil.rmtree(out/'pending')
        raise
    finally:
        try:
            for active in (model, restored):
                if active is not None:
                    active.release()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
