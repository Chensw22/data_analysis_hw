"""Fixed v81 full-parameter episode screen; no export or submission entry point.

Original development rows and three folds are reused. This is adaptive model
screening, not independent confirmation. Earlier failed candidates stay frozen.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

import iterate
import mitra_candidate as mitra
import mitra_full_episode as core

ROOT = Path(__file__).resolve().parent
BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
NAME, CONTROL = 'v81_Mitra_full_episode50', 'v71_Mitra2_zeroshot'
REVIEW = 'reports/top5_full_episode_review.json'
RESOURCE_ROOT = 'reports/mitra_full_episode_resources'
SETTINGS = dict(**core.SETTINGS, device='mps', mps_memory_fraction=.65,
    threads=4, threshold=.5, trainable='all original parameters',
    budgets=dict(screen=21600, validate=64800, export=10800),
    training_loss='native ten-class cross entropy', validation_prefix_cache=False)
OWN_SOURCES = ['mitra_full_episode_candidate.py', 'test_mitra_full_episode_candidate.py',
    'mitra_full_episode.py', 'test_mitra_full_episode.py',
    'mitra_full_episode_probe.py', 'test_mitra_full_episode_probe.py', REVIEW,
    RESOURCE_ROOT+'/freeze.json', RESOURCE_ROOT+'/result.json', RESOURCE_ROOT+'/requirements.lock']
RUN_OUT = ACTIVE_RUNTIME = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def verify_reference():
    reference = json.loads((ROOT/'artifacts/run_038/protocol.json').read_text())
    freeze = json.loads((ROOT/RESOURCE_ROOT/'freeze.json').read_text())
    resource = json.loads((ROOT/RESOURCE_ROOT/'result.json').read_text())
    review = json.loads((ROOT/REVIEW).read_text())
    for saved in (reference, freeze):
        for relative, expected in saved['sources'].items():
            if mitra.sha256(ROOT/relative) != expected:
                raise RuntimeError('Historical source changed: '+relative)
    lock = mitra.environment_lock()
    env_sha = hashlib.sha256(lock.encode()).hexdigest()
    if not (env_sha == reference['mitra_environment_sha256'] == freeze['environment_sha256']
            == mitra.sha256(ROOT/RESOURCE_ROOT/'requirements.lock')):
        raise RuntimeError('Use the unchanged resource-probed Mitra environment.')
    mitra.verify_checkpoint(mitra.CHECKPOINT_DIR)
    if (freeze['checkpoint_sha256'] != mitra.WEIGHT_SHA
            or freeze['recipe'] != resource['recipe'] or resource['status'] != 'complete'
            or not resource['synthetic_only'] or resource['accuracy_computed']
            or resource['reload_max_abs_difference'] > 2e-6
            or resource['recipe']['support_rows'] != SETTINGS['episode_support']
            or resource['recipe']['query_rows'] != SETTINGS['episode_query']
            or resource['recipe']['trainable_parameters'] != SETTINGS['trainable_parameters']
            or resource['recipe']['mps_memory_fraction'] != SETTINGS['mps_memory_fraction']
            or resource['recipe']['query_chunk'] != SETTINGS['query_chunk']
            or len(resource['updates']) != 2
            or [r['optimizer_state_steps'] for r in resource['updates']] != [[1], [2]]
            or not all(all(r['updated_blocks'].values()) for r in resource['updates'])
            or resource['validation']['support_rows'] != 3708
            or resource['final_inference']['support_rows'] != 8693):
        raise RuntimeError('A complete matching full-parameter resource probe is required.')
    if review['candidate'] != NAME or review['settings'] != SETTINGS:
        raise RuntimeError('Reviewed full-parameter recipe changed.')
    if review['weekly_reserve_percent'] != 30 or review['automatic_export']:
        raise RuntimeError('Unexpected resource or export policy.')
    if any(SETTINGS[k] != v for k, v in core.SETTINGS.items()):
        raise RuntimeError('Wrapper and training-core recipes differ.')
    return reference, lock


def configure_runtime(out, stage='screen'):
    global RUN_OUT, ACTIVE_RUNTIME
    stage = 'validate' if stage == 'stability' else stage
    if stage not in SETTINGS['budgets']:
        raise ValueError(stage)
    RUN_OUT = Path(out)
    ACTIVE_RUNTIME = (RUN_OUT, stage)
    mitra.DEADLINE = time.monotonic()+SETTINGS['budgets'][stage]


def start_runtime(stage):
    if RUN_OUT is None or stage not in SETTINGS['budgets']:
        raise RuntimeError('Configure the exact output and stage before starting.')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS backend unavailable; no CPU fallback.')
    torch.set_num_threads(SETTINGS['threads'])
    torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
    configure_runtime(RUN_OUT, stage)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in (NAME, CONTROL):
        raise ValueError(name)
    if RUN_OUT is None or ACTIVE_RUNTIME is None or ACTIVE_RUNTIME[0] != RUN_OUT:
        raise RuntimeError('Configure the exact runtime before fitting.')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS backend unavailable; no CPU fallback.')
    torch.set_num_threads(SETTINGS['threads'])
    torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
    mitra.check_deadline()
    a, b = np.asarray(train_idx, dtype=np.int64), np.asarray(predict_idx, dtype=np.int64)
    x = matrices[False][0]
    if (a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b)
            or len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b)
            or np.intersect1d(a, b).size or min(a.min(), b.min()) < 0
            or max(a.max(), b.max()) >= len(x)):
        raise ValueError('Unique disjoint in-range training/query IDs required.')
    y = mitra.binary_labels(y, len(a))
    log = RUN_OUT/f'{name}_seed{seed}_{mitra.ids_hash(a)[:12]}.json'
    if log.exists():
        raise RuntimeError('Existing fit log: no silent resume or overwrite.')
    if name == NAME:
        artifact = core.fit_full_episode(x.iloc[a], y, a, seed,
            device=SETTINGS['device'], log_path=log, max_steps=SETTINGS['max_steps'])
    else:
        artifact = mitra.fit_mitra(x.iloc[a], y, a, seed, SETTINGS['device'], 'zero-shot',
            query_chunk=SETTINGS['query_chunk'], log_path=log)
    try:
        probability = artifact.predict_proba(x.iloc[b])[:, 1]
        mitra.check_deadline()
        if probability.shape != (len(b),) or not np.isfinite(probability).all():
            raise FloatingPointError('Incomplete/nonfinite final prediction.')
        artifact.metadata.update(prediction_ids_sha256=mitra.ids_hash(b),
            prediction_rows=len(b), final_prediction_path='original_MitraPredictor',
            automatic_further_extension_allowed=False)
        write_json(log, artifact.metadata)
        return probability, artifact
    except Exception as error:
        artifact.metadata.update(status='failed_prediction', error_type=type(error).__name__, error=str(error))
        write_json(log, artifact.metadata)
        raise
    finally:
        artifact.release()


def record_protocol(out):
    out = Path(out)
    path = out/'protocol.json'
    record = json.loads(path.read_text())
    if 'full_episode_policy' in record:
        assert_frozen(out)
        return
    reference, lock = verify_reference()
    for relative in sorted(set(reference['sources']) | set(OWN_SOURCES)):
        record['sources'][relative] = mitra.sha256(ROOT/relative)
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    (out/'full_episode_environment.lock.txt').write_text(lock)
    record['full_episode_policy'] = dict(settings=copy.deepcopy(SETTINGS),
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        resource_freeze_sha256=mitra.sha256(ROOT/RESOURCE_ROOT/'freeze.json'),
        resource_result_sha256=mitra.sha256(ROOT/RESOURCE_ROOT/'result.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        weekly_reserve_percent=30, automatic_eligibility=False,
        assessment='Adaptive reused-development screen only; no stable-superiority or upload decision.',
        stopping_rule='Fixed horizon50; no automatic100/200 extension or reduced inference context.')
    write_json(path, record)


def evaluate(name, matrices, train, dev, seed, out):
    if len(dev) != 6954 or seed != iterate.SEEDS[0]:
        raise ValueError('Screen requires original6954development rows and original seed.')
    record_protocol(out)  # Freeze all sources/resources before the first baseline.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:
        return
    if (iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE
            or BASE_FIT.__module__ != 'iterate'):
        raise RuntimeError('Use a fresh process without another patched candidate.')
    iterate.CONFIGS = {iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
        CONTROL: dict(context=False, model='mitra_zeroshot', **SETTINGS),
        NAME: dict(context=False, model='mitra_full_episode', **SETTINGS)}
    iterate.BLEND = []
    iterate.fit, iterate.evaluate = fit, evaluate


def assert_frozen(out):
    configure()
    out = Path(out)
    iterate.assert_frozen(out)
    record = json.loads((out/'protocol.json').read_text())['full_episode_policy']
    lock_sha = hashlib.sha256(mitra.environment_lock().encode()).hexdigest()
    if (record['settings'] != SETTINGS or record['environment_sha256'] != lock_sha
            or mitra.sha256(out/'full_episode_environment.lock.txt') != lock_sha):
        raise RuntimeError('Frozen full-parameter recipe/environment changed.')
    verify_reference()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen'])
    parser.add_argument('--run', default='run_046')
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.run
    if out.exists():
        raise RuntimeError('Screen destination exists; refuse overwrite.')
    verify_reference()
    configure()
    configure_runtime(out)
    start_runtime('screen')
    def timeout(*_):
        raise TimeoutError('Frozen full-parameter screen budget exceeded; no partial score.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, SETTINGS['budgets']['screen'])
    try:
        iterate.screen(out)
        assert_frozen(out)
        selection = json.loads((out/'selection.json').read_text())
        selection.update(stage='screen_only_pending_separate_review', automatic_eligibility=False)
        write_json(out/'selection.json', selection)
    except Exception as error:
        if out.exists():
            write_json(out/'screen_failure.json', dict(status='failed_inconclusive', error=repr(error)))
            (out/'selection.json').unlink(missing_ok=True)
            if (out/'screen_results.csv').exists():
                (out/'screen_results.csv').replace(out/'screen_results.partial.csv')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        mitra.DEADLINE = None


if __name__ == '__main__':
    main()
