"""Fixed last-block Mitra adaptation: honest inner step selection and fresh refit.

Screening and the separate top5 validator evaluate the complete procedure.
No online submission is performed here. Existing run038 sources stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import iterate
BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
import mitra_candidate as mitra
import mitra_lastblock as core

ROOT = Path(__file__).resolve().parent
NAME = 'v75_Mitra_lastblock50'
CONTROL = 'v71_Mitra2_zeroshot'
SETTINGS = dict(max_steps=50, inner_fraction=.2, training_microbatch=128,
    query_chunk=2318, learning_rate=3e-6, weight_decay=.3,
    device='mps', mps_memory_fraction=.65, threshold=.5,
    trainable='layers.11.*,final_layer_norm.*,final_layer.*',
    validation_prefix_cache='CPU, one exact inner split only',
    final_probability_path='original_MitraPredictor_including_step0',
    checkpoint_sha256=mitra.WEIGHT_SHA,
    screen_budget_seconds=14400, stability_budget_seconds=43200, export_budget_seconds=7200,
    budgets=dict(screen=14400, validate=43200, export=7200))
OWN_SOURCES = ['mitra_lastblock_candidate.py', 'test_mitra_lastblock_candidate.py',
    'mitra_lastblock.py', 'test_mitra_lastblock.py', 'top5_validate.py',
    'test_top5_validate.py', 'audit_top5_references.py', 'test_audit_top5_references.py',
    'reports/top5_strategy_review.json', 'reports/top5_validation_review.json',
    'reports/top5_authorization.json']
RUN_OUT = None
ACTIVE_RUNTIME = None


def write_json(path, value):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(path)


@torch.no_grad()
def cached_validation_probabilities(model, cached, device):
    """Frozen block-entry cache only; trainable support LN/KV are recomputed."""
    support_cpu, query_cpu = cached
    support = support_cpu.to(device)
    probabilities = []
    for start in range(0, query_cpu.shape[1], SETTINGS['training_microbatch']):
        mitra.check_deadline()
        query = query_cpu[:, start:start+SETTINGS['training_microbatch']].to(device)
        probabilities.append(core.query_tail(model, support, query)[:, :2].softmax(-1).cpu().numpy())
    result = np.concatenate(probabilities)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite validation probabilities.')
    return result


def episode_update(model, optimizer, data, labels, y, ids, rng, step):
    mitra.check_deadline()
    si, qi = mitra.official_episode_indices(y, ids, rng)
    s, q = torch.as_tensor(si, device=data.device), torch.as_tensor(qi, device=data.device)
    return core.optimizer_step(model, optimizer, data[s], labels[s], data[q], labels[q],
                               ids[si], ids[qi], step, SETTINGS['training_microbatch'])


def fit_lastblock(x, y, ids, seed, device='mps', log_path=None, max_steps=50):
    """Only training X/y/IDs are accepted; max_steps override is for CPU tests."""
    y = mitra.binary_labels(y, len(x))
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) != len(y) or len(np.unique(ids)) != len(ids) or max_steps < 0:
        raise ValueError('Unique aligned training IDs and nonnegative max_steps required.')
    started = time.monotonic()
    record = dict(status='incomplete', version=NAME, seed=int(seed),
        training_rows=len(y), train_ids_sha256=mitra.ids_hash(ids),
        checkpoint_sha256=mitra.WEIGHT_SHA, max_steps=max_steps,
        inner_history=[], refit_history=[], selected_steps=0,
        selection_loss='binary_log_loss', training_loss='native_10_class_CE',
        selection_weights_discarded=True, training_episode_cache=False,
        final_prediction_path=SETTINGS['final_probability_path'])
    write_json(log_path, record)
    model = optimizer = data = labels = cached = valid_data = s = q = None
    try:
        inner, valid = train_test_split(np.arange(len(y)), test_size=SETTINGS['inner_fraction'],
                                        stratify=y, random_state=seed)
        core.assert_disjoint(ids[inner], ids[valid], len(inner), len(valid))
        record.update(inner_train_ids_sha256=mitra.ids_hash(ids[inner]),
            inner_validation_ids_sha256=mitra.ids_hash(ids[valid]),
            inner_training_rows=len(inner), inner_validation_rows=len(valid))
        mitra.seed_everything(seed, device)
        prep = mitra.MitraPreprocessor().fit(x.iloc[inner], y[inner], seed)
        model, config = mitra.load_initial_model(mitra.CHECKPOINT_DIR, device)
        core.configure(model)
        prefix_hash = core.frozen_digest(model)
        data = torch.as_tensor(prep.transform(x.iloc[inner]), device=device)
        labels = torch.as_tensor(y[inner], device=device)
        valid_data = torch.as_tensor(prep.transform(x.iloc[valid]), device=device)
        tick = time.monotonic()
        s, q = core.frozen_prefix(model, data, labels, valid_data)
        cached = (s.cpu(), q.cpu())
        s = q = valid_data = None
        mitra.release_memory()
        record.update(validation_prefix_cache_seconds=time.monotonic()-tick,
            validation_prefix_cache_bytes=sum(t.numel()*t.element_size() for t in cached),
            validation_prefix_cache_device=[str(t.device) for t in cached])
        optimizer = mitra.optimizer_for(model)
        rng = np.random.RandomState(seed)
        best_loss, selected = float('inf'), 0
        for step in range(max_steps+1):
            mitra.check_deadline()
            tick = time.monotonic()
            row = (episode_update(model, optimizer, data, labels, y[inner], ids[inner], rng, step)
                   if step else dict(step=0, optimizer_steps=0, learning_rate=None))
            model.eval()
            probability = cached_validation_probabilities(model, cached, device)
            loss = float(log_loss(y[valid], probability, labels=[0, 1]))
            if not np.isfinite(loss):
                raise FloatingPointError('Nonfinite inner logloss; no skipped evaluations.')
            row.update(validation_log_loss=loss, seconds=time.monotonic()-tick)
            record['inner_history'].append(row)
            if loss < best_loss:  # Strict: earliest step wins exact ties.
                best_loss, selected = loss, step
            record.update(selected_steps=selected, best_inner_log_loss=best_loss)
            write_json(log_path, record)
            print(f'{NAME} inner step={step}/{max_steps} logloss={loss:.8f} selected={selected}', flush=True)
        if core.frozen_digest(model) != prefix_hash:
            raise RuntimeError('Frozen prefix changed during selection.')
        record.update(selection_at_zero=selected == 0, selection_at_cap=selected == max_steps,
            inner_optimizer_steps=max_steps, inner_resources=mitra.snapshot(device),
            frozen_prefix_unchanged_selection=True)
        # Only selected count transfers. No selected weights/preprocessing/cache.
        model = optimizer = data = labels = cached = prep = None
        mitra.release_memory()
        mitra.check_deadline()
        mitra.seed_everything(seed, device)
        prep = mitra.MitraPreprocessor().fit(x, y, seed)
        model, config = mitra.load_initial_model(mitra.CHECKPOINT_DIR, device)
        core.configure(model)
        prefix_hash = core.frozen_digest(model)
        support = prep.transform(x)
        data, labels = torch.as_tensor(support, device=device), torch.as_tensor(y, device=device)
        rng = np.random.RandomState(seed)
        optimizer = mitra.optimizer_for(model) if selected else None
        for step in range(1, selected+1):
            row = episode_update(model, optimizer, data, labels, y, ids, rng, step)
            record['refit_history'].append(row)
            write_json(log_path, record)
            print(f'{NAME} fresh refit step={step}/{selected} CE={row["training_full_head_ce"]:.8f}', flush=True)
        if core.frozen_digest(model) != prefix_hash:
            raise RuntimeError('Frozen prefix changed during fresh refit.')
        record.update(status='fit_complete', refit_optimizer_steps=selected,
            fresh_refit_from_original_checkpoint=True, frozen_prefix_unchanged_refit=True,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            resources_after_refit=mitra.snapshot(device), seconds=time.monotonic()-started)
        artifact = mitra.MitraPredictor(prep, config, model.state_dict(), support, y,
            device=device, metadata=record, query_chunk=SETTINGS['query_chunk'])
        write_json(log_path, record)
        return artifact
    except Exception as error:
        record.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        write_json(log_path, record)
        raise
    finally:
        model = optimizer = data = labels = cached = valid_data = s = q = None
        mitra.release_memory()


def configure_runtime(out, stage='screen'):
    """Exact caller-owned log destination, without creating any directories."""
    global RUN_OUT, ACTIVE_RUNTIME
    RUN_OUT = Path(out)
    phase = 'validate' if stage == 'stability' else stage
    if phase not in SETTINGS['budgets']:
        raise ValueError(stage)
    mitra.DEADLINE = time.monotonic()+SETTINGS['budgets'][phase]
    ACTIVE_RUNTIME = (RUN_OUT, phase)


def start_runtime(stage):
    global ACTIVE_RUNTIME
    if stage not in SETTINGS['budgets']:
        raise ValueError(stage)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS backend unavailable; no CPU fallback.')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
    mitra.DEADLINE = time.monotonic()+SETTINGS['budgets'][stage]
    ACTIVE_RUNTIME = (RUN_OUT, stage)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in [NAME, CONTROL]:
        raise ValueError(name)
    if RUN_OUT is None:
        raise RuntimeError('Call configure_runtime with the exact log destination.')
    if ACTIVE_RUNTIME is None or ACTIVE_RUNTIME[0] != RUN_OUT:
        start_runtime('validate' if RUN_OUT.name == 'stability' else 'screen')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS backend unavailable; no CPU fallback.')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
    a, b = np.asarray(train_idx, dtype=np.int64), np.asarray(predict_idx, dtype=np.int64)
    core.assert_disjoint(a, b, len(a), len(b))
    x = matrices[False][0]
    key = mitra.ids_hash(a)[:12]
    log = RUN_OUT/f'{name}_seed{seed}_{key}.json'
    if log.exists():
        raise RuntimeError('Fit log already exists; no silent resume or overwrite.')
    if name == NAME:
        artifact = fit_lastblock(x.iloc[a], y, a, seed, SETTINGS['device'], log,
                                 SETTINGS['max_steps'])
    else:
        artifact = mitra.fit_mitra(x.iloc[a], y, a, seed, SETTINGS['device'], 'zero-shot',
            query_chunk=SETTINGS['query_chunk'], log_path=log)
    try:
        probability = artifact.predict_proba(x.iloc[b])[:, 1]
        mitra.check_deadline()
        if not np.isfinite(probability).all() or len(probability) != len(b):
            raise FloatingPointError('Incomplete/nonfinite final prediction.')
        artifact.metadata.update(prediction_ids_sha256=mitra.ids_hash(b),
            prediction_rows=len(b), final_prediction_path='original_MitraPredictor')
        write_json(log, artifact.metadata)
        return probability, artifact
    except Exception as error:
        artifact.metadata.update(status='failed_prediction', error_type=type(error).__name__, error=str(error))
        write_json(log, artifact.metadata)
        raise
    finally:
        artifact.release()


def verify_reference():
    history = ROOT/'artifacts/run_038'
    reference = json.loads((history/'protocol.json').read_text())
    for name, expected in reference['sources'].items():
        if mitra.sha256(ROOT/name) != expected:
            raise RuntimeError(f'Historical Mitra source changed: {name}')
    lock = mitra.environment_lock()
    if hashlib.sha256(lock.encode()).hexdigest() != reference['mitra_environment_sha256']:
        raise RuntimeError('Original Mitra environment differs.')
    mitra.verify_checkpoint(mitra.CHECKPOINT_DIR)
    import top5_validate
    top5_validate.check_policy()
    return reference, lock


def record_protocol(out):
    path = out/'protocol.json'
    record = json.loads(path.read_text())
    if 'lastblock_policy' in record:
        return
    reference, lock = verify_reference()
    for name in sorted(set(reference['sources']) | set(OWN_SOURCES)):
        record['sources'][name] = mitra.sha256(ROOT/name)
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    (out/'lastblock_environment.lock.txt').write_text(lock)
    record.update(lastblock_policy=dict(settings=SETTINGS,
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        selection='inner80/20, steps0..50 minbinaryLL earliestties; freshfullrefit',
        persistence='CPU state/preprocessor/fulltraining context; original architecture inference',
        eligibility='Separate frozen top5 A/S policy. No automatic historical83gate override.'))
    write_json(path, record)


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # Includes validator/audit/policy before first baseline.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:
        return
    if iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE or BASE_FIT.__module__ != 'iterate':
        raise RuntimeError('Use a fresh process, not another patched wrapper.')
    iterate.CONFIGS = {iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
        CONTROL: dict(context=False, model='mitra_zeroshot', **SETTINGS),
        NAME: dict(context=False, model='mitra_lastblock', **SETTINGS)}
    iterate.BLEND = []
    iterate.fit, iterate.evaluate = fit, evaluate


def assert_frozen(out):
    configure()
    iterate.assert_frozen(out)
    record = json.loads((Path(out)/'protocol.json').read_text())
    expected = record['lastblock_policy']['environment_sha256']
    if record['lastblock_policy']['settings'] != SETTINGS:
        raise RuntimeError('Frozen last-block recipe changed.')
    if (hashlib.sha256(mitra.environment_lock().encode()).hexdigest() != expected
            or mitra.sha256(Path(out)/'lastblock_environment.lock.txt') != expected):
        raise RuntimeError('Frozen last-block environment changed.')
    verify_reference()


def export(out):
    assert_frozen(out)
    eligibility = json.loads((out/'stability/eligibility.json').read_text())
    if (eligibility.get('status') != 'complete' or not eligibility.get('A_eligible')
            or eligibility.get('candidate') != NAME
            or eligibility.get('original_screen_protocol_sha256') != mitra.sha256(out/'protocol.json')
            or eligibility.get('validation_policy_sha256') != mitra.sha256(ROOT/'reports/top5_validation_review.json')
            or eligibility.get('authorization_sha256') != mitra.sha256(ROOT/'reports/top5_authorization.json')):
        raise RuntimeError('Complete source-matched A/S eligibility is required.')
    output = out/'export_lastblock'
    output.mkdir(exist_ok=False)
    configure_runtime(output, stage='export')
    start_runtime('export')
    train, test, matrices, _ = iterate.dataset()
    probability, model = fit(NAME, matrices, np.arange(len(train)), train.Transported.astype(int),
        np.arange(len(train), len(train)+len(test)), iterate.SEEDS[0])
    assert_frozen(out)
    joblib.dump(model, output/(NAME+'.joblib'))
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=probability)).to_csv(output/'test_probabilities.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=probability>=.5)).to_csv(output/'submission.csv', index=False)
    write_json(output/'export.json', dict(status='complete', candidate=NAME,
        model_metadata=model.metadata, uploaded=False, eligibility_sha256=mitra.sha256(out/'stability/eligibility.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen', 'export'])
    parser.add_argument('--run', default='run_041')
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.run
    if args.stage == 'screen' and out.exists():
        raise RuntimeError('Screen destination exists; refuse overwrite.')
    if args.stage == 'export' and (out/'export_lastblock').exists():
        raise RuntimeError('Export destination exists; refuse overwrite before cleanup scope.')
    configure()
    configure_runtime(out)
    start_runtime(args.stage)
    def timeout(*_):
        raise TimeoutError('Frozen last-block stage budget exceeded; no partial score.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, SETTINGS['budgets'][args.stage])
    try:
        if args.stage == 'screen':
            iterate.screen(out)
            assert_frozen(out)
            selection = json.loads((out/'selection.json').read_text())
            selection.update(stage='screen_only_pending_separate_A_S_review', automatic_eligibility=False)
            write_json(out/'selection.json', selection)
        else:
            export(out)
    except Exception as error:
        if out.exists():
            write_json(out/f'{args.stage}_failure.json', dict(status='failed_inconclusive', error=repr(error)))
            if args.stage == 'screen':
                (out/'selection.json').unlink(missing_ok=True)
                if (out/'screen_results.csv').exists():
                    (out/'screen_results.csv').replace(out/'screen_results.partial.csv')
            else:
                for name in ['submission.csv', 'test_probabilities.csv', 'export.json']:
                    (out/'export_lastblock'/name).unlink(missing_ok=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        mitra.DEADLINE = None


if __name__ == '__main__':
    main()
