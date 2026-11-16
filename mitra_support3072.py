"""One paired-support extension of v81; no data loading, screen or export CLI.

Selection/refit, native loss and backend remain v81. Only episode support grows
from1024 to3072 while query IDs, original support prefix and RNG calls agree.
"""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import mitra_candidate as base
import mitra_full_episode as original
import mitra_long_context_probe as probe

ROOT = Path(__file__).resolve().parent
SETTINGS = {**original.SETTINGS, 'episode_support':3072, 'sampler':probe.RECIPE['sampler']}
SOURCES = ['mitra_support3072.py', 'mitra_long_context_probe.py', 'mitra_full_episode.py', 'mitra_candidate.py']
fresh_model = original.fresh_model


def source_hashes():
    return {name:base.sha256(ROOT/name) for name in SOURCES}


def assert_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError('Training sources changed during the fixed fit.')


def episode_step(model, optimizer, data, targets, ids, rng, reference_rng, step, checkpoint):
    """Reuse the qualified update; cloned RNGs map local sampling to global IDs."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.shape != (len(data),) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique aligned global episode IDs required.')
    # Neither clone advances either live sampler. The qualified update performs
    # precisely the one actual permutation (plus its independent counterfactual).
    support, query = probe.paired_episode(len(data), copy.deepcopy(rng), copy.deepcopy(reference_rng))
    if np.intersect1d(ids[support], ids[query]).size:
        raise RuntimeError('Episode support/query global IDs overlap.')
    row = probe.update(model, optimizer, data, targets, rng, reference_rng, step, checkpoint)
    if (row['support_ids_sha256'] != base.ids_hash(support)
            or row['query_ids_sha256'] != base.ids_hash(query)):
        raise RuntimeError('Actual update differs from the cloned sampling counterfactual.')
    row.update(local_support_ids_sha256=row['support_ids_sha256'], local_query_ids_sha256=row['query_ids_sha256'],
               support_ids_sha256=base.ids_hash(ids[support]), query_ids_sha256=base.ids_hash(ids[query]),
               support_prefix_ids_sha256=base.ids_hash(ids[support[:1024]]),
               learning_rate=row['lr'], cumulative_optimizer_steps=step, finite_gradients=True)
    return row


def validation_prediction(model, optimizer, data, targets, valid_data):
    moments = probe.moment_identity(optimizer)
    steps = probe.optimizer_steps(optimizer)
    size = probe.original.optimizer_bytes(optimizer)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Gradients must be cleared before full-context validation.')
    probabilities = base.predict_arrays(model, data, targets, valid_data, chunk=2318)
    if (probabilities.shape != (len(valid_data), 2) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1., atol=2e-6, rtol=0)
            or any(p.grad is not None for p in model.parameters())
            or probe.moment_identity(optimizer) != moments or probe.optimizer_steps(optimizer) != steps
            or probe.original.optimizer_bytes(optimizer) != size):
        raise RuntimeError('Full-context probabilities or resident optimizer state changed.')
    return probabilities


def fit_support3072(x, y, ids, seed, device='mps', log_path=None, max_steps=50):
    if max_steps != 50:
        raise ValueError('Exactly the frozen50-step selection horizon is permitted.')
    y = base.binary_labels(y, len(x))
    raw_ids = np.asarray(ids)
    ids = raw_ids.astype(np.int64)
    if ids.shape != y.shape or not np.array_equal(raw_ids, ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique aligned integer training IDs required.')
    inner, valid = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    if len(inner) < 3328 or len(ids) < 3328:
        raise ValueError('Inner and refit pools must fit3072+256 distinct rows; no smaller-context rescue.')
    if np.intersect1d(ids[inner], ids[valid]).size:
        raise ValueError('Inner validation overlaps training IDs.')
    started = time.monotonic()
    frozen = source_hashes()
    environment_sha = hashlib.sha256(base.environment_lock().encode()).hexdigest()
    metadata = dict(status='incomplete', recipe=copy.deepcopy(SETTINGS), seed=int(seed), training_rows=len(y),
        training_ids_sha256=base.ids_hash(ids), inner_train_ids_sha256=base.ids_hash(ids[inner]),
        inner_validation_ids_sha256=base.ids_hash(ids[valid]), inner_training_rows=len(inner),
        inner_validation_rows=len(valid), inner_history=[], refit_history=[], selected_steps=0,
        checkpoint_sha256=base.WEIGHT_SHA, validation_support_order='fixed all inner-training rows',
        precision='MPS bf16 autocast with FP32 weights' if device == 'mps' else 'CPU FP32',
        source_hashes=frozen, environment_sha256=environment_sha,
        paired_reference='v81_Mitra_full_episode50', episode_global_ids_verified=True)
    model = optimizer = data = targets = valid_data = prep = rng = reference_rng = predictor = None
    phase = 'inner'
    def checkpoint(stage, resource):
        base.check_deadline(); assert_sources(frozen)
        metadata['current_stage'] = phase+':'+stage
        metadata['current_resources'] = resource
        base.save_training_log(log_path, metadata)
    try:
        base.check_deadline(); base.save_training_log(log_path, metadata)
        base.seed_everything(seed, device)
        prep = base.MitraPreprocessor().fit(x.iloc[inner], y[inner], seed)
        model, config = fresh_model(device)
        data = torch.as_tensor(prep.transform(x.iloc[inner]), device=device)
        targets = torch.as_tensor(y[inner], device=device)
        valid_data = torch.as_tensor(prep.transform(x.iloc[valid]), device=device)
        optimizer = base.optimizer_for(model)
        rng, reference_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        best_loss, selected_steps = float('inf'), 0
        for step in range(51):
            base.check_deadline(); assert_sources(frozen)
            tick = time.monotonic()
            row = episode_step(model, optimizer, data, targets, ids[inner], rng, reference_rng, step, checkpoint) if step else dict(step=0)
            row['stage'] = 'inner'
            checkpoint(f'step{step}:before_validation', probe.synchronized_snapshot(device))
            probabilities = validation_prediction(model, optimizer, data, targets, valid_data)
            score = float(log_loss(y[valid], probabilities, labels=[0, 1]))
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite inner binary logloss.')
            if score < best_loss:
                best_loss, selected_steps = score, step
            row.update(validation_log_loss=score, validation_support_rows=len(data),
                       validation_query_rows=len(valid_data), validation_full_support=True,
                       validation_optimizer_steps=probe.optimizer_steps(optimizer),
                       validation_optimizer_state_tensor_bytes=probe.original.optimizer_bytes(optimizer),
                       total_seconds=time.monotonic()-tick)
            checkpoint(f'step{step}:after_validation', probe.synchronized_snapshot(device))
            metadata['inner_history'].append(row)
            metadata.update(selected_steps=selected_steps, best_inner_log_loss=best_loss)
            base.save_training_log(log_path, metadata)
            if step % 10 == 0:
                print(f'Mitra support3072 inner step={step} selected={selected_steps}', flush=True)
        metadata.update(selection_at_cap=selected_steps == 50, inner_optimizer_steps=50,
                        inner_final_resources=probe.synchronized_snapshot(device))
        # Only the update count crosses this boundary. All fitted tensors, the
        # optimizer, preprocessing and both sampler streams are replaced.
        model = optimizer = data = targets = valid_data = prep = rng = reference_rng = None
        base.release_memory()
        phase = 'refit'
        base.check_deadline(); assert_sources(frozen)
        base.seed_everything(seed, device)
        prep = base.MitraPreprocessor().fit(x, y, seed)
        model, config = fresh_model(device)
        support = prep.transform(x)
        data = torch.as_tensor(support, device=device)
        targets = torch.as_tensor(y, device=device)
        rng, reference_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        if selected_steps:
            optimizer = base.optimizer_for(model)
            for step in range(1, selected_steps+1):
                row = episode_step(model, optimizer, data, targets, ids, rng, reference_rng, step, checkpoint)
                row['stage'] = 'refit'
                metadata['refit_history'].append(row)
                base.save_training_log(log_path, metadata)
                if step == 1 or step % 10 == 0:
                    print(f'Mitra support3072 refit step={step}/{selected_steps}', flush=True)
        optimizer = data = targets = rng = reference_rng = None
        base.release_memory()
        base.check_deadline(); assert_sources(frozen)
        if hashlib.sha256(base.environment_lock().encode()).hexdigest() != environment_sha:
            raise RuntimeError('Training environment changed during fit.')
        metadata.update(status='fit_complete', selected_steps=selected_steps,
            refit_optimizer_steps=selected_steps, fresh_refit_from_original_checkpoint=True,
            full_prediction_support=True, query_chunk=2318, feature_count=prep.n_features,
            parameters=sum(p.numel() for p in model.parameters()), seconds=time.monotonic()-started,
            resources_after_refit=probe.synchronized_snapshot(device), current_stage='fit_complete')
        predictor = base.MitraPredictor(prep, config, model.state_dict(), support, y,
                                       device, metadata, query_chunk=2318)
        base.save_training_log(log_path, metadata)
        base.check_deadline()
        return predictor
    except BaseException as error:
        metadata.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        base.save_training_log(log_path, metadata)
        if predictor is not None:
            predictor.release()
        raise
    finally:
        model = optimizer = data = targets = valid_data = prep = rng = reference_rng = None
        base.release_memory()
