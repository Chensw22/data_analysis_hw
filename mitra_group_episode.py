"""Fixed group-disjoint full-parameter adaptation; no patching of older recipes."""
import time
import json
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

import mitra_candidate as base
from mitra_full_episode_probe import optimizer_bytes

SETTINGS = dict(max_steps=50, episode_support=1024, episode_query=256,
    inner_split='first of StratifiedGroupKFold(5,shuffle=True,random_state=fitseed)', query_chunk=2318, trainable_parameters=75670026,
    selection='minimum full-context inner binary logloss; earliest exact tie including step0',
    sampler='fresh default_rng permutation; query=perm[1024:1280]; support=first1024 excluding every queryGroupId; shortage fails without resampling',
    groups='PassengerId prefix used only for inner split and episode sampling, never as a feature',
    checkpoint_sha256=base.WEIGHT_SHA, learning_rate=3e-6, weight_decay=.3,
    warmup='literal official schedule', fresh_full_refit=True)


def group_hash(groups):
    return hashlib.sha256(json.dumps(list(map(str,groups)),separators=(',', ':')).encode()).hexdigest()


def checked_groups(groups,rows):
    values=np.asarray(groups)
    if values.shape!=(rows,) or any(value is None for value in values):
        raise ValueError('One nonmissing GroupId per training row is required.')
    values=values.astype(str)
    if np.isin(values,['','nan','<NA>','None']).any():
        raise ValueError('Missing GroupId is forbidden.')
    return values


def sample_group_episode(groups,rng,support=1024,query=256):
    groups=checked_groups(groups,len(groups))
    if min(support,query)<1 or support+query>len(groups):
        raise ValueError('Nonempty support/query must fit the training pool.')
    permutation=rng.permutation(len(groups))
    query_ids=permutation[support:support+query]
    eligible=permutation[~np.isin(groups[permutation],groups[query_ids])]
    if len(eligible)<support:
        raise ValueError('Insufficient group-disjoint support; do not resample query.')
    support_ids=eligible[:support]
    if set(groups[support_ids]) & set(groups[query_ids]):
        raise RuntimeError('Episode group boundary failed.')
    return support_ids,query_ids


def inner_split(y,groups,seed):
    groups=checked_groups(groups,len(y))
    inner,valid=next(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(
        np.zeros(len(y)),y,groups))
    if (not len(inner) or not len(valid) or np.intersect1d(inner,valid).size
            or len(inner)+len(valid)!=len(y) or set(groups[inner])&set(groups[valid])):
        raise ValueError('Invalid group-disjoint inner boundary.')
    return inner,valid


def fresh_model(device):
    model, config = base.load_initial_model(base.CHECKPOINT_DIR, device)
    if (sum(p.numel() for p in model.parameters()) != SETTINGS['trainable_parameters']
            or any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
        raise RuntimeError('All 75,670,026 original FP32 parameters must be trainable.')
    return model, config


def episode_step(model, optimizer, data, targets, ids, groups, rng, step):
    base.check_deadline()
    tick = time.monotonic()
    support, query = sample_group_episode(groups, rng)
    if np.intersect1d(ids[support], ids[query]).size:
        raise ValueError('Episode query IDs leaked into support.')
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = base.literal_learning_rate(step)
    model.train()
    support_tensor = torch.as_tensor(support, device=data.device)
    query_tensor = torch.as_tensor(query, device=data.device)
    logits = base.model_forward(model, data[support_tensor], targets[support_tensor], data[query_tensor])
    if logits.shape != (len(query), 10):
        raise ValueError('Full native ten-class head required for CE.')
    loss = F.cross_entropy(logits, targets[query_tensor])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training CE.')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Missing or nonfinite training gradients.')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite updated weights.')
    state_steps = {int(state['step'].item()) for state in optimizer.state.values()}
    if state_steps != {step}:
        raise RuntimeError('Optimizer was reset or failed to advance.')
    row = dict(step=step, learning_rate=optimizer.param_groups[0]['lr'],
        training_full_head_ce=float(loss.detach().cpu()),
        support_rows=len(support), query_rows=len(query), disjoint_ids=True, group_overlap_count=0,
        support_group_ids_sha256=group_hash(groups[support]),query_group_ids_sha256=group_hash(groups[query]),
        support_ids_sha256=base.ids_hash(ids[support]), query_ids_sha256=base.ids_hash(ids[query]),
        finite_gradients=True, optimizer_state_tensor_bytes=optimizer_bytes(optimizer),
        cumulative_optimizer_steps=step, seconds=time.monotonic()-tick)
    optimizer.zero_grad(set_to_none=True)  # Before full-context validation, while moments remain resident.
    return row


def fit_group_episode(x, y, ids, groups, seed, device='mps', log_path=None, max_steps=50):
    if max_steps != 50:
        raise ValueError('This fixed recipe permits exactly a 50-step selection horizon.')
    y = base.binary_labels(y, len(x))
    ids = np.asarray(ids, dtype=np.int64)
    if ids.shape != y.shape or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique aligned training IDs required.')
    groups = checked_groups(groups,len(y))
    inner, valid = inner_split(y,groups,seed)
    if np.intersect1d(ids[inner], ids[valid]).size:
        raise ValueError('Inner validation IDs overlap training IDs.')
    started = time.monotonic()
    metadata = dict(status='incomplete', recipe=SETTINGS, seed=int(seed), training_rows=len(y),
        training_ids_sha256=base.ids_hash(ids),training_group_ids_sha256=group_hash(groups),
        inner_group_overlap_count=0,inner_split='first_stratified_group5',
        inner_train_ids_sha256=base.ids_hash(ids[inner]),
        inner_validation_ids_sha256=base.ids_hash(ids[valid]),
        inner_training_rows=len(inner), inner_validation_rows=len(valid),
        inner_history=[], refit_history=[], selected_steps=0,
        checkpoint_sha256=base.WEIGHT_SHA, validation_support_order='fixed all inner-training rows',
        precision='MPS bf16 autocast with FP32 weights' if device == 'mps' else 'CPU FP32')
    model = optimizer = data = targets = valid_data = prep = None
    try:
        base.save_training_log(log_path, metadata)
        base.seed_everything(seed, device)
        prep = base.MitraPreprocessor().fit(x.iloc[inner], y[inner], seed)
        model, config = fresh_model(device)
        data = torch.as_tensor(prep.transform(x.iloc[inner]), device=device)
        targets = torch.as_tensor(y[inner], device=device)
        valid_data = torch.as_tensor(prep.transform(x.iloc[valid]), device=device)
        optimizer = base.optimizer_for(model)
        rng = np.random.default_rng(seed)
        best_loss, selected_steps = float('inf'), 0
        for step in range(51):
            base.check_deadline()
            tick = time.monotonic()
            row = episode_step(model, optimizer, data, targets, ids[inner], groups[inner], rng, step) if step else dict(step=0)
            probabilities = base.predict_arrays(model, data, targets, valid_data, chunk=2318)
            score = float(log_loss(y[valid], probabilities, labels=[0, 1]))
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite inner binary logloss.')
            if score < best_loss:  # Strict comparison preserves earliest tie, including zero.
                best_loss, selected_steps = score, step
            row.update(validation_log_loss=score, total_seconds=time.monotonic()-tick)
            metadata['inner_history'].append(row)
            metadata.update(selected_steps=selected_steps, best_inner_log_loss=best_loss)
            base.save_training_log(log_path, metadata)
            if step % 10 == 0:
                print(f'Mitra group episode inner step={step} selected={selected_steps}', flush=True)
        metadata.update(selection_at_cap=selected_steps == 50, inner_optimizer_steps=50,
                        inner_final_resources=base.snapshot(device))
        # Transfer only the selected update count. Drop every fitted object/RNG.
        model = optimizer = data = targets = valid_data = prep = rng = None
        base.release_memory()
        base.seed_everything(seed, device)
        prep = base.MitraPreprocessor().fit(x, y, seed)
        model, config = fresh_model(device)
        support = prep.transform(x)
        data = torch.as_tensor(support, device=device)
        targets = torch.as_tensor(y, device=device)
        rng = np.random.default_rng(seed)
        if selected_steps:
            optimizer = base.optimizer_for(model)
            for step in range(1, selected_steps+1):
                row = episode_step(model, optimizer, data, targets, ids, groups, rng, step)
                metadata['refit_history'].append(row)
                base.save_training_log(log_path, metadata)
                if step == 1 or step % 10 == 0:
                    print(f'Mitra group episode refit step={step}/{selected_steps}', flush=True)
        optimizer = data = targets = None
        base.release_memory()
        metadata.update(status='fit_complete', selected_steps=selected_steps,
            refit_optimizer_steps=selected_steps, fresh_refit_from_original_checkpoint=True,
            full_prediction_support=True, query_chunk=2318, feature_count=prep.n_features,
            parameters=sum(p.numel() for p in model.parameters()),
            seconds=time.monotonic()-started, resources_after_refit=base.snapshot(device))
        predictor = base.MitraPredictor(prep, config, model.state_dict(), support, y,
                                       device, metadata, query_chunk=2318)
        base.save_training_log(log_path, metadata)
        return predictor
    except BaseException as error:
        metadata.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        base.save_training_log(log_path, metadata)
        raise
    finally:
        model = optimizer = data = targets = valid_data = prep = None
        base.release_memory()
