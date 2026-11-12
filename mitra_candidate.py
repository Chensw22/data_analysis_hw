"""Mitra-v2 resource probes and predeclared development-only training pipelines.

The five vendored upstream files are unmodified. This adapter adds train-only
schema handling, full support, bounded query batches and portable CPU state.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from third_party.mitra._internal.config.enums import Task
from third_party.mitra._internal.data.preprocessor import Preprocessor
from third_party.mitra._internal.models.tab2d import Tab2D, MultiheadAttention as UpstreamAttention

ROOT = Path(__file__).resolve().parent
NAMES = ('v71_Mitra2_zeroshot', 'v72_Mitra2_adapt50')
COMPACT = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
           'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNumber', 'Side']
CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
NUMERIC = [c for c in COMPACT if c not in CATEGORICAL]
CHECKPOINT_DIR = ROOT / 'artifacts/pretrained/mitra-v2'
WEIGHT_SHA = '5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09'
WEIGHT_BYTES = 302717904
MODEL_CONFIG = dict(dim=512, dim_output=10, n_layers=12, n_heads=4, task='CLASSIFICATION')
SETTINGS = dict(seed_policy='formal_fit_uses_passed_cv_seed_plus_fold; synthetic_resource_probe_uses_42',
                query_chunk=128, weights_and_input_dtype='float32',
                mps_autocast_dtype='bfloat16', cpu_autocast=False,
                support_policy='all_rows_no_sampling_no_cap', n_estimators=1,
                random_mirror_x=True, mps_memory_fraction=.65,
                sdpa_batch_chunk=1, sdpa_long_sequence_threshold=512,
                sdpa_checkpoint_each_chunk=True)
ADAPTATION = dict(max_steps=50, learning_rate=3e-6, weight_decay=.3,
                 warmup_steps=10, max_episode_support=16384, max_episode_query=1024,
                 inner_validation_fraction=.2)
RUN_OUT = None
DEADLINE = None
RUNTIME = None
FRAMEWORK = None
BASE_FIT = None
BASE_EVALUATE = None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(directory=CHECKPOINT_DIR):
    directory = Path(directory)
    weights = directory / 'model.safetensors'
    if not weights.is_file() or weights.stat().st_size != WEIGHT_BYTES or sha256(weights) != WEIGHT_SHA:
        raise ValueError('Pinned Mitra-v2 checkpoint missing or hash/size mismatch; no automatic download.')
    config = json.loads((directory / 'config.json').read_text())
    if config != MODEL_CONFIG:
        raise ValueError('Unexpected pinned model config.')
    return config


def binary_labels(y, expected_rows):
    values = np.asarray(y)
    if values.shape != (expected_rows,) or not np.isin(values, [0, 1]).all():
        raise ValueError('Expected one binary label per support row.')
    return values.astype(np.int64, copy=True)


class MitraPreprocessor:
    """Deterministic raw schema, followed by the official Preprocessor.

    All-missing numerical columns are filled with zero before upstream fitting
    and therefore removed by its constant filter. Unknown/missing categories
    reserve distinct codes. No held-out values contribute to vocabulary/means.
    """
    def fit(self, x, y, seed=42):
        y = binary_labels(y, len(x))
        self.columns = list(COMPACT)
        self.categories = list(CATEGORICAL)
        self.vocab = {}
        for col in self.categories:
            missing = self.category_missing(x[col])
            known = sorted(set(x.loc[~missing, col].astype(str)))
            self.vocab[col] = {value: i + 2 for i, value in enumerate(known)}
        array = self.encode(x)
        self.all_missing = np.isnan(array).all(axis=0)
        array[:, self.all_missing] = 0.
        self.upstream = Preprocessor(dim_embedding=None, n_classes=2, dim_output=10,
            use_quantile_transformer=False, use_feature_count_scaling=False,
            use_random_transforms=False, shuffle_classes=False, shuffle_features=False,
            random_mirror_regression=False, random_mirror_x=SETTINGS['random_mirror_x'],
            task=Task.CLASSIFICATION)
        previous = np.random.get_state()
        try:
            np.random.seed(seed)
            self.upstream.fit(array.copy(), y.copy())
        finally:
            np.random.set_state(previous)
        self.n_features = self.transform(x).shape[1]
        if self.n_features < 1:
            raise ValueError('At least one nonconstant feature is required.')
        return self

    @staticmethod
    def category_missing(values):
        # iterate already encodes missing categoricals using this sentinel.
        return values.isna() | values.astype('string').eq('__MISSING__').fillna(False)

    def encode(self, x):
        result = np.empty((len(x), len(self.columns)), dtype=np.float32)
        for index, col in enumerate(self.columns):
            if col in self.categories:
                values = x[col]
                encoded = values.astype(str).map(self.vocab[col]).fillna(0).to_numpy(dtype=np.float32)
                encoded[self.category_missing(values).to_numpy(dtype=bool)] = 1.
                result[:, index] = encoded
            else:
                result[:, index] = pd.to_numeric(x[col], errors='raise').to_numpy(dtype=np.float32)
        if np.isinf(result).any():
            raise ValueError('Infinite input is not supported.')
        return result

    def transform(self, x):
        array = self.encode(x)
        array[:, self.all_missing] = 0.
        result = self.upstream.transform_X(array.copy())
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite transformed input.')
        return np.ascontiguousarray(result, dtype=np.float32)


class BatchChunkedAttention(UpstreamAttention):
    """Exact independent-batch decomposition of upstream non-Flash SDPA.

    Q/K sequence axes and the full softmax denominator never change. Long row
    attention is split across independent feature batches. Short feature
    attention retains the original vectorized implementation. Each SDPA block
    is checkpointed during autograd, so saved attention matrices do not add up
    across blocks. The upstream layer checkpoint remains enabled as well.
    """
    def forward(self, query, key, value, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=None, max_seqlen_k=None):
        if ((self.use_flash_attn and cu_seqlens_q is not None)
                or max(query.shape[1], key.shape[1]) <= self.sequence_threshold
                or query.shape[0] <= self.batch_chunk):
            return super().forward(query, key, value, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k)
        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise ValueError('Independent batch chunking requires equal Q/K/V batch dimensions.')
        q, k, v = self.q(query), self.k(key), self.v(value)
        # Same axis order as upstream einops: b t (h d) -> b h t d.
        def heads(tensor):
            return tensor.reshape(tensor.shape[0], tensor.shape[1], self.n_heads,
                                  self.dim // self.n_heads).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        pieces = []
        for start in range(0, q.shape[0], self.batch_chunk):
            chunk = slice(start, start+self.batch_chunk)
            inputs = (q[chunk], k[chunk], v[chunk])
            if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
                output = checkpoint(F.scaled_dot_product_attention, *inputs, use_reentrant=False)
            else:
                output = F.scaled_dot_product_attention(*inputs)
            pieces.append(output)
        output = torch.cat(pieces, dim=0).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.dim)
        return self.o(output)


def install_attention_backend(model, batch_chunk=None, sequence_threshold=None):
    batch_chunk = SETTINGS['sdpa_batch_chunk'] if batch_chunk is None else batch_chunk
    sequence_threshold = SETTINGS['sdpa_long_sequence_threshold'] if sequence_threshold is None else sequence_threshold
    if batch_chunk < 1 or sequence_threshold < 0:
        raise ValueError('Invalid independent-batch backend settings.')
    for module in model.modules():
        if isinstance(module, UpstreamAttention):
            # No new parameters/buffers or state_dict key changes; preserve the
            # exact loaded projection modules. Engines are rebuilt, not pickled.
            module.__class__ = BatchChunkedAttention
            module.batch_chunk = batch_chunk
            module.sequence_threshold = sequence_threshold
    return model


def make_model(config, state=None, device='cpu', memory_backend=True):
    # Meta construction avoids a second randomly initialized 303MB model.
    if state is None:
        model = Tab2D(**config, use_pretrained_weights=False, path_to_weights='', device=device).to(device)
    else:
        with torch.device('meta'):
            model = Tab2D(**config, use_pretrained_weights=False, path_to_weights='', device=device)
        model.to_empty(device=device)
        model.load_state_dict(state, strict=True)
    return install_attention_backend(model) if memory_backend else model


def model_forward(model, support, labels, query):
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError('Support/query feature schema differs.')
    if len(support) < 1 or labels.shape != (len(support),):
        raise ValueError('Empty/misaligned support.')
    device = support.device
    # Upstream quantile embedding writes its input; never mutate saved context.
    # Match official MPS trainer mixed precision. Input quantiles and persisted
    # weights remain FP32; unsupported bf16 is a reported failure, not a fallback.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'mps'):
        logits = model(support.clone().unsqueeze(0), labels.clone().unsqueeze(0), query.clone().unsqueeze(0),
            torch.zeros((1, support.shape[1]), dtype=torch.bool, device=device),
            torch.zeros((1, len(support)), dtype=torch.bool, device=device),
            torch.zeros((1, len(query)), dtype=torch.bool, device=device))
    return logits[0].float()


@torch.no_grad()
def predict_arrays(model, support, labels, query, chunk=128):
    if chunk < 1:
        raise ValueError('Positive query chunk required.')
    model.eval()
    result = []
    for start in range(0, len(query), chunk):
        check_deadline()
        logits = model_forward(model, support, labels, query[start:start + chunk])[:, :2]
        result.append(logits.softmax(dim=-1).float().cpu().numpy())
    values = np.concatenate(result) if result else np.empty((0, 2), dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite predictions.')
    return values


def episode_indices(n_rows, query_rows, generator):
    if not 0 < query_rows < n_rows:
        raise ValueError('Episode needs nonempty disjoint support and query.')
    order = torch.randperm(n_rows, generator=generator)
    return order[query_rows:], order[:query_rows]


def backward_probe(model, x, y, query_rows=128, seed=42, support_rows=None):
    """One disposable synthetic AdamW step, including optimizer-state memory."""
    support_ids, query_ids = episode_indices(len(y), query_rows, torch.Generator().manual_seed(seed))
    if support_rows is not None:
        if not 0 < support_rows <= len(support_ids):
            raise ValueError('Invalid episode support row count.')
        support_ids = support_ids[:support_rows]
    support_ids, query_ids = support_ids.to(x.device), query_ids.to(x.device)
    model.train()
    model.zero_grad(set_to_none=True)
    # Literal official first warmup update: .1 * 3e-6. This probe does not
    # freeze the full adaptation loop, only its optimizer-state resource cost.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-7, weight_decay=.3,
                                 betas=(.9, .999), eps=1e-8)
    logits = model_forward(model, x[support_ids], y[support_ids], x[query_ids])
    # Upstream trains against all ten pretrained head outputs; only inference
    # discards unused classes. Binary CE here would change the adaptation loss.
    loss = F.cross_entropy(logits, y[query_ids])
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    if not finite:
        raise FloatingPointError('Nonfinite/missing synthetic episode gradients.')
    tracked = next(p for p in model.parameters()
                   if p.grad is not None and bool(torch.count_nonzero(p.grad)))
    original = tracked.detach().clone()
    optimizer.step()
    changed = not torch.equal(original, tracked)
    if not changed or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Synthetic AdamW step did not produce finite updated weights.')
    optimizer_bytes = sum(value.numel()*value.element_size() for state in optimizer.state.values()
                          for value in state.values() if torch.is_tensor(value))
    summary = dict(support_rows=len(support_ids), query_rows=len(query_ids),
                   disjoint=True, parameters_with_grad=len(gradients), finite_gradients=True,
                   optimizer_steps=1, weight_update_confirmed=changed,
                   optimizer_state_tensor_bytes=optimizer_bytes,
                   resources_with_optimizer=snapshot(x.device.type), loss_value_reported=False)
    model.zero_grad(set_to_none=True)
    return summary


class MitraPredictor:
    def __init__(self, prep, config, state, support, labels, device='cpu', metadata=None,
                 query_chunk=None):
        self.preprocessor = prep
        self.config = dict(config)
        self.state = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.support = np.array(support, dtype=np.float32, copy=True)
        self.labels = binary_labels(labels, len(self.support))
        self.device = device
        self.query_chunk = SETTINGS['query_chunk'] if query_chunk is None else int(query_chunk)
        if self.query_chunk < 1:
            raise ValueError('Positive query chunk required.')
        self.metadata = dict(metadata or {})
        self._engine = None

    def engine(self):
        if self._engine is None:
            self._engine = make_model(self.config, self.state, self.device)
        return self._engine

    def predict_proba(self, x):
        query = self.preprocessor.transform(x)
        return predict_arrays(self.engine(), torch.as_tensor(self.support, device=self.device),
            torch.as_tensor(self.labels, device=self.device),
            torch.as_tensor(query, device=self.device), self.query_chunk)

    def release(self):
        self._engine = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_engine'] = None
        # CPU is the portable reload default; caller explicitly opts into MPS.
        state['device'] = 'cpu'
        return state


def snapshot(device):
    if device == 'mps':
        torch.mps.synchronize()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(max_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024))
    if device == 'mps':
        result.update(mps_current_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory(),
                      mps_measurement='snapshots_not_peak')
    return result


def check_deadline():
    if DEADLINE is not None and time.monotonic() > DEADLINE:
        raise TimeoutError('Mitra stage deadline exceeded; incomplete result is not a candidate score.')


def seed_everything(seed, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'mps':
        torch.mps.manual_seed(seed)


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def load_initial_model(directory, device):
    check_deadline()
    config = verify_checkpoint(directory)  # Fresh integrity check for every fit.
    state = load_file(str(Path(directory)/'model.safetensors'), device='cpu')
    model = make_model(config, state, device)
    del state
    return model, config


def official_episode_indices(y, ids, rng):
    """Pinned sampler, independent from deterministic validation inference RNG."""
    y, ids = np.asarray(y), np.asarray(ids)
    order = rng.permutation(len(y))
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if counts.min() >= 5:
        a, b = next(StratifiedKFold(5, shuffle=True, random_state=rng).split(order, y[order]))
    else:
        a, b = train_test_split(np.arange(len(y)), test_size=.2, random_state=rng)
    support = order[a][:ADAPTATION['max_episode_support']]
    query = order[b][:ADAPTATION['max_episode_query']]
    support = support[rng.choice(len(support), size=len(support), replace=False)]
    if not len(support) or not len(query) or np.intersect1d(ids[support], ids[query]).size:
        raise ValueError('Adaptation episode requires nonempty disjoint row IDs.')
    return support, query


def literal_learning_rate(step):
    if step < 1:
        raise ValueError('Optimizer updates start at step1.')
    # Official scheduler makes9 increments and remains at .91*base from step10.
    return ADAPTATION['learning_rate'] * (.1 + .09 * min(step-1, 9))


def optimizer_for(model):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=ADAPTATION['learning_rate'], weight_decay=ADAPTATION['weight_decay'],
        betas=(.9, .999), eps=1e-8)


def adaptation_step(model, optimizer, data, targets, y, ids, rng, step):
    check_deadline()
    started = time.monotonic()
    support, query = official_episode_indices(y, ids, rng)
    support_tensor = torch.as_tensor(support, device=data.device)
    query_tensor = torch.as_tensor(query, device=data.device)
    lr = literal_learning_rate(step)
    for group in optimizer.param_groups:
        group['lr'] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model_forward(model, data[support_tensor], targets[support_tensor], data[query_tensor])
    loss = F.cross_entropy(logits, targets[query_tensor])  # Full native10-class head.
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite adaptation CE.')
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite adaptation gradient.')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite adapted parameters.')
    return dict(step=step, learning_rate=lr, training_full_head_ce=float(loss.detach().cpu()),
        support_rows=len(support), query_rows=len(query),
        support_ids_sha256=ids_hash(ids[support]), query_ids_sha256=ids_hash(ids[query]),
        disjoint_ids=True, optimizer_steps=1, seconds=time.monotonic()-started)


def save_training_log(path, metadata):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(metadata, indent=2)+'\n')
        temporary.replace(path)


def fit_mitra(x, y, ids, seed, device, recipe, checkpoint_dir=CHECKPOINT_DIR,
              query_chunk=128, log_path=None):
    if recipe not in ['zero-shot', 'adapt50']:
        raise ValueError('Unknown Mitra recipe.')
    if recipe == 'adapt50':
        raise RuntimeError('adapt50 disabled: representative AdamW resource probe OOM; draft loop unvalidated.')
    y = binary_labels(y, len(x))
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) != len(y) or len(np.unique(ids)) != len(ids):
        raise ValueError('Training row IDs must be unique and aligned.')
    started = time.monotonic()
    metadata = dict(status='incomplete', recipe=recipe, seed=int(seed),
        checkpoint_sha256=WEIGHT_SHA, training_rows=len(y), training_ids_sha256=ids_hash(ids),
        full_prediction_support=True, query_chunk=query_chunk,
        precision='mps_bfloat16_autocast' if device == 'mps' else 'cpu_float32',
        inner_history=[], refit_history=[], selected_steps=0,
        validation_support_order='fixed_training_order_no_episode_rng_consumption')
    save_training_log(log_path, metadata)
    selected_steps = 0
    model = optimizer = data = targets = valid_data = None
    try:
        if recipe == 'adapt50':
            inner, valid = train_test_split(np.arange(len(y)),
                test_size=ADAPTATION['inner_validation_fraction'], stratify=y, random_state=seed)
            if np.intersect1d(ids[inner], ids[valid]).size:
                raise ValueError('Inner validation IDs leaked into training context.')
            seed_everything(seed, device)
            prep = MitraPreprocessor().fit(x.iloc[inner], y[inner], seed)
            model, config = load_initial_model(checkpoint_dir, device)
            data = torch.as_tensor(prep.transform(x.iloc[inner]), device=device)
            targets = torch.as_tensor(y[inner], device=device)
            valid_data = torch.as_tensor(prep.transform(x.iloc[valid]), device=device)
            optimizer = optimizer_for(model)
            rng = np.random.RandomState(seed)
            best_loss = float('inf')
            metadata.update(inner_training_rows=len(inner), inner_validation_rows=len(valid),
                inner_train_ids_sha256=ids_hash(ids[inner]), inner_validation_ids_sha256=ids_hash(ids[valid]))
            for step in range(ADAPTATION['max_steps']+1):
                check_deadline()
                tick = time.monotonic()
                row = (adaptation_step(model, optimizer, data, targets, y[inner], ids[inner], rng, step)
                       if step else dict(step=0, optimizer_steps=0, learning_rate=None))
                probabilities = predict_arrays(model, data, targets, valid_data, query_chunk)
                score = float(log_loss(y[valid], probabilities, labels=[0, 1]))
                if not np.isfinite(score):
                    raise FloatingPointError('Nonfinite inner validation logloss.')
                row.update(validation_log_loss=score, total_seconds=time.monotonic()-tick)
                metadata['inner_history'].append(row)
                if score < best_loss:
                    best_loss, selected_steps = score, step
                metadata.update(selected_steps=selected_steps, best_inner_log_loss=best_loss)
                save_training_log(log_path, metadata)
                if step % 5 == 0:
                    print(f'Mitra inner step={step} logloss={score:.6f} selected={selected_steps}', flush=True)
            metadata.update(inner_optimizer_steps=ADAPTATION['max_steps'],
                selection_at_cap=selected_steps == ADAPTATION['max_steps'],
                inner_final_resources=snapshot(device))
            # Selected weights are deliberately discarded; only the step count transfers.
            model = optimizer = data = targets = valid_data = None
            release_memory()
        seed_everything(seed, device)
        prep = MitraPreprocessor().fit(x, y, seed)
        model, config = load_initial_model(checkpoint_dir, device)
        support = prep.transform(x)
        data = torch.as_tensor(support, device=device)
        targets = torch.as_tensor(y, device=device)
        refit_started = time.monotonic()
        if selected_steps:
            optimizer = optimizer_for(model)
            rng = np.random.RandomState(seed)
            for step in range(1, selected_steps+1):
                metadata['refit_history'].append(adaptation_step(model, optimizer, data, targets, y, ids, rng, step))
                save_training_log(log_path, metadata)
                if step == 1 or step % 5 == 0:
                    print(f'Mitra refit step={step}/{selected_steps}', flush=True)
        metadata.update(status='fit_complete', selected_steps=selected_steps,
            refit_optimizer_steps=selected_steps, refit_seconds=time.monotonic()-refit_started,
            parameters=sum(p.numel() for p in model.parameters()),
            feature_count=prep.n_features, resources_after_refit=snapshot(device),
            seconds=time.monotonic()-started, fresh_refit_from_original_checkpoint=True)
        predictor = MitraPredictor(prep, config, model.state_dict(), support, y, device,
                                   metadata, query_chunk=query_chunk)
        save_training_log(log_path, metadata)
        return predictor
    except Exception as error:
        metadata.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        save_training_log(log_path, metadata)
        raise
    finally:
        model = optimizer = data = targets = valid_data = None
        release_memory()


def fit(name, matrices, train_idx, y, predict_idx, seed):
    check_deadline()
    if name == FRAMEWORK.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in NAMES or name not in FRAMEWORK.CONFIGS:
        raise ValueError('Unapproved Mitra candidate.')
    train_idx, predict_idx = np.asarray(train_idx), np.asarray(predict_idx)
    if np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Training/prediction IDs must be disjoint.')
    key = ids_hash(train_idx)[:12]
    path = RUN_OUT/f'{name}_seed{seed}_{key}_training.json' if RUN_OUT else None
    predictor = fit_mitra(matrices[False][0].iloc[train_idx], y, train_idx, seed,
        RUNTIME['device'], 'zero-shot' if name == NAMES[0] else 'adapt50',
        RUNTIME['checkpoint_dir'], RUNTIME['query_chunk'], path)
    try:
        started = time.monotonic()
        probability = predictor.predict_proba(matrices[False][0].iloc[predict_idx])[:, 1]
        predictor.metadata.update(prediction_seconds=time.monotonic()-started,
            prediction_rows=len(predict_idx), status='complete',
            resources_after_prediction=snapshot(RUNTIME['device']))
        save_training_log(path, predictor.metadata)
        return probability, predictor
    finally:
        predictor.release()


def environment_lock():
    return ''.join(f'{name}=={version}\n' for name, version in sorted(
        (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()))


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'mitra_policy' in protocol:
        return
    declared = json.loads((ROOT/'reports/mitra_protocol.json').read_text())
    if declared['runtime_decision'] != RUNTIME['decision']:
        raise RuntimeError('Mitra runtime decision changed before first baseline.')
    verify_checkpoint(RUNTIME['checkpoint_dir'])
    freeze_sources(out)
    source_manifest = json.loads((out/'freeze.json').read_text())['source_sha256']
    protocol['sources'].update(source_manifest)
    # Include reviewed sampler protocol in the formal source contract.
    review = ROOT/'reports/mitra_adaptation_review.json'
    protocol['sources']['reports/mitra_adaptation_review.json'] = sha256(review)
    shutil.copy2(review, out/'mitra_adaptation_review.json')
    lock = environment_lock()
    (out/'requirements-mitra.lock.txt').write_text(lock)
    protocol.update(mitra_policy=declared, mitra_environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        mitra_runtime={k: str(v) if isinstance(v, Path) else v for k, v in RUNTIME.items()},
        mitra_checkpoint_sha256=WEIGHT_SHA)
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # Before baseline or any candidate training.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def configure_runtime(args):
    global FRAMEWORK, BASE_FIT, BASE_EVALUATE, RUNTIME, RUN_OUT
    decision = json.loads((ROOT/'reports/mitra_protocol.json').read_text())['runtime_decision']
    if decision.get('status') != 'approved_before_cv':
        raise RuntimeError('Mitra recipes/query chunk/budgets await resource approval before any CV.')
    recipes = decision.get('enabled_recipes')
    if recipes != ['zero-shot']:
        raise ValueError('Only zero-shot is enabled; adapt50 failed the prescore resource gate.')
    chunk = decision.get('query_chunk')
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError('Freeze a positive query chunk before CV.')
    if args.query_chunk is not None and args.query_chunk != chunk:
        raise ValueError('CLI query chunk differs from approved protocol.')
    if decision.get('device') != args.device:
        raise ValueError('CLI device differs from approved protocol.')
    budgets = decision.get('stage_budget_seconds', {})
    if not all(isinstance(budgets.get(stage), (int, float)) and budgets[stage] > 0
               for stage in ['screen', 'confirm', 'export']):
        raise ValueError('Stage-specific resource budgets must be frozen before CV.')
    import iterate
    FRAMEWORK = iterate
    BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
    RUNTIME = dict(decision=decision, device=args.device, query_chunk=chunk,
                   checkpoint_dir=Path(args.checkpoint_dir))
    RUN_OUT = ROOT/'artifacts'/args.run
    iterate.CONFIGS = {iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT)}
    for recipe in recipes:
        name = NAMES[0] if recipe == 'zero-shot' else NAMES[1]
        iterate.CONFIGS[name] = dict(context=False, model='mitra', recipe=recipe,
            query_chunk=chunk, checkpoint_sha256=WEIGHT_SHA, settings={**SETTINGS, 'query_chunk': chunk},
            adaptation=ADAPTATION if recipe == 'adapt50' else None)
    iterate.BLEND = []
    iterate.fit, iterate.evaluate = fit, evaluate
    if args.stage in ['confirm', 'export']:
        iterate.assert_frozen(RUN_OUT)
        saved = json.loads((RUN_OUT/'protocol.json').read_text())
        expected = saved['mitra_environment_sha256']
        if hashlib.sha256(environment_lock().encode()).hexdigest() != expected or sha256(RUN_OUT/'requirements-mitra.lock.txt') != expected:
            raise RuntimeError('Mitra dependency environment changed.')
        if saved['mitra_runtime']['decision'] != decision:
            raise RuntimeError('Frozen runtime decision changed.')
    return budgets[args.stage]


def synthetic_frame(n, seed=42):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({c: rng.lognormal(3., 2., n) for c in NUMERIC})
    for col, count in zip(CATEGORICAL, [3, 2, 3, 2, 7, 2]):
        x[col] = np.asarray([f'level_{i}' for i in range(count)])[rng.integers(count, size=n)]
    # Cover missing values without introducing a train/holdout data dependency.
    x.loc[np.arange(0, n, 19), 'Age'] = np.nan
    x.loc[np.arange(0, n, 23), 'HomePlanet'] = None
    return x[COMPACT]


def freeze_sources(output):
    output = Path(output)
    source_dir = output / 'source'
    source_dir.mkdir(parents=True, exist_ok=False)
    files = [ROOT/'mitra_candidate.py', ROOT/'test_mitra_candidate.py', ROOT/'reports/mitra_protocol.json',
             *sorted((ROOT/'third_party/mitra').rglob('*.py')),
             ROOT/'third_party/mitra/LICENSE', ROOT/'third_party/mitra/NOTICE', ROOT/'third_party/mitra/MODEL_CARD.md']
    manifest = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[str(relative)] = sha256(path)
    packages = {}
    for name in ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'einops', 'einx', 'huggingface_hub', 'safetensors', 'loguru']:
        packages[name] = importlib.metadata.version(name)
    (output/'freeze.json').write_text(json.dumps(dict(source_sha256=manifest,
        packages=packages, python=sys.version, platform=platform.platform()), indent=2)+'\n')


def resource_probe(args):
    output = Path(args.output)
    freeze_sources(output)  # Must precede the first model/weight load.
    result = dict(status='incomplete', kind='synthetic_resource_only_no_accuracy',
        context_rows=args.context_rows, query_rows=args.query_rows,
        weights_and_input_dtype='float32', autocast_dtype='bfloat16' if args.device == 'mps' else None,
        full_context=True, device=args.device, query_chunk=args.query_chunk or SETTINGS['query_chunk'],
        attention_backend=dict(batch_chunk=SETTINGS['sdpa_batch_chunk'],
            long_sequence_threshold=SETTINGS['sdpa_long_sequence_threshold'],
            checkpoint_each_training_chunk=True, full_key_sequence=True))
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Mitra resource probe deadline exceeded.')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds or 1200)
    predictor = None
    try:
        if args.device == 'mps':
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable; do not silently substitute CPU.')
            torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
            result['mps_memory_fraction'] = SETTINGS['mps_memory_fraction']
            result['mps_allocator_limit_bytes'] = int(torch.mps.recommended_max_memory()*SETTINGS['mps_memory_fraction'])
        if sys.platform == 'darwin':
            result['machine_memory_pressure_before'] = subprocess.run(
                ['/usr/bin/memory_pressure', '-Q'], capture_output=True, text=True,
                timeout=5, check=True).stdout.strip()
        config = verify_checkpoint(args.checkpoint_dir)
        raw = synthetic_frame(args.context_rows, 42)
        result['support_source'] = 'synthetic'
        result['support_labels'] = 'synthetic_alternating_binary_no_real_target_read'
        query = synthetic_frame(args.query_rows, 43)
        labels = np.arange(args.context_rows, dtype=np.int64) % 2
        prep = MitraPreprocessor().fit(raw, labels)
        state = load_file(str(Path(args.checkpoint_dir)/'model.safetensors'), device='cpu')
        predictor = MitraPredictor(prep, config, state, prep.transform(raw), labels, args.device,
                                   query_chunk=args.query_chunk)
        del state
        result['before'] = snapshot(args.device)
        timings = []
        expected = None
        for repeat in range(2):
            tick = time.monotonic()
            probabilities = predictor.predict_proba(query)
            timing = time.monotonic() - tick
            timings.append(timing)
            if expected is not None:
                np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-6)
            expected = probabilities.copy()
            print(json.dumps(dict(repeat=repeat, seconds=timing, resources=snapshot(args.device))), flush=True)
        result.update(inference_seconds=timings, after_inference=snapshot(args.device),
                      resolved_support_rows=len(predictor.support), features=prep.n_features,
                      parameters=sum(p.numel() for p in predictor.engine().parameters()),
                      full_fold_query_chunks=int(np.ceil(2318/predictor.query_chunk)),
                      estimated_full_fold_seconds=timings[-1]*2318/args.query_rows,
                      extrapolation_note='Rough row-ratio extrapolation only; fixed support recomputation makes timing nonlinear.')
        if args.compare_query_chunk is not None:
            predictor.query_chunk = args.compare_query_chunk
            other = predictor.predict_proba(query)
            result['query_chunk_comparison'] = dict(other_chunk=args.compare_query_chunk,
                max_absolute_difference=float(np.max(np.abs(other-expected))))
            np.testing.assert_allclose(other, expected, rtol=1e-5, atol=1e-6)
            predictor.query_chunk = args.query_chunk or SETTINGS['query_chunk']
        if args.backward:
            tick = time.monotonic()
            # Resource dimensions match the official five-way first-fold
            # episode; the synthetic random partition is not a training sampler.
            episode_query_total = int(np.ceil(args.context_rows/5))
            result['backward'] = backward_probe(predictor.engine(),
                torch.as_tensor(predictor.support, device=args.device),
                torch.as_tensor(predictor.labels, device=args.device),
                query_rows=min(1024, episode_query_total),
                support_rows=args.context_rows-episode_query_total)
            result['backward_seconds'] = time.monotonic()-tick
            result['after_backward'] = snapshot(args.device)
            # The resource-only update must not contaminate zero-shot reload
            # comparison. CPU state is immutable; discard the modified engine.
            predictor.release()
        if not args.skip_roundtrip:
            predictor.release()
            with tempfile.TemporaryDirectory(dir=output) as directory:
                path = Path(directory)/'probe.joblib'
                joblib.dump(predictor, path)
                del predictor
                predictor = joblib.load(path)
                assert predictor.device == 'cpu'
                predictor.device = args.device
                restored = predictor.predict_proba(query)
                np.testing.assert_allclose(restored, expected, rtol=1e-5, atol=1e-6)
                result['roundtrip_max_abs_difference'] = float(np.max(np.abs(restored-expected)))
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        result['total_seconds'] = time.monotonic()-start
        if predictor is not None:
            predictor.release()
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


def main():
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--context-rows', type=int, default=4636)
    parser.add_argument('--query-rows', type=int, default=128)
    parser.add_argument('--query-chunk', type=int,
                        help='Probe: fixed explicit chunk. CV: must match approved runtime protocol.')
    parser.add_argument('--compare-query-chunk', type=int, help='Resource-only invariance check on identical queries.')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--skip-roundtrip', action='store_true')
    parser.add_argument('--budget-seconds', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', default='run_038')
    args = parser.parse_args()
    if args.context_rows < 2 or not 0 < args.query_rows < args.context_rows or (args.budget_seconds is not None and args.budget_seconds < 1):
        parser.error('Require context>query>0 and positive budget.')
    if any(value is not None and value < 1 for value in [args.query_chunk, args.compare_query_chunk]):
        parser.error('Query chunks must be positive.')
    torch.set_num_threads(4)
    if args.stage == 'probe':
        if args.output is None:
            parser.error('Resource probe needs a fresh --output directory.')
        resource_probe(args)
        return
    if args.budget_seconds is not None or args.output is not None or args.compare_query_chunk is not None:
        parser.error('Formal stages use frozen protocol budgets/run path; probe-only overrides are invalid.')
    budget = configure_runtime(args)
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('Approved MPS device unavailable; no fallback.')
        torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
    DEADLINE = time.monotonic()+budget
    def timeout(*_):
        raise TimeoutError('Mitra formal stage deadline exceeded; incomplete stage.')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        {'screen': FRAMEWORK.screen, 'confirm': FRAMEWORK.confirm, 'export': FRAMEWORK.export}[args.stage](RUN_OUT)
    except Exception as error:
        if RUN_OUT.exists():
            (RUN_OUT/f'{args.stage}_failure.json').write_text(json.dumps(dict(
                status='failed_inconclusive', error_type=type(error).__name__, error=str(error)), indent=2)+'\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        DEADLINE = None


if __name__ == '__main__':
    sys.modules['mitra_candidate'] = sys.modules[__name__]
    MitraPreprocessor.__module__ = 'mitra_candidate'
    MitraPredictor.__module__ = 'mitra_candidate'
    main()
