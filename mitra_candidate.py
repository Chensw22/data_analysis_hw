"""Mitra-v2 resource prototype. Deliberately refuses screen until resource review.

The five vendored upstream files are unmodified. This adapter adds train-only
schema handling, full support, bounded query batches and portable CPU state.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
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
SETTINGS = dict(seed=42, query_chunk=128, weights_and_input_dtype='float32',
                mps_autocast_dtype='bfloat16', cpu_autocast=False,
                support_policy='all_rows_no_sampling_no_cap', n_estimators=1,
                random_mirror_x=True, mps_memory_fraction=.65,
                sdpa_batch_chunk=1, sdpa_long_sequence_threshold=512,
                sdpa_checkpoint_each_chunk=True)


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
            known = sorted(set(x[col].dropna().astype(str)))
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

    def encode(self, x):
        result = np.empty((len(x), len(self.columns)), dtype=np.float32)
        for index, col in enumerate(self.columns):
            if col in self.categories:
                values = x[col]
                encoded = values.astype(str).map(self.vocab[col]).fillna(0).to_numpy(dtype=np.float32)
                encoded[values.isna().to_numpy()] = 1.
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
    def __init__(self, prep, config, state, support, labels, device='cpu', metadata=None):
        self.preprocessor = prep
        self.config = dict(config)
        self.state = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.support = np.array(support, dtype=np.float32, copy=True)
        self.labels = binary_labels(labels, len(self.support))
        self.device = device
        self.query_chunk = SETTINGS['query_chunk']
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
        full_context=True, device=args.device,
        attention_backend=dict(batch_chunk=SETTINGS['sdpa_batch_chunk'],
            long_sequence_threshold=SETTINGS['sdpa_long_sequence_threshold'],
            checkpoint_each_training_chunk=True, full_key_sequence=True))
    start = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Mitra resource probe deadline exceeded.')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
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
        predictor = MitraPredictor(prep, config, state, prep.transform(raw), labels, args.device)
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
                      full_fold_query_chunks=int(np.ceil(2318/SETTINGS['query_chunk'])),
                      estimated_full_fold_seconds=timings[-1]*int(np.ceil(2318/args.query_rows)))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--context-rows', type=int, default=4636)
    parser.add_argument('--query-rows', type=int, default=128)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--skip-roundtrip', action='store_true')
    parser.add_argument('--budget-seconds', type=int, default=1200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage != 'probe':
        raise RuntimeError('Resource prototype only: screen/confirm/export not authorized until probe review.')
    if args.context_rows < 2 or not 0 < args.query_rows < args.context_rows or args.budget_seconds < 1:
        parser.error('Require context>query>0 and positive budget.')
    resource_probe(args)


if __name__ == '__main__':
    sys.modules['mitra_candidate'] = sys.modules[__name__]
    MitraPreprocessor.__module__ = 'mitra_candidate'
    MitraPredictor.__module__ = 'mitra_candidate'
    main()
