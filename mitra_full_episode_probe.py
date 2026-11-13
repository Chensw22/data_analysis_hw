"""Synthetic resource gate only; does not authorize a training recipe or score."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import mitra_candidate as mitra

ROOT = Path(__file__).resolve().parent
RECIPE = dict(pool_rows=8693, support_rows=1024, query_rows=256, steps=2,
              validation_support=3708, validation_query=928, final_support=8693,
              final_query=2318, query_chunk=2318, seed=42, budget_seconds=1200,
              trainable_parameters=75670026, mps_memory_fraction=.65,
              weights_dtype='float32', mps_autocast='bfloat16', native_head=10,
              learning_rates=[3e-7, 5.7e-7], weight_decay=.3,
              sampler='fresh uniform permutation of synthetic training pool; disjoint within episode',
              limitation='resource compatibility only; not official full-size episodic sampler or candidate authorization')


def sample_episode(n, rng, support=1024, query=256):
    if min(support, query) < 1 or support + query > n:
        raise ValueError('Episode must have nonempty disjoint support/query.')
    indices = rng.permutation(n)[:support + query]
    return indices[:support], indices[support:]


def block_hashes(model):
    result = {}
    for name, parameter in model.named_parameters():
        key = '.'.join(name.split('.')[:2]) if name.startswith('layers.') else name.split('.')[0]
        digest = result.setdefault(key, hashlib.sha256())
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {key: value.hexdigest() for key, value in result.items()}


def optimizer_bytes(optimizer):
    return sum(v.numel() * v.element_size() for state in optimizer.state.values()
               for v in state.values() if torch.is_tensor(v))


def two_updates(model, optimizer, data, targets, rng, support=1024, query=256):
    records = []
    for step in (1, 2):
        mitra.check_deadline()
        tick = time.monotonic()
        support_ids, query_ids = sample_episode(len(data), rng, support, query)
        before = block_hashes(model)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = mitra.literal_learning_rate(step)
        model.train()
        logits = mitra.model_forward(model, data[support_ids], targets[support_ids], data[query_ids])
        if logits.shape != (query, 10):
            raise ValueError('Training must retain the native ten-class head.')
        loss = F.cross_entropy(logits, targets[query_ids])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite CE.')
        loss.backward()
        grad_blocks = set()
        gradients = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('Nonfinite gradient: ' + name)
                gradients += parameter.numel()
                key = '.'.join(name.split('.')[:2]) if name.startswith('layers.') else name.split('.')[0]
                if bool(torch.count_nonzero(parameter.grad)):
                    grad_blocks.add(key)
        if any(f'layers.{i}' not in grad_blocks for i in range(len(model.layers))):
            raise RuntimeError('A transformer block has no nonzero gradient.')
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError('Nonfinite updated weights.')
        after = block_hashes(model)
        changed = {key: before[key] != after[key] for key in before}
        if any(not changed[f'layers.{i}'] for i in range(len(model.layers))):
            raise RuntimeError('A transformer block failed to update.')
        state_steps = [int(state['step'].item()) for state in optimizer.state.values()]
        if not state_steps or set(state_steps) != {step}:
            raise RuntimeError('AdamW state did not persist across updates.')
        records.append(dict(step=step, lr=optimizer.param_groups[0]['lr'],
            support_rows=len(support_ids), query_rows=len(query_ids),
            support_ids_sha256=mitra.ids_hash(support_ids), query_ids_sha256=mitra.ids_hash(query_ids),
            disjoint=not bool(np.intersect1d(support_ids, query_ids).size),
            parameters_with_finite_grad=gradients, nonzero_gradient_blocks=sorted(grad_blocks),
            updated_blocks=changed, optimizer_state_tensor_bytes=optimizer_bytes(optimizer),
            optimizer_state_steps=sorted(set(state_steps)), seconds=time.monotonic()-tick,
            resources=mitra.snapshot(data.device.type)))
        print(json.dumps(records[-1]), flush=True)
        del logits, loss
    optimizer.zero_grad(set_to_none=True)
    return records


def check_reload(before, after):
    if before.shape != (RECIPE['final_query'], 2) or after.shape != before.shape:
        raise ValueError('Incomplete final inference.')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise FloatingPointError('Nonfinite inference.')
    difference = float(np.max(np.abs(before-after)))
    if difference > 2e-6 or not np.array_equal(before[:, 1] >= .5, after[:, 1] >= .5):
        raise RuntimeError('Reload probability/threshold mismatch.')
    return difference


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT/'artifacts/run_038/protocol.json').read_text())
    env = mitra.environment_lock()
    if hashlib.sha256(env.encode()).hexdigest() != original['mitra_environment_sha256']:
        raise RuntimeError('Original frozen Mitra environment required.')
    sources = dict(original['sources'])
    for name, sha in sources.items():
        if mitra.sha256(ROOT/name) != sha:
            raise RuntimeError('Frozen Mitra source changed: '+name)
    for name in ('mitra_full_episode_probe.py', 'test_mitra_full_episode_probe.py',
                 'reports/mitra_adaptation_review.json'):
        sources[name] = mitra.sha256(ROOT/name)
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    mitra.verify_checkpoint()
    record = dict(recipe=RECIPE, sources=sources, checkpoint_sha256=mitra.WEIGHT_SHA,
        checkpoint_directory=str(mitra.CHECKPOINT_DIR), environment_sha256=hashlib.sha256(env.encode()).hexdigest(),
        backend=mitra.SETTINGS, commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    (output/'requirements.lock').write_text(env)
    (output/'freeze.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Fresh probe output required; existing evidence is immutable.')
    start = time.monotonic()
    result = dict(status='incomplete', synthetic_only=True, candidate_authorized=False,
                  accuracy_computed=False, recipe=RECIPE)
    predictor = model = optimizer = None
    def timeout(*_):
        raise TimeoutError('Entire synthetic probe exceeded 1200 seconds; no fallback.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1200)
    try:
        record = freeze(output)
        mitra.DEADLINE = start + 1200
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS required; no CPU fallback.')
        torch.mps.set_per_process_memory_fraction(.65)
        result['before'] = mitra.snapshot('mps')
        result['allocator_limit_bytes'] = int(torch.mps.recommended_max_memory()*.65)
        mitra.seed_everything(42, 'mps')
        raw = mitra.synthetic_frame(8693, 42)
        labels = np.arange(8693, dtype=np.int64) % 2
        raw_query = mitra.synthetic_frame(2318, 43)  # Never passed to prep.fit.
        prep = mitra.MitraPreprocessor().fit(raw, labels, 42)
        if prep.n_features != 13:
            raise RuntimeError('Synthetic schema must preserve all thirteen features.')
        support = prep.transform(raw)
        data = torch.as_tensor(support, device='mps')
        targets = torch.as_tensor(labels, device='mps')
        model, config = mitra.load_initial_model(mitra.CHECKPOINT_DIR, 'mps')
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 75670026:
            raise RuntimeError('Every pretrained parameter must remain trainable.')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('Model weights must remain FP32.')
        optimizer = mitra.optimizer_for(model)
        result['updates'] = two_updates(model, optimizer, data, targets, np.random.default_rng(42))
        tick = time.monotonic()
        validation = mitra.predict_arrays(model, data[:3708], targets[:3708],
            torch.as_tensor(prep.transform(raw_query.iloc[:928]), device='mps'), chunk=2318)
        if validation.shape != (928, 2) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Validation incomplete or gradients retained.')
        result['validation'] = dict(support_rows=3708, query_rows=928,
            optimizer_state_tensor_bytes=optimizer_bytes(optimizer), seconds=time.monotonic()-tick,
            resources=mitra.snapshot('mps'))
        del optimizer, data, targets, validation
        optimizer = None
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        del model
        model = None
        mitra.release_memory()
        predictor = mitra.MitraPredictor(prep, config, state, support, labels, 'mps',
            metadata=dict(resource_only=True, recipe=RECIPE, original_checkpoint_sha256=mitra.WEIGHT_SHA), query_chunk=2318)
        del state
        tick = time.monotonic()
        expected = predictor.predict_proba(raw_query)
        result['final_inference'] = dict(support_rows=len(predictor.support), query_rows=len(raw_query),
            seconds=time.monotonic()-tick, resources=mitra.snapshot('mps'))
        predictor.release()
        artifact = output/'synthetic_adapted.joblib'
        joblib.dump(predictor, artifact)
        del predictor
        predictor = joblib.load(artifact)
        if predictor.device != 'cpu' or predictor.query_chunk != 2318 or len(predictor.labels) != 8693:
            raise RuntimeError('Reload context/recipe differs.')
        predictor.device = 'mps'
        actual = predictor.predict_proba(raw_query)
        result['reload_max_abs_difference'] = check_reload(expected, actual)
        result['artifact_sha256'] = mitra.sha256(artifact)
        for name, sha in record['sources'].items():
            if mitra.sha256(ROOT/name) != sha:
                raise RuntimeError('Source changed during probe: '+name)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        optimizer = model = None
        if predictor is not None:
            predictor.release()
        mitra.release_memory()
        result['total_seconds'] = time.monotonic()-start
        if output.exists():
            (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
