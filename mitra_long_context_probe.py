"""Fixed synthetic 3072/256 resource gate, externally supervised; no accuracy/CV.

Existing v81 layer and SDPA checkpointing, attention backend, FP32 weights and
MPS bf16 autocast are unchanged. A successful probe never authorizes a candidate.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

import mitra_candidate as base
import mitra_full_episode_probe as original

ROOT = Path(__file__).resolve().parent
DESIGN = 'reports/mitra_long_context_resource_design.json'
DESIGN_SHA = '71da8c8d75835137d79fb62e900f10cc8c5d81520900db695ec76d50a69648af'
OLD = 'reports/mitra_full_episode_resources'
OWN = ['mitra_long_context_probe.py', 'test_mitra_long_context_probe.py', 'quota_guard.py', DESIGN]
BUDGET = 1200
PARAMETERS = 75670026
OPTIMIZER_BYTES = 605361776
RECIPE = dict(seed=42, pool_rows=8693, support_rows=3072, query_rows=256,
              steps=4, validation_shapes=[[3708, 928], [6954, 1739]], query_chunk=2318,
              sequence=['step1', 'step2', 'validation3708x928', 'step3', 'validation6954x1739', 'step4'],
              weights_dtype='float32', mps_autocast='bfloat16', mps_memory_fraction=.65, threads=4,
              native_head=10, learning_rate=3e-6, weight_decay=.3, budget_seconds=BUDGET,
              trainable_parameters=PARAMETERS, existing_double_checkpoint_unchanged=True,
              sampler='support=concat(perm[:1024],perm[1280:3328]);query=perm[1024:1280]')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Fixed complete synthetic probe budget exhausted.')


def paired_episode(n, rng, reference_rng):
    if n < 3328:
        raise ValueError('At least3328 distinct rows required; no smaller-context rescue.')
    perm = rng.permutation(n)
    support = np.concatenate([perm[:1024], perm[1280:3328]])
    query = perm[1024:1280]
    old_support, old_query = original.sample_episode(n, reference_rng)
    if (not np.array_equal(support[:1024], old_support) or not np.array_equal(query, old_query)
            or rng.bit_generator.state != reference_rng.bit_generator.state
            or len(np.unique(support)) != 3072 or len(np.unique(query)) != 256
            or np.intersect1d(support, query).size):
        raise RuntimeError('Paired v81 query/support-prefix/RNG counterfactual failed.')
    return support, query


def quota_guard(session):
    from quota_guard import latest_weekly, assess
    try:
        allowed = assess(latest_weekly(session), reserve=30)['can_start_more_work']
    except (OSError, ValueError, TypeError, KeyError):
        allowed = False
    if not allowed:
        raise RuntimeError('Fresh reserve30 quota guard did not allow this probe.')


def require_committed(names):
    names = sorted(set(names))
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *names], cwd=ROOT).decode()
    if (set(filter(None, tracked.split('\0'))) != set(names)
            or subprocess.check_output(['git', 'status', '--porcelain', '--', *names], cwd=ROOT).strip()):
        raise RuntimeError('All probe sources/design/evidence must be committed before MPS.')


def check_frozen(frozen):
    for name, digest in frozen['sources'].items():
        if base.sha256(ROOT/name) != digest:
            raise RuntimeError('Frozen source/evidence changed: '+name)
    if hashlib.sha256(base.environment_lock().encode()).hexdigest() != frozen['environment_sha256']:
        raise RuntimeError('Use the unchanged original Mitra environment.')
    base.verify_checkpoint()
    if base.WEIGHT_SHA != frozen['checkpoint_sha256'] or base.SETTINGS != frozen['backend']:
        raise RuntimeError('Pinned checkpoint/backend differs.')


def contract():
    if base.sha256(ROOT/DESIGN) != DESIGN_SHA:
        raise RuntimeError('Fixed resource design changed.')
    design = read(ROOT/DESIGN)
    old = read(ROOT/OLD/'freeze.json')
    sources = dict(old['sources'])
    for name, digest in design['evidence_files_sha256'].items():
        if name in sources and sources[name] != digest:
            raise RuntimeError('Historical evidence source disagreement.')
        sources[name] = digest
    sources.update({name:base.sha256(ROOT/name) for name in OWN})
    lock = base.environment_lock()
    if lock != (ROOT/OLD/'requirements.lock').read_text():
        raise RuntimeError('Exact existing full-resource environment lock required.')
    frozen = dict(recipe=RECIPE, sources=sources, design_sha256=DESIGN_SHA,
                  checkpoint_sha256=old['checkpoint_sha256'], backend=old['backend'],
                  environment_sha256=old['environment_sha256'],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    check_frozen(frozen)
    require_committed(sources)
    return frozen, lock


def synchronized_snapshot(device):
    if device == 'mps':
        torch.mps.synchronize()
    return base.snapshot(device)


def moment_identity(optimizer):
    return tuple((id(p), id(s['exp_avg']), id(s['exp_avg_sq'])) for p, s in optimizer.state.items())


def optimizer_steps(optimizer):
    return sorted({int(state['step'].item()) for state in optimizer.state.values()})


def update(model, optimizer, tensor, labels, rng, reference_rng, step, checkpoint):
    base.check_deadline()
    device = tensor.device.type
    synchronized_snapshot(device)
    started = time.monotonic()
    support, query = paired_episode(len(tensor), rng, reference_rng)
    checkpoint(f'step{step}:before_forward', synchronized_snapshot(device))
    before = original.block_hashes(model)
    previous_moments = moment_identity(optimizer)
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = base.literal_learning_rate(step)
    model.train()
    s, q = torch.as_tensor(support, device=tensor.device), torch.as_tensor(query, device=tensor.device)
    logits = base.model_forward(model, tensor[s], labels[s], tensor[q])
    if logits.shape != (256, 10):
        raise RuntimeError('Native ten-class256-query head required.')
    loss = F.cross_entropy(logits, labels[q])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training CE.')
    checkpoint(f'step{step}:after_forward', synchronized_snapshot(device))
    loss.backward()
    grad_blocks, grad_count = set(), 0
    for name, p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise FloatingPointError('Missing/nonfinite gradient: '+name)
        grad_count += p.numel()
        if name.startswith('layers.') and bool(torch.count_nonzero(p.grad)):
            grad_blocks.add('.'.join(name.split('.')[:2]))
    blocks = {f'layers.{i}' for i in range(len(model.layers))}
    if not blocks <= grad_blocks:
        raise RuntimeError('Every transformer block needs nonzero gradients.')
    checkpoint(f'step{step}:after_backward', synchronized_snapshot(device))
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite updated weights.')
    if optimizer_steps(optimizer) != [step] or (previous_moments and moment_identity(optimizer) != previous_moments):
        raise RuntimeError('Persistent optimizer moments/step counter changed.')
    after = original.block_hashes(model)
    changed = {k: before[k] != after[k] for k in before}
    if not all(changed[k] for k in blocks):
        raise RuntimeError('Every transformer block must update.')
    snapshot = synchronized_snapshot(device)
    checkpoint(f'step{step}:after_optimizer', snapshot)
    value = float(loss.detach().cpu())
    del logits, loss, s, q
    optimizer.zero_grad(set_to_none=True)
    synchronized_snapshot(device)
    return dict(step=step, lr=optimizer.param_groups[0]['lr'], support_rows=3072, query_rows=256,
                disjoint_ids=True, query_equal_v81=True, support_prefix_equal_v81=True, rng_state_equal_v81=True,
                support_ids_sha256=base.ids_hash(support), query_ids_sha256=base.ids_hash(query),
                training_full_head_ce=value, parameters_with_finite_grad=grad_count,
                nonzero_gradient_blocks=sorted(grad_blocks), updated_blocks=changed,
                optimizer_state_tensor_bytes=original.optimizer_bytes(optimizer), optimizer_state_steps=[step],
                seconds=time.monotonic()-started, resources=snapshot)


def validate(model, optimizer, tensor, labels, query, support_rows, query_rows, checkpoint):
    base.check_deadline()
    device = tensor.device.type
    synchronized_snapshot(device)
    started = time.monotonic()
    moments, steps, size = moment_identity(optimizer), optimizer_steps(optimizer), original.optimizer_bytes(optimizer)
    if not moments or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Validation needs resident moments and cleared gradients.')
    checkpoint(f'validation{support_rows}x{query_rows}:before', synchronized_snapshot(device))
    p = base.predict_arrays(model, tensor[:support_rows], labels[:support_rows], query[:query_rows], chunk=2318)
    if (p.shape != (query_rows, 2) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1., atol=2e-6, rtol=0)
            or any(v.grad is not None for v in model.parameters())
            or moments != moment_identity(optimizer) or steps != optimizer_steps(optimizer)
            or size != original.optimizer_bytes(optimizer)):
        raise RuntimeError('Full-context validation/moment identity/numerics failed.')
    snapshot = synchronized_snapshot(device)
    checkpoint(f'validation{support_rows}x{query_rows}:after', snapshot)
    return dict(support_rows=support_rows, query_rows=query_rows, full_support=True, gradients_cleared=True,
                moments_unchanged=True, optimizer_state_tensor_bytes=size, optimizer_state_steps=steps,
                probabilities_finite_normalized=True, resources=snapshot, seconds=time.monotonic()-started)


def sequence(model, optimizer, tensor, labels, query, checkpoint, persist):
    rng, reference = np.random.default_rng(42), np.random.default_rng(42)
    updates, validations = [], []
    for step in range(1, 5):
        row = update(model, optimizer, tensor, labels, rng, reference, step, checkpoint)
        updates.append(row); persist(updates, validations)
        if step in [2, 3]:
            n, q = (3708, 928) if step == 2 else (6954, 1739)
            validations.append(validate(model, optimizer, tensor, labels, query, n, q, checkpoint))
            persist(updates, validations)
    return updates, validations


def cost_gate(updates, validations):
    if [u['step'] for u in updates] != [1, 2, 3, 4] or [(v['support_rows'], v['query_rows']) for v in validations] != [(3708, 928), (6954, 1739)]:
        raise RuntimeError('No projection from partial or reordered resource sequence.')
    times = [u['seconds'] for u in updates]+[v['seconds'] for v in validations]
    if not all(np.isfinite(t) and t > 0 for t in times):
        raise RuntimeError('Positive finite synchronized durations required.')
    warm = max(u['seconds'] for u in updates[1:])
    screen = 1.25*(300*warm+153*validations[0]['seconds']+600)
    export = 1.25*(100*warm+51*validations[1]['seconds']+600)
    return dict(warm_update_max_seconds=warm, screen_projection_seconds=screen,
                full_export_projection_seconds=export, screen_ceiling_seconds=43200,
                full_export_ceiling_seconds=18000, passed=screen <= 43200 and export <= 18000,
                automatic_cv_authorization=False, automatic_export_authorization=False)


def qualify(record):
    if record.get('status') != 'worker_complete' or record.get('release_complete') is not True:
        raise RuntimeError('Complete worker and release evidence required.')
    if record.get('recipe') != RECIPE or len(record['updates']) != 4 or len(record['validations']) != 2:
        raise RuntimeError('Wrong resource recipe/coverage.')
    for step, row in enumerate(record['updates'], 1):
        if (row['step'] != step or row['optimizer_state_steps'] != [step]
                or row['parameters_with_finite_grad'] != PARAMETERS
                or row['optimizer_state_tensor_bytes'] != OPTIMIZER_BYTES
                or row['support_rows'] != 3072 or row['query_rows'] != 256
                or abs(row['lr']-base.literal_learning_rate(step)) > 1e-15
                or any(row.get(k) is not True for k in ['disjoint_ids', 'query_equal_v81', 'support_prefix_equal_v81', 'rng_state_equal_v81'])
                or any(row['updated_blocks'].get(f'layers.{k}') is not True for k in range(12))):
            raise RuntimeError('Incomplete update/gradient/persistent-state evidence.')
    for row, step in zip(record['validations'], [2, 3]):
        if (row['optimizer_state_steps'] != [step] or row['optimizer_state_tensor_bytes'] != OPTIMIZER_BYTES
                or any(row.get(k) is not True for k in ['full_support', 'gradients_cleared', 'moments_unchanged', 'probabilities_finite_normalized'])):
            raise RuntimeError('Full validation and unchanged moments required.')
    return cost_gate(record['updates'], record['validations'])


def worker(output, frozen, deadline):
    """Spawned child only. Parent owns final success/timeout classification."""
    output = Path(output)
    result = dict(status='incomplete', recipe=RECIPE, synthetic_only=True, accuracy_computed=False,
                  candidate_authorized=False, updates=[], validations=[], snapshots=[], release_complete=False)
    model = optimizer = tensor = labels = query = prep = raw = raw_query = None
    base.DEADLINE = deadline
    started = time.monotonic()
    def persist(updates=None, validations=None):
        if updates is not None: result['updates'] = updates
        if validations is not None: result['validations'] = validations
        result['worker_seconds'] = time.monotonic()-started
        write(output/'partial.json', result)
    def checkpoint(stage, values):
        check_deadline(deadline)
        result['current_stage'] = stage
        result['snapshots'].append(dict(stage=stage, **values))
        persist()
    try:
        check_deadline(deadline); check_frozen(frozen)
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None, '0') or not torch.backends.mps.is_available():
            raise RuntimeError('MPS required without fallback.')
        torch.set_num_threads(4); torch.mps.set_per_process_memory_fraction(.65)
        base.seed_everything(42, 'mps')
        checkpoint('before_load', synchronized_snapshot('mps'))
        result['allocator_limit_bytes'] = int(torch.mps.recommended_max_memory()*.65)
        raw = base.synthetic_frame(8693, 42)
        y = np.arange(8693, dtype=np.int64) % 2
        raw_query = base.synthetic_frame(1739, 43)
        prep = base.MitraPreprocessor().fit(raw, y, 42)
        if prep.n_features != 13:
            raise RuntimeError('All thirteen synthetic features must remain active.')
        tensor = torch.as_tensor(prep.transform(raw), device='mps')
        labels = torch.as_tensor(y, device='mps')
        query = torch.as_tensor(prep.transform(raw_query), device='mps')
        model, _ = base.load_initial_model(base.CHECKPOINT_DIR, 'mps')
        if (sum(p.numel() for p in model.parameters()) != PARAMETERS or len(model.layers) != 12
                or any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
            raise RuntimeError('Original all-trainable FP32 model required.')
        optimizer = base.optimizer_for(model)
        sequence(model, optimizer, tensor, labels, query, checkpoint, persist)
        check_deadline(deadline); check_frozen(frozen)
    except BaseException as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        persist()
    finally:
        # No bound methods/loop parameter references retain model tensors here.
        model = optimizer = tensor = labels = query = prep = raw = raw_query = None
        try:
            base.release_memory()
            result['resources_after_release'] = synchronized_snapshot('mps')
            check_deadline(deadline)
            result['release_complete'] = True
            if result['status'] == 'incomplete':
                result['status'] = 'worker_complete'
                result['cost_gate'] = qualify(result)
        except BaseException as error:
            result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        persist()
        write(output/'worker_result.json', result)


def supervise(process, deadline):
    """Kill only our spawned probe child at deadline, even inside native MPS."""
    process.join(timeout=max(0., deadline-time.monotonic()))
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        if process.is_alive():
            raise RuntimeError('Probe child could not be reaped after SIGKILL.')
        raise TimeoutError('External supervisor killed the probe child at the1200s deadline.')
    if process.exitcode != 0:
        raise RuntimeError('Probe child exited abnormally: '+str(process.exitcode))


def run(output, session_file):
    output = Path(output)
    if output.exists():
        raise FileExistsError('No overwrite/resume of any prior probe evidence.')
    quota_guard(session_file)
    started = time.monotonic(); deadline = started+BUDGET
    output.mkdir(parents=True)
    process = None
    result = dict(status='failed_inconclusive', synthetic_only=True, accuracy_computed=False,
                  candidate_authorized=False, resource_qualification=False, recipe=RECIPE,
                  externally_supervised=True, budget_seconds=BUDGET)
    try:
        frozen, lock = contract()
        write(output/'freeze.json', frozen)
        (output/'requirements.lock').write_text(lock)
        for name in frozen['sources']:
            target = output/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        check_deadline(deadline)
        process = multiprocessing.get_context('spawn').Process(target=worker, args=(str(output), frozen, deadline))
        process.start()
        supervise(process, deadline)
        check_deadline(deadline)
        child = read(output/'worker_result.json')
        if child.get('status') != 'worker_complete':
            result.update(child_error_type=child.get('error_type'), child_error=child.get('error'),
                          child_current_stage=child.get('current_stage'))
            raise RuntimeError('Synthetic worker failed: '+str(child.get('error_type'))+': '+str(child.get('error')))
        cost = qualify(child)
        check_frozen(frozen); require_committed(frozen['sources']); check_deadline(deadline)
        result.update(status='complete', resource_qualification=True, cost_gate=cost,
                      cost_qualified_for_root_review=cost['passed'],
                      next_action=('Separate committed root decision required; no CV/export authorization.' if cost['passed']
                                   else 'Archive this3072branch: fixed prospective cost gate failed; no budget/context rescue.'),
                      worker_result_sha256=base.sha256(output/'worker_result.json'),
                      **{k:child[k] for k in ['updates', 'validations', 'snapshots', 'release_complete', 'resources_after_release']})
        result['total_seconds'] = time.monotonic()-started
        write(output/'result.json', result)
        check_deadline(deadline)  # Includes successful final serialization in the budget.
        return result
    except BaseException as error:
        if process is not None and process.is_alive():
            process.kill(); process.join(timeout=5)
        result.update(status='failed_inconclusive', resource_qualification=False,
                      cost_qualified_for_root_review=False, error_type=type(error).__name__, error=str(error),
                      total_seconds=time.monotonic()-started,
                      next_action='Archive this3072branch; no support/backend/precision/budget rescue.')
        if isinstance(error, TimeoutError):
            write(output/'external_timeout.json', dict(status='external_deadline_exceeded',
                  seconds=result['total_seconds'], budget_seconds=BUDGET,
                  worker_pid=process.pid if process is not None else None,
                  child_reaped=process is None or not process.is_alive(), partial_evidence_preserved=True))
        if (output/'partial.json').exists():
            result['partial_evidence_sha256'] = base.sha256(output/'partial.json')
        write(output/'result.json', result)
        raise
    finally:
        if process is not None and not process.is_alive():
            process.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/probes/mitra_long_context_3072'))
    parser.add_argument('--session-file', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.session_file), indent=2))
