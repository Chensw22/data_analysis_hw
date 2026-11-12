"""Mitra last-block adaptation core and synthetic resource gates; no CV driver.

The unchanged pretrained prefix is evaluated once per episode. The final block's
support-update branch is dead with respect to query logits and is omitted. Its
input-support normalization and key/value projections remain differentiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import torch
from torch.nn import functional as F

import mitra_candidate as base


def configure(model):
    if model.use_flash_attn or not len(model.layers):
        raise ValueError('This reviewed adapter requires a non-Flash model with layers.')
    prefix = f'layers.{len(model.layers)-1}.'
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith((prefix, 'final_layer_norm.', 'final_layer.')))
        parameter.grad = None
    return model


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == 'mps')


def synchronize(device):
    if torch.device(device).type == 'mps':
        torch.mps.synchronize()


def validate_inputs(model, support, labels, query):
    if model.use_flash_attn:
        raise ValueError('Flash layout has not been adapted.')
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError('Support/query must have the same two-dimensional schema.')
    if not len(support) or not len(query) or labels.shape != (len(support),):
        raise ValueError('Empty or misaligned support/query.')
    expected = f'layers.{len(model.layers)-1}.'
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != name.startswith((expected, 'final_layer_norm.', 'final_layer.')):
            raise ValueError('Configure the exact last-block parameter boundary first.')


@torch.no_grad()
def frozen_prefix(model, support, labels, query):
    """Return detached block-entry representations, never trainable projections.

This cache is episode-specific: support labels and the quantile transformation
depend on the exact support set. A caller must not reuse it for a new episode.
"""
    validate_inputs(model, support, labels, query)
    with amp(support.device):
        xs, xq = support.clone()[None], query.clone()[None]
        ps = torch.zeros((1, len(support)), dtype=torch.bool, device=support.device)
        pq = torch.zeros((1, len(query)), dtype=torch.bool, device=query.device)
        pf = torch.zeros((1, support.shape[1]), dtype=torch.bool, device=support.device)
        xs, xq = model.x_quantile(xs, xq, ps, pf)
        xs, xq = model.x_embedding(xs), model.x_embedding(xq)
        ys, yq = model.y_embedding(labels.clone()[None], ps, len(query))
        s, q = torch.cat((ys, xs), dim=2), torch.cat((yq, xq), dim=2)
        pf = torch.cat((torch.zeros((1, 1), dtype=torch.bool, device=support.device), pf), dim=1)
        for layer in model.layers[:-1]:
            s, q = layer(s, q, None, None, 1, ps, pq, pf)
    return s.detach(), q.detach()


def query_tail(model, support_entry, query_entry):
    """Exact query path of the last non-Flash block, including support gradients."""
    with amp(query_entry.device):
        layer = model.layers[-1]
        b, n, f, d = query_entry.shape
        s = layer.layer_norm1(support_entry)  # Must NOT be cached/detached.
        q = layer.layer_norm1(query_entry)
        sf = s.permute(0, 2, 1, 3).reshape(b*f, s.shape[1], d)
        qf = q.permute(0, 2, 1, 3).reshape(b*f, n, d)
        row = layer.attention1(qf, sf, sf).reshape(b, f, n, d).permute(0, 2, 1, 3)
        q = query_entry + row
        q = q + layer.linear2(F.gelu(layer.linear1(layer.layer_norm2(q))))
        qf = layer.layer_norm3(q).reshape(b*n, f, d)
        q = q + layer.attention2(qf, qf, qf).reshape(b, n, f, d)
        q = q + layer.linear4(F.gelu(layer.linear3(layer.layer_norm4(q))))
        # Norm and head operate independently per feature token. Only label token
        # is consumed by the upstream classification loss and prediction.
        logits = model.final_layer(model.final_layer_norm(q[:, :, 0, :]))
    return logits[0].float()


def split_forward(model, support, labels, query, microbatch=128):
    if microbatch < 1:
        raise ValueError('Positive query microbatch required.')
    s, q = frozen_prefix(model, support, labels, query)
    return torch.cat([query_tail(model, s, q[:, start:start+microbatch])
                      for start in range(0, len(query), microbatch)])


def assert_disjoint(support_ids, query_ids, n_support, n_query):
    a, b = np.asarray(support_ids), np.asarray(query_ids)
    if (a.ndim != 1 or b.ndim != 1 or len(a) != n_support or len(b) != n_query
            or len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b)
            or np.intersect1d(a, b).size):
        raise ValueError('Unique disjoint support/query IDs are required.')


def backward_episode(model, support, support_labels, query, query_labels,
                     support_ids, query_ids, microbatch=128):
    """One full-episode gradient; optimizer.zero_grad/step are caller-owned."""
    assert_disjoint(support_ids, query_ids, len(support), len(query))
    if microbatch < 1 or query_labels.shape != (len(query),):
        raise ValueError('Invalid microbatch or query targets.')
    synchronize(support.device)
    begin = time.monotonic()
    s, q = frozen_prefix(model, support, support_labels, query)
    synchronize(support.device)
    after_prefix = time.monotonic()
    value = 0.
    for start in range(0, len(query), microbatch):
        stop = min(start+microbatch, len(query))
        logits = query_tail(model, s, q[:, start:stop])
        loss = F.cross_entropy(logits, query_labels[start:stop], reduction='sum') / len(query)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite full-head query CE.')
        loss.backward()
        value += float(loss.detach().cpu())
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite gradient.')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen prefix unexpectedly has gradients.')
    synchronize(support.device)
    return dict(training_full_head_ce=value, prefix_seconds=after_prefix-begin,
                backward_tail_seconds=time.monotonic()-after_prefix,
                support_rows=len(support), query_rows=len(query),
                microbatches=(len(query)+microbatch-1)//microbatch,
                support_ids_sha256=base.ids_hash(support_ids),
                query_ids_sha256=base.ids_hash(query_ids), disjoint=True)


def optimizer_bytes(optimizer):
    return sum(v.numel()*v.element_size() for state in optimizer.state.values()
               for v in state.values() if torch.is_tensor(v))


def frozen_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            result.update(name.encode())
            result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def optimizer_step(model, optimizer, support, labels, query, targets,
                   support_ids, query_ids, step, microbatch=128):
    lr = base.literal_learning_rate(step)
    for group in optimizer.param_groups:
        group['lr'] = lr
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    record = backward_episode(model, support, labels, query, targets,
                              support_ids, query_ids, microbatch)
    nonzero = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and bool(p.grad.abs().max() > 0)]
    synchronize(support.device)
    begin = time.monotonic()
    optimizer.step()
    synchronize(support.device)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite parameters after AdamW.')
    changed = [n for n, p in model.named_parameters()
               if p.requires_grad and not torch.equal(p.detach(), before[n])]
    last = f'layers.{len(model.layers)-1}.'
    if not any(n.startswith(last) for n in nonzero) or not any(n.startswith(last) for n in changed):
        raise RuntimeError('Last block must have a real gradient and parameter update.')
    record.update(step=step, optimizer_steps=1, learning_rate=lr,
                  optimizer_seconds=time.monotonic()-begin,
                  optimizer_state_tensor_bytes=optimizer_bytes(optimizer),
                  nonzero_gradient_names=nonzero, changed_parameter_names=changed,
                  finite_gradients=True, finite_parameters=True,
                  resources=base.snapshot(str(support.device)))
    return record


def freeze_probe(output):
    base.freeze_sources(output)
    manifest = json.loads((output/'freeze.json').read_text())
    for filename in ['mitra_lastblock.py', 'test_mitra_lastblock.py', 'reports/top5_strategy_review.json']:
        source = base.ROOT/filename
        dest = output/'source'/filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest['source_sha256'][filename] = base.sha256(source)
    (output/'freeze.json').write_text(json.dumps(manifest, indent=2)+'\n')


def probe(args):
    """Synthetic-only resource measurements; never load competition labels."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    freeze_probe(output)
    shapes = {'small': (520, 130, 650), 'outer': (3708, 928, 4636),
              'final': (6954, 1024, 8693)}
    ns, nq, context_rows = shapes[args.stage]
    result = dict(status='running', synthetic_labels_only=True, accuracy_computed=False,
                  stage=args.stage, device=args.device, support_rows=ns, query_rows=nq,
                  inference_context_rows=context_rows, query_microbatch=128, steps=[])
    model = optimizer = predictor = restored = None
    started = time.monotonic()
    try:
        if args.device == 'mps':
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS unavailable; no silent CPU fallback.')
            torch.mps.set_per_process_memory_fraction(base.SETTINGS['mps_memory_fraction'])
        base.seed_everything(42, args.device)
        model, config = base.load_initial_model(args.checkpoint_dir, args.device)
        configure(model)
        result.update(checkpoint_sha256=base.WEIGHT_SHA,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      settings=dict(base.SETTINGS), before=base.snapshot(args.device))
        frame = base.synthetic_frame(context_rows+128, 42)
        targets = np.arange(context_rows, dtype=np.int64)%2
        prep = base.MitraPreprocessor().fit(frame.iloc[:context_rows], targets, 42)
        array = prep.transform(frame.iloc[:context_rows])
        data = torch.as_tensor(array, device=args.device)
        y = torch.as_tensor(targets, device=args.device)
        generator = torch.Generator().manual_seed(42)
        original_prefix = frozen_digest(model)
        optimizer = base.optimizer_for(model)
        for step in [1, 2]:
            order = torch.randperm(context_rows, generator=generator).numpy()
            si, qi = order[:ns], order[ns:ns+nq]
            sidx = torch.as_tensor(si, device=args.device)
            qidx = torch.as_tensor(qi, device=args.device)
            record = optimizer_step(model, optimizer, data[sidx], y[sidx], data[qidx], y[qidx],
                                    si, qi, step)
            record['frozen_prefix_unchanged'] = frozen_digest(model) == original_prefix
            if not record['frozen_prefix_unchanged']:
                raise RuntimeError('Frozen prefix weights changed.')
            result['steps'].append(record)
            (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        # Time a fixed validation-prefix construction and one cached-tail pass.
        si = np.arange(ns)
        qi = np.arange(ns, ns+nq)
        synchronize(args.device)
        begin = time.monotonic()
        s, q = frozen_prefix(model, data[:ns], y[:ns], data[ns:ns+nq])
        synchronize(args.device)
        result['validation_prefix_seconds'] = time.monotonic()-begin
        begin = time.monotonic()
        with torch.no_grad():
            for start in range(0, nq, 128):
                logits = query_tail(model, s, q[:, start:start+128])
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Nonfinite cached validation logits.')
        synchronize(args.device)
        result['cached_validation_tail_seconds'] = time.monotonic()-begin
        del s, q, logits
        # Full export-shaped context, with synthetic labels; no subsampling.
        query = torch.as_tensor(prep.transform(frame.iloc[context_rows:]), device=args.device)
        begin = time.monotonic()
        model.eval()
        with torch.no_grad():
            expected = split_forward(model, data, y, query)[:, :2].softmax(-1).cpu().numpy()
        synchronize(args.device)
        result['full_context_split_inference_seconds'] = time.monotonic()-begin
        original_path = base.predict_arrays(model, data, y, query, 128)
        tolerance = 2e-3 if args.device == 'mps' else 2e-6
        split_error = float(np.max(np.abs(original_path-expected)))
        result.update(split_vs_original_max_abs_error=split_error,
                      split_vs_original_threshold_disagreements=int(np.sum(
                          (original_path[:, 1] >= .5) != (expected[:, 1] >= .5))),
                      split_vs_original_atol=tolerance, split_vs_original_rtol=tolerance,
                      probability_path_for_future_scoring='original_MitraPredictor_including_step0')
        if not np.allclose(original_path, expected, atol=tolerance, rtol=tolerance):
            raise AssertionError(f'Original architecture differs from split path: {split_error}')
        result['after_updates'] = base.snapshot(args.device)
        # Upstream portable predictor persists CPU weights/context and restores
        # the original full architecture, independently checking the split path.
        predictor = base.MitraPredictor(prep, config, model.state_dict(), array, targets,
                                        device=args.device, query_chunk=128)
        joblib.dump(predictor, output/'synthetic_model.joblib')
        del optimizer, model, data, y, query
        optimizer = model = None
        predictor.release()
        predictor = None
        base.release_memory()
        restored = joblib.load(output/'synthetic_model.joblib')
        restored.device = args.device
        actual = restored.predict_proba(frame.iloc[context_rows:])
        error = float(np.max(np.abs(actual-original_path)))
        # Persistence is a separate check from split-path floating-point error.
        restore_tolerance = 2e-6
        if not np.allclose(actual, original_path, atol=restore_tolerance, rtol=restore_tolerance):
            raise AssertionError(f'Restored full architecture differs from original full path: {error}')
        result.update(status='passed', roundtrip_max_abs_error=error,
                      roundtrip_bitwise_equal=bool(np.array_equal(actual, original_path)),
                      roundtrip_threshold_disagreements=int(np.sum(
                          (actual[:, 1] >= .5) != (original_path[:, 1] >= .5))),
                      roundtrip_atol=restore_tolerance, roundtrip_rtol=restore_tolerance,
                      total_seconds=time.monotonic()-started)
    except Exception as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__,
                      error=str(error), total_seconds=time.monotonic()-started)
        raise
    finally:
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        if restored is not None:
            restored.release()
        if predictor is not None:
            predictor.release()
        model = optimizer = None
        base.release_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['small', 'outer', 'final'])
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--checkpoint-dir', type=Path, default=base.CHECKPOINT_DIR)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(probe(args), indent=2))


if __name__ == '__main__':
    main()
