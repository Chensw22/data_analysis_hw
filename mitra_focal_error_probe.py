"""One matched fixed15 focal-loss experiment; training labels alone define difficulty."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None, 'FALSE'):
    raise RuntimeError('FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss

import audit_top5_references as audit
import error_rescue_feature_probe as previous
import exaone_validation_data as data
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import mitra_full_episode as training
from mitra_full_episode_probe import sample_episode, optimizer_bytes

ROOT = Path(__file__).resolve().parent
NAME = 'v100_Mitra_focal_gamma2_fixed15'
PLAN = 'reports/error_targeted_20261117/focal_protocol.json'
REPORT = ROOT / 'reports/error_targeted_20261117/focal_result.json'
OUT = ROOT / 'artifacts/error_targeted_20261117/focal_fold0'
SEED, STEPS, BUDGET = 20261112, 15, 2400
CONTROL = ROOT / 'artifacts/error_rescue_20261117/feature_probe_fold0/compact13_fixed15_fold0.npz'
CONTROL_LOG = CONTROL.with_suffix('.json')


def focal_loss(logits, target, gamma=2.):
    ce = F.cross_entropy(logits, target, reduction='none')
    return ((1 - torch.exp(-ce)).pow(gamma) * ce).mean()


def check_loss():
    # Gamma0 must recover CE including its gradient; gamma2 must differentiate the weight.
    logits = torch.tensor([[4., -1., 0.], [-1., 2., 0.], [0., 1., 3.]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0, 0, 1])
    ce = F.cross_entropy(logits, target)
    zero = focal_loss(logits, target, 0.)
    g_ce = torch.autograd.grad(ce, logits, retain_graph=True)[0]
    g_zero = torch.autograd.grad(zero, logits, retain_graph=True)[0]
    assert torch.equal(ce, zero) and torch.equal(g_ce, g_zero)
    logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    expected = (-((1-logp.exp())**2)*logp).mean()
    actual = focal_loss(logits, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
    grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=1e-15)
    assert torch.isfinite(grad).all()
    print('Focal loss CE-limit, direct-formula and full-gradient checks passed.', flush=True)


def focal_step(model, optimizer, tensor, targets, ids, rng, step):
    base.check_deadline()
    started = time.monotonic()
    support, query = sample_episode(len(tensor), rng)
    assert not np.intersect1d(ids[support], ids[query]).size
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = base.literal_learning_rate(step)
    model.train()
    s = torch.as_tensor(support, device=tensor.device)
    q = torch.as_tensor(query, device=tensor.device)
    logits = base.model_forward(model, tensor[s], targets[s], tensor[q])
    if logits.shape != (256, 10):
        raise ValueError('Retain native10-class head and original256 queries.')
    loss = focal_loss(logits, targets[q])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite focal loss.')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise FloatingPointError('Nonfinite gradients.')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite weights.')
    assert {int(s['step'].item()) for s in optimizer.state.values()} == {step}
    result = dict(step=step, focal_loss=float(loss.detach().cpu()),
                  support_ids_sha256=base.ids_hash(ids[support]), query_ids_sha256=base.ids_hash(ids[query]),
                  learning_rate=base.literal_learning_rate(step), optimizer_state_tensor_bytes=optimizer_bytes(optimizer),
                  seconds=time.monotonic()-started)
    optimizer.zero_grad(set_to_none=True)
    return result


def fit_predict(features, train_ids, y_train, query_ids, expected_draws):
    # No validation targets or error masks are passed to this function.
    y_train = base.binary_labels(y_train, len(train_ids))
    assert len(train_ids) == 4636 and len(query_ids) == 2318
    assert not np.intersect1d(train_ids, query_ids).size
    base.seed_everything(SEED, 'mps')
    prep = base.MitraPreprocessor().fit(features.loc[train_ids], y_train, SEED)
    support = prep.transform(features.loc[train_ids])
    assert support.shape == (4636, 13)
    model = optimizer = tensor = targets = predictor = None
    log = dict(status='incomplete', name=NAME, updates=[], fixed_steps=STEPS, gamma=2., alpha=1.)
    try:
        model, config = training.fresh_model('mps')
        optimizer = base.optimizer_for(model)
        tensor = torch.as_tensor(support, device='mps')
        targets = torch.as_tensor(y_train, device='mps')
        rng = np.random.default_rng(SEED)
        for step in range(1, STEPS+1):
            counts.require_false()
            row = focal_step(model, optimizer, tensor, targets, train_ids, rng, step)
            assert (row['support_ids_sha256'], row['query_ids_sha256']) == expected_draws[step-1]
            log['updates'].append(row)
            base.save_training_log(OUT/'training_log.json', log)
            print(f'{NAME} update{step}/15', flush=True)
        optimizer = tensor = targets = None
        base.release_memory()
        predictor = base.MitraPredictor(prep, config, model.state_dict(), support, y_train, 'mps', log, query_chunk=2318)
        model = None
        base.release_memory()
        p = predictor.predict_proba(features.loc[query_ids])[:, 1]
        assert p.shape == (2318,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        log['status'] = 'complete'
        base.save_training_log(OUT/'training_log.json', log)
        return p
    finally:
        if predictor is not None:
            predictor.release()
        model = optimizer = tensor = targets = predictor = None
        base.release_memory()


def run():
    if OUT.exists() or REPORT.exists():
        raise RuntimeError('One new fixed experiment; no overwrite.')
    plan = json.loads((ROOT/PLAN).read_text())
    assert plan['gamma'] == 2 and plan['steps'] == STEPS and plan['seed'] == SEED
    source_names = list(set(previous.OWN + [Path(__file__).name, PLAN,
        'reports/error_rescue_20261117/feature_probe_result.json',
        'reports/error_rescue_20261117/feature_probe_root_audit.json']))
    previous_model, lock = counts.verify_reference()
    source_names += [n for n in previous_model['source_hashes'] if n.startswith('third_party/mitra/')]
    commit = previous.committed(source_names)
    control_report = json.loads((ROOT/'reports/error_rescue_20261117/feature_probe_result.json').read_text())
    assert audit.file_hash(CONTROL) == control_report['evidence_files'][str(CONTROL.relative_to(ROOT))]
    old_freeze = json.loads((CONTROL.parent/'freeze.json').read_text())
    previous.verify_sources(old_freeze)
    control_log = json.loads(CONTROL_LOG.read_text())
    assert control_log['status'] == 'complete' and control_log['optimizer_steps'] == STEPS
    draws = [(s['support_ids_sha256'], s['query_ids_sha256']) for s in control_log['updates']]
    inputs = data.load_data()
    splits, refs, reference_evidence = data.random_references(inputs)
    a, b = splits[0]
    train_ids, query_ids = inputs['dev'][a], inputs['dev'][b]
    cached = np.load(CONTROL, allow_pickle=False)
    np.testing.assert_array_equal(cached['train_ids'], train_ids)
    np.testing.assert_array_equal(cached['query_ids'], query_ids)
    np.testing.assert_array_equal(cached['query_y'], inputs['y'][b])
    OUT.mkdir(parents=True)
    frozen = dict(source_commit=commit, sources={n:audit.file_hash(ROOT/n) for n in source_names},
                  environment_sha256=hashlib.sha256(lock.encode()).hexdigest(), protocol=plan)
    frozen['sources'].update(inputs['evidence']); frozen['sources'].update(reference_evidence)
    frozen['sources'].update({str(p.relative_to(ROOT)):audit.file_hash(p) for p in [CONTROL, CONTROL_LOG]})
    audit.json_write(OUT/'freeze.json', frozen)
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Fixed focal experiment exceeded2400s.')
    handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    base.DEADLINE = started+BUDGET
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS required; no fallback.')
        torch.set_num_threads(4)
        torch.mps.set_per_process_memory_fraction(.65)
        p = fit_predict(inputs['features'], train_ids, inputs['y'][a].copy(), query_ids, draws)
        y, control = inputs['y'][b], cached['prob']
        masks = dict(all=np.ones(len(y), dtype=bool), five_common_wrong=cached['common_wrong_mask'],
                     five_common_correct=cached['common_correct_mask'])
        x = inputs['features'].loc[query_ids]
        masks['Earth_asleep'] = (x.HomePlanet.eq('Earth') & x.CryoSleep.eq('True')).to_numpy()
        masks['Earth_awake'] = (x.HomePlanet.eq('Earth') & x.CryoSleep.eq('False')).to_numpy()
        result = dict(status='complete_single_fold_adaptive_diagnostic', name=NAME, source_commit=commit,
            accuracy=float(np.mean((p>=.5)==y)), log_loss=float(log_loss(y, p, labels=[0,1])),
            control_accuracy=float(np.mean((control>=.5)==y)), control_log_loss=float(log_loss(y, control, labels=[0,1])),
            paired={name:previous.paired(p, control, y, mask) for name,mask in masks.items()},
            common_wrong_correct=int((((p>=.5)==y)&masks['five_common_wrong']).sum()),
            common_correct_damaged=int((((p>=.5)!=y)&masks['five_common_correct']).sum()),
            fixed_steps=STEPS, matched_episode_ids=True, seconds=time.monotonic()-started,
            interpretation='Only loss changed versus cached fixed15compact13; one reused fold, no stable-superiority claim or automatic upload.')
        np.savez(OUT/'predictions.npz', prob=p, control=control, train_ids=train_ids, query_ids=query_ids,
                 y=y, **{k+'_mask':v for k,v in masks.items()})
        result['prediction_sha256'] = audit.file_hash(OUT/'predictions.npz')
        previous.verify_sources(frozen)
        audit.json_write(REPORT, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        audit.json_write(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        base.release_memory()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check_loss() if args.check else run()
