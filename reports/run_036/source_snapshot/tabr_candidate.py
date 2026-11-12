"""Frozen TabR-S method screening with honest training-only retrieval libraries."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import iterate
from third_party.tabr.model import TabRModel

BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
NAME = 'v68_TabR_S'
NUMERIC = ['Age', 'CabinNumber', *iterate.base.SPEND]
CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
SETTINGS = dict(d_main=265, context_size=96, encoder_n_blocks=0, predictor_n_blocks=1,
    d_multiplier=2, context_dropout=0.38920071545944357,
    dropout0=0.38852797479169876, dropout1=0.,
    learning_rate=0.0003121273641315169, weight_decay=1.2260352006404615e-6,
    batch_size=256, max_epochs=200, patience=16, inner_validation_fraction=.2,
    epoch_metric='log_loss', preprocessing='TabM_compact13_and_7_categorical_missing_flags',
    retrieval='training_only_exact_torch_squared_L2_topk_exclude_self_ID',
    refit='fresh_preprocessor_model_optimizer_fullouter_selected_epoch_count')
SOURCES = ['tabr_candidate.py', 'test_tabr_candidate.py', 'third_party/tabr/model.py', 'third_party/tabr/LICENSE',
           'third_party/tabr/upstream_tabr.py.txt', 'reports/tabr_protocol.json']
iterate.CONFIGS = {iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
                   NAME: dict(context=False, model='tabr_s', **SETTINGS)}
iterate.BLEND = []
RUN_OUT = None
DEVICE_OVERRIDE = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resource_snapshot(device):
    snapshot = dict(max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if device == 'mps':
        torch.mps.synchronize()
        snapshot.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                        mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return snapshot


def mean_training_loss(losses, n_rows):
    weights = [min(SETTINGS['batch_size'], n_rows-i*SETTINGS['batch_size'])
               for i in range(len(losses))]
    return float(np.average(losses, weights=weights))


class TabRPreprocessor:
    """Same compact fields and transformations as TabM, with dense one-hot output."""
    def fit(self, x):
        self.numeric_columns = list(NUMERIC)
        self.categorical_columns = list(CATEGORICAL)
        self.spending_columns = list(iterate.base.SPEND)
        raw = x[self.numeric_columns].astype(float)
        self.medians = raw.median().fillna(0.)
        numeric = self.impute_log(raw)
        self.keep = numeric.nunique().gt(1).to_numpy()
        if not self.keep.any():
            raise ValueError('At least one nonconstant numeric feature is required.')
        self.scaler = StandardScaler().fit(numeric.iloc[:, self.keep])
        self.vocabularies = []
        for col in self.categorical_columns:
            known = sorted(set(x[col].astype('string').fillna('__MISSING__'))-{'__MISSING__'})
            self.vocabularies.append({'__MISSING__': 1, **{v: i+2 for i, v in enumerate(known)}})
        self.cardinalities = [len(v)+1 for v in self.vocabularies]+[2]*len(self.numeric_columns)
        self.n_features = int(self.keep.sum())+sum(self.cardinalities)
        return self

    def impute_log(self, raw):
        values = raw.fillna(self.medians).copy()
        if values[self.spending_columns].lt(0).any().any():
            raise ValueError('Negative spending is invalid for log1p.')
        values[self.spending_columns] = np.log1p(values[self.spending_columns])
        return values

    def transform(self, x):
        raw = x[self.numeric_columns].astype(float)
        missing = raw.isna().to_numpy(dtype=np.int64)
        numeric = self.scaler.transform(self.impute_log(raw).iloc[:, self.keep]).astype(np.float32)
        cats = [x[col].astype('string').fillna('__MISSING__').map(vocab).fillna(0).to_numpy(dtype=np.int64)
                for col, vocab in zip(self.categorical_columns, self.vocabularies)]
        encoded = [np.eye(size, dtype=np.float32)[values]
                   for size, values in zip(self.cardinalities, [*cats, *missing.T])]
        result = np.concatenate([numeric, *encoded], axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite TabR features.')
        return result


def make_model(preprocessor, seed, device):
    seed_everything(seed)
    return TabRModel(preprocessor.n_features, **{k: SETTINGS[k] for k in
        ['d_main', 'context_size', 'context_dropout', 'dropout0', 'dropout1']}).to(device)


def optimizer_for(model):
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            excluded = (module_name == 'label_encoder' or name.endswith('bias')
                        or isinstance(module, torch.nn.LayerNorm))
            (no_decay if excluded else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': SETTINGS['weight_decay']},
                              {'params': no_decay, 'weight_decay': 0.}],
                             lr=SETTINGS['learning_rate'])


def train_epoch(model, optimizer, library, labels, ids, generator, max_batches=None):
    model.train()
    losses = []
    order = torch.randperm(len(labels), generator=generator, device='cpu')
    for batch, indices in enumerate(order.split(SETTINGS['batch_size'])):
        if max_batches is not None and batch >= max_batches:
            break
        idx = indices.to(library.device)
        optimizer.zero_grad(set_to_none=True)
        # Every query's own label is excluded by immutable row ID. Other training
        # rows, including other batch members, remain legitimate neighbors.
        logits = model(library[idx], library, labels, ids, query_ids=ids[idx])
        loss = F.binary_cross_entropy_with_logits(logits, labels[idx].float())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite TabR training loss.')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite TabR gradient.')
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.inference_mode()
def predict_arrays(model, query, library, labels, ids, query_ids=None, batch_size=256):
    model.eval()
    values = []
    for start in range(0, len(query), batch_size):
        q_ids = None if query_ids is None else query_ids[start:start+batch_size]
        logits = model(query[start:start+batch_size], library, labels, ids, query_ids=q_ids)
        values.append(logits.sigmoid().cpu().numpy())
    result = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite TabR predictions.')
    return result


def tensor(data, device, dtype=torch.float32):
    return torch.as_tensor(data, dtype=dtype, device=device)


def select_epochs(x, y, ids, seed, device):
    a, b = train_test_split(np.arange(len(y)), test_size=SETTINGS['inner_validation_fraction'],
                            stratify=y, random_state=seed)
    if np.intersect1d(ids[a], ids[b]).size:
        raise ValueError('Inner validation rows may not enter the retrieval library.')
    prep = TabRPreprocessor().fit(x.iloc[a])
    model = make_model(prep, seed, device)
    library, query = tensor(prep.transform(x.iloc[a]), device), tensor(prep.transform(x.iloc[b]), device)
    labels, lib_ids = tensor(y[a], device, torch.long), tensor(ids[a], device, torch.long)
    query_ids = tensor(ids[b], device, torch.long)
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    best_loss, best_epoch, stale, history = float('inf'), 1, 0, []
    epoch_log, steps = [], 0
    initial_resources = resource_snapshot(device)
    for epoch in range(1, SETTINGS['max_epochs']+1):
        started = time.monotonic()
        losses = train_epoch(model, optimizer, library, labels, lib_ids, generator)
        train_seconds = time.monotonic()-started
        steps += len(losses)
        score = float(log_loss(y[b], predict_arrays(model, query, library, labels, lib_ids,
                                                     query_ids), labels=[0, 1]))
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite inner validation metric.')
        history.append(score)
        epoch_log.append(dict(epoch=epoch, training_bce=mean_training_loss(losses, len(a)),
            validation_log_loss=score, optimizer_steps=len(losses), cumulative_optimizer_steps=steps,
            training_seconds=train_seconds, seconds=time.monotonic()-started))
        if score < best_loss:
            best_loss, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'TabR inner epoch={epoch} logloss={score:.6f} best_epoch={best_epoch}', flush=True)
        if stale >= SETTINGS['patience']:
            break
    metadata = dict(inner_training_rows=len(a), inner_validation_rows=len(b),
        inner_library_ids_sha256=hashlib.sha256(ids[a].tobytes()).hexdigest(),
        inner_validation_ids_sha256=hashlib.sha256(ids[b].tobytes()).hexdigest(),
        inner_best_log_loss=best_loss, inner_loss_history=history,
        inner_epoch_log=epoch_log, inner_optimizer_steps=steps,
        inner_resources_before=initial_resources, inner_resources_after=resource_snapshot(device),
        inner_stop_reason='early_stopping' if stale >= SETTINGS['patience'] else 'max_epochs',
        reached_epoch_cap=len(history) == SETTINGS['max_epochs'],
        best_epoch_at_cap=best_epoch == SETTINGS['max_epochs'])
    del model, optimizer, library, query, labels, lib_ids, query_ids
    release_memory()
    return best_epoch, metadata


class TabRPredictor:
    def __init__(self, preprocessor, model, library, labels, ids, device, metadata):
        self.preprocessor, self.model = preprocessor, model.cpu()
        self.library = library.detach().cpu().clone()
        self.labels, self.ids = labels.detach().cpu().clone(), ids.detach().cpu().clone()
        self.device, self.metadata = str(device), dict(metadata)
        self.batch_size = SETTINGS['batch_size']

    def predict_proba(self, x, query_ids=None):
        # Default input is out-of-sample. Callers evaluating training rows can
        # explicitly supply their original row IDs to preserve self-exclusion.
        device = self.device if self.device != 'mps' or torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)
        try:
            query = tensor(self.preprocessor.transform(x), device)
            q_ids = None if query_ids is None else tensor(query_ids, device, torch.long)
            p = predict_arrays(self.model, query, self.library.to(device), self.labels.to(device),
                               self.ids.to(device), q_ids, self.batch_size)
        finally:
            self.model.cpu()
        return np.column_stack([1-p, p])


def fit_tabr(x, y, ids, seed, device):
    selected, metadata = select_epochs(x, y, ids, seed, device)
    print(f'TabR selected_epochs={selected}; fresh full-fold refit on {len(y)} rows', flush=True)
    prep = TabRPreprocessor().fit(x)
    model = make_model(prep, seed, device)
    library = tensor(prep.transform(x), device)
    labels, lib_ids = tensor(y, device, torch.long), tensor(ids, device, torch.long)
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    epoch_log, steps = [], 0
    initial_resources = resource_snapshot(device)
    for epoch in range(1, selected+1):
        started = time.monotonic()
        losses = train_epoch(model, optimizer, library, labels, lib_ids, generator)
        steps += len(losses)
        epoch_log.append(dict(epoch=epoch, training_bce=mean_training_loss(losses, len(y)),
            optimizer_steps=len(losses), cumulative_optimizer_steps=steps,
            seconds=time.monotonic()-started))
        if epoch == 1 or epoch % 10 == 0:
            print(f'TabR refit epoch={epoch}/{selected}', flush=True)
    metadata.update(selected_epochs=selected, outer_training_rows=len(y),
        library_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        device=str(device), seed=int(seed), retained_numeric_columns=
        [c for c, keep in zip(prep.numeric_columns, prep.keep) if keep],
        one_hot_cardinalities=prep.cardinalities, input_dimension=prep.n_features)
    metadata.update(refit_epoch_log=epoch_log, refit_optimizer_steps=steps,
        refit_stop_reason='selected_epoch_count_completed',
        refit_resources_before=initial_resources, refit_resources_after=resource_snapshot(device),
        parameters=sum(p.numel() for p in model.parameters()))
    return TabRPredictor(prep, model, library, labels, lib_ids, device, metadata)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name != NAME:
        raise ValueError(name)
    ids, query_ids, target = np.asarray(train_idx, dtype=np.int64), np.asarray(predict_idx, dtype=np.int64), np.asarray(y, dtype=np.int64)
    if len(ids) != len(target) or len(np.unique(ids)) != len(ids) or np.intersect1d(ids, query_ids).size:
        raise ValueError('Training labels must align; train/query row IDs must be disjoint.')
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError('TabR requires binary true training labels.')
    device = DEVICE_OVERRIDE or ('mps' if torch.backends.mps.is_available() else 'cpu')
    started = time.monotonic()
    model = fit_tabr(matrices[False][0].iloc[ids], target, ids, seed, device)
    p = model.predict_proba(matrices[False][0].iloc[query_ids], query_ids=query_ids)[:, 1]
    model.metadata['seconds'] = time.monotonic()-started
    if RUN_OUT is not None:
        key = hashlib.sha256(ids.tobytes()).hexdigest()[:12]
        (RUN_OUT/f'tabr_training_seed{seed}_{key}.json').write_text(json.dumps(model.metadata, indent=2)+'\n')
    release_memory()
    return p, model


def environment_lock():
    return ''.join(f'{name}=={version}\n' for name, version in sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'tabr_policy' in protocol:
        return
    for filename in SOURCES:
        source = iterate.ROOT/filename
        destination = out/'source_snapshot'/filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out/'requirements-tabr.lock.txt').write_text(lock)
    protocol['tabr_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['tabr_policy'] = json.loads((iterate.ROOT/'reports/tabr_protocol.json').read_text())
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def check_environment(out):
    expected = json.loads((out/'protocol.json').read_text())['tabr_environment_sha256']
    if hashlib.sha256(environment_lock().encode()).hexdigest() != expected or hashlib.sha256(
            (out/'requirements-tabr.lock.txt').read_bytes()).hexdigest() != expected:
        raise RuntimeError('TabR current or saved dependency environment changed since screening.')


def probe():
    """Explicit opt-in real-data resource probe; never invoked by the unit tests."""
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    x = matrices[False][0].iloc[dev[a]]
    prep = TabRPreprocessor().fit(x)
    device = DEVICE_OVERRIDE or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = make_model(prep, iterate.SEEDS[0], device)
    library, labels = tensor(prep.transform(x), device), tensor(y[a], device, torch.long)
    ids = tensor(dev[a], device, torch.long)
    optimizer = optimizer_for(model)
    before = model.K.weight.detach().cpu().clone()
    started = time.monotonic()
    generator = torch.Generator(device='cpu').manual_seed(iterate.SEEDS[0])
    epoch_log = []
    for epoch in range(1, 4):
        epoch_started = time.monotonic()
        losses = train_epoch(model, optimizer, library, labels, ids, generator)
        snapshot = resource_snapshot(device)
        epoch_log.append(dict(epoch=epoch, training_bce=mean_training_loss(losses, len(a)),
            optimizer_steps=len(losses), seconds=time.monotonic()-epoch_started, resources=snapshot))
        print('TabR resource epoch', json.dumps(epoch_log[-1]), flush=True)
    train_seconds = time.monotonic()-started
    query = tensor(prep.transform(matrices[False][0].iloc[dev[b[:128]]]), device)
    p = predict_arrays(model, query, library, labels, ids, tensor(dev[b[:128]], device, torch.long))
    delta = float((model.K.weight.detach().cpu()-before).abs().max())
    if not delta > 0 or not np.isfinite(delta):
        raise RuntimeError('Resource probe must change trainable retrieval weights.')
    predictor = TabRPredictor(prep, model, library, labels, ids, device, {'resource_probe_only': True})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'tabr_probe.joblib'
        joblib.dump(predictor, path)
        restored = joblib.load(path)
        p_roundtrip = restored.predict_proba(matrices[False][0].iloc[dev[b[:128]]],
                                             query_ids=dev[b[:128]])[:, 1]
    np.testing.assert_allclose(p_roundtrip, p, rtol=0, atol=1e-6)
    result = dict(training_rows=len(a), query_rows=len(p), epochs=3, epoch_log=epoch_log,
        optimizer_steps=sum(row['optimizer_steps'] for row in epoch_log),
        training_seconds=train_seconds, total_seconds=time.monotonic()-started,
        retrieval_key_weight_max_change=delta, device=device,
        parameters=sum(p.numel() for p in model.parameters()),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        serialization_max_absolute_difference=float(np.max(np.abs(p_roundtrip-p))),
        valid_probabilities=bool(np.isfinite(p).all()), accuracy_computed=False,
        note='Resource probe only; weights are discarded and never reused by formal CV.')
    if device == 'mps':
        result['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT, DEVICE_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_036')
    parser.add_argument('--device', choices=['cpu', 'mps'])
    args = parser.parse_args()
    DEVICE_OVERRIDE = args.device
    torch.set_num_threads(4)
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT/'artifacts'/args.run
    if args.stage in ['confirm', 'export']:
        check_environment(RUN_OUT)
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    sys.modules['tabr_candidate'] = sys.modules[__name__]
    for cls in [TabRPreprocessor, TabRPredictor]:
        cls.__module__ = 'tabr_candidate'
    main()
