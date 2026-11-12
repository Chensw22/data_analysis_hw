"""Paper-based SCARF and a matched scratch MLP; local PyTorch implementation."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.nn import functional as F

import iterate
BASE_DATASET, BASE_EVALUATE = iterate.dataset, iterate.evaluate
import spatial_expert
from contrast_features import add_contrasts, ContrastPredictor

BASE_FIT = spatial_expert.ORIGINAL_FIT
SCRATCH = 'v69_matched_scratch_MLP'
SCARF = 'v70_SCARF'
REFERENCE = 'v44_seed_ensemble'
NUMERIC = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNumber']
CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
SPEND = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
SETTINGS = dict(width=256, encoder_layers=4, batch_size=128, learning_rate=.001,
                pretrain_max_epochs=1000, pretrain_patience=10, pretrain_min_epochs=10,
                supervised_max_epochs=500, supervised_patience=40, supervised_min_epochs=30,
                min_delta=.0001, inner_fraction=.2, unlabeled_validation_fraction=.1,
                corruption_rate=.6, temperature=1., validation_corruption_passes=10)
SOURCES = ['scarf_candidate.py', 'test_scarf_candidate.py', 'reports/scarf_protocol.json',
           'contrast_features.py', 'spatial_expert.py']
TEST_ROWS = None
RUN_OUT = None
DEVICE = 'mps'
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    REFERENCE: dict(context=False, model='seed_ensemble', seed_offsets=[0, 100, 200],
                    knots=5, C=.1, weight=.5),
    SCRATCH: dict(context=False, model='matched_scratch_mlp', settings=SETTINGS),
    SCARF: dict(context=False, model='scarf', settings=SETTINGS,
                unlabeled_pool='applicable_training_attributes_plus_actual_test'),
}
iterate.BLEND = []


def release(device):
    gc.collect()
    if str(device).startswith('mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def resource_snapshot(device):
    result = dict(process_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if str(device).startswith('mps'):
        torch.mps.synchronize()
        result['mps_driver_snapshot_bytes'] = int(torch.mps.driver_allocated_memory())
    return result


def categories(series):
    return series.fillna('__MISSING__').astype(str)


class Preprocessor:
    """Fit only the supervised training partition; corruption groups are semantic columns."""
    def fit(self, raw):
        x = raw[NUMERIC].astype(float).copy()
        if (x[SPEND] < 0).any().any():
            raise ValueError('Spending must be nonnegative.')
        x[SPEND] = np.log1p(x[SPEND])
        self.median = x.median().fillna(0.).to_numpy()
        values = np.where(x.isna(), self.median, x.to_numpy())
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale == 0] = 1.
        self.vocabularies = {col: sorted(set(categories(raw[col])) - {'__MISSING__'})
                             for col in CATEGORICAL}
        self.groups = [[i, len(NUMERIC)+i] for i in range(len(NUMERIC))]
        offset = 2*len(NUMERIC)
        for col in CATEGORICAL:
            width = len(self.vocabularies[col])+2  # unknown and missing are distinct.
            self.groups.append(list(range(offset, offset+width)))
            offset += width
        self.output_dim = offset
        return self

    def transform(self, raw):
        numeric = raw[NUMERIC].astype(float).copy()
        missing = numeric.isna().to_numpy()
        if (numeric[SPEND] < 0).any().any():
            raise ValueError('Spending must be nonnegative.')
        numeric[SPEND] = np.log1p(numeric[SPEND])
        values = np.where(missing, self.median, numeric.to_numpy())
        parts = [(values-self.mean)/self.scale, missing.astype(float)]
        for col in CATEGORICAL:
            mapping = {value: index+2 for index, value in enumerate(self.vocabularies[col])}
            mapping['__MISSING__'] = 1
            codes = categories(raw[col]).map(mapping).fillna(0).to_numpy(dtype=int)
            parts.append(np.eye(len(mapping)+1, dtype=np.float32)[codes])
        x = np.concatenate(parts, axis=1).astype(np.float32)
        if x.shape[1] != self.output_dim or not np.isfinite(x).all():
            raise RuntimeError('Invalid encoded input.')
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, width, layers):
        super().__init__()
        sequence = []
        for layer in range(layers):
            sequence.extend([nn.Linear(input_dim if layer == 0 else width, width), nn.ReLU()])
        self.layers = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layers(x)


class ContrastiveNetwork(nn.Module):
    def __init__(self, input_dim, settings):
        super().__init__()
        width = settings['width']
        self.encoder = Encoder(input_dim, width, settings['encoder_layers'])
        self.projection = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, x):
        return F.normalize(self.projection(self.encoder(x)), dim=1)


class Classifier(nn.Module):
    def __init__(self, input_dim, settings):
        super().__init__()
        width = settings['width']
        self.encoder = Encoder(input_dim, width, settings['encoder_layers'])
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def make_contrastive(input_dim, seed, device):
    torch.manual_seed(seed)
    return ContrastiveNetwork(input_dim, SETTINGS).to(device)


def make_classifier(input_dim, seed, device, encoder_state=None):
    torch.manual_seed(seed)
    model = Classifier(input_dim, SETTINGS)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    return model.to(device)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def corrupt(x, donors, groups, rng, rate):
    """Copy whole feature groups; neither this function nor pretraining accepts y."""
    count = int(np.floor(rate*len(groups)))
    selected = np.argsort(rng.random((len(x), len(groups))), axis=1)[:, :count]
    result = x.copy()
    for group_index, columns in enumerate(groups):
        rows = np.flatnonzero(np.any(selected == group_index, axis=1))
        donor_rows = rng.integers(len(donors), size=len(rows))
        result[np.ix_(rows, columns)] = donors[np.ix_(donor_rows, columns)]
    return result


def batches(n, rng=None):
    order = np.arange(n) if rng is None else rng.permutation(n)
    size = SETTINGS['batch_size']
    return [order[start:start+size] for start in range(0, n, size)]


def finite_step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss.')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        raise RuntimeError('Missing or nonfinite parameter gradients.')
    optimizer.step()


def progress(kind, history, cadence):
    row = history[-1]
    if row['epoch'] == 1 or row['epoch'] % cadence == 0:
        print('SCARF phase', json.dumps(dict(phase=kind, **row)), flush=True)


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'],
                            betas=(.9, .999), eps=1e-8, weight_decay=0.)


def contrastive_loss(model, clean, changed, device):
    a = model(torch.as_tensor(clean, dtype=torch.float32, device=device))
    b = model(torch.as_tensor(changed, dtype=torch.float32, device=device))
    logits = a @ b.T / SETTINGS['temperature']
    return F.cross_entropy(logits, torch.arange(len(clean), device=device))


def static_validation(clean, donors, groups, seed):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(SETTINGS['validation_corruption_passes']):
        for indices in batches(len(clean), rng):
            if len(indices) >= 2:
                batch = clean[indices]
                result.append((batch, corrupt(batch, donors, groups, rng, SETTINGS['corruption_rate'])))
    if not result:
        raise ValueError('Contrastive validation requires at least two rows.')
    return result


def contrastive_validation(model, pairs, device):
    model.eval()
    values = []
    with torch.no_grad():
        for a, b in pairs:
            value = float(contrastive_loss(model, a, b, device).cpu())
            values.append((len(a), value))
    loss = sum(n*v for n, v in values)/sum(n for n, _ in values)
    if not np.isfinite(loss):
        raise RuntimeError('Nonfinite contrastive validation loss.')
    return loss


def pretrain(pool, groups, seed, device, fixed_epochs=None, selection_cap_override=None):
    """Select epochs on a fixed unlabeled split, or freshly train the complete pool."""
    started = time.monotonic()
    if fixed_epochs is None:
        a, b = train_test_split(np.arange(len(pool)), test_size=SETTINGS['unlabeled_validation_fraction'],
                                random_state=seed+31)
        data = pool[a]
        validation = static_validation(pool[b], data, groups, seed+32)
        cap = SETTINGS['pretrain_max_epochs'] if selection_cap_override is None else int(selection_cap_override)
    else:
        data, validation, cap = pool, None, int(fixed_epochs)
    model = make_contrastive(pool.shape[1], seed, device)
    initial = next(model.encoder.parameters()).detach().cpu().clone()
    opt = optimizer(model)
    rng = np.random.default_rng(seed+33)
    history, steps, stale = [], 0, 0
    best_loss, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(1, cap+1):
        model.train()
        loss_sum = seen = 0
        for indices in batches(len(data), rng):
            if len(indices) < 2:
                continue
            clean = data[indices]
            changed = corrupt(clean, data, groups, rng, SETTINGS['corruption_rate'])
            loss = contrastive_loss(model, clean, changed, device)
            finite_step(loss, model, opt)
            loss_sum += len(indices)*float(loss.detach().cpu())
            seen += len(indices)
            steps += 1
        if not seen:
            raise RuntimeError('No contrastive training batches.')
        val = contrastive_validation(model, validation, device) if validation is not None else None
        history.append(dict(epoch=epoch, train_loss=loss_sum/seen, validation_loss=val,
                            optimizer_steps=steps, seconds=time.monotonic()-started))
        progress('pretrain_selection' if validation is not None else 'pretrain_refit', history, 25)
        if validation is not None:
            if val < best_loss-SETTINGS['min_delta']:
                best_loss, best_epoch, best_state, stale = val, epoch, cpu_state(model.encoder), 0
            else:
                stale += 1
            if epoch >= SETTINGS['pretrain_min_epochs'] and stale >= SETTINGS['pretrain_patience']:
                break
    delta = float((next(model.encoder.parameters()).detach().cpu()-initial).abs().max())
    if not np.isfinite(delta) or delta == 0:
        raise RuntimeError('Contrastive training did not update encoder parameters.')
    if validation is None:
        best_epoch, best_state = cap, cpu_state(model.encoder)
    metadata = dict(kind='pretrain_refit' if validation is None else 'pretrain_epoch_selection',
                    pool_rows=len(pool), training_rows=len(data), validation_rows=0 if validation is None else len(b),
                    selected_epoch=best_epoch, last_epoch=len(history), optimizer_steps=steps,
                    cap_reached=bool(validation is not None and len(history) == cap),
                    best_epoch_at_cap=bool(validation is not None and best_epoch == cap),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    first_encoder_parameter_max_change=delta, history=history,
                    elapsed_seconds=time.monotonic()-started, **resource_snapshot(device))
    del model, opt
    release(device)
    return best_state, best_epoch, metadata


def supervised_loss(model, x, y, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for indices in batches(len(x)):
            logits = model(torch.as_tensor(x[indices], device=device))
            loss = F.binary_cross_entropy_with_logits(logits, torch.as_tensor(y[indices], dtype=torch.float32, device=device))
            losses.append((len(indices), float(loss.cpu())))
    result = sum(n*v for n, v in losses)/sum(n for n, _ in losses)
    if not np.isfinite(result):
        raise RuntimeError('Nonfinite supervised validation loss.')
    return result


def supervise(x, y, seed, device, encoder_state=None, validation=None, fixed_epochs=None):
    started = time.monotonic()
    model = make_classifier(x.shape[1], seed, device, encoder_state)
    opt = optimizer(model)
    rng = np.random.default_rng(seed+41)
    cap = SETTINGS['supervised_max_epochs'] if fixed_epochs is None else int(fixed_epochs)
    best_loss, best_epoch, stale, steps = float('inf'), 0, 0, 0
    history = []
    for epoch in range(1, cap+1):
        model.train()
        loss_sum = 0.
        for indices in batches(len(x), rng):
            logits = model(torch.as_tensor(x[indices], device=device))
            loss = F.binary_cross_entropy_with_logits(logits,
                torch.as_tensor(y[indices], dtype=torch.float32, device=device))
            finite_step(loss, model, opt)
            loss_sum += len(indices)*float(loss.detach().cpu())
            steps += 1
        val = supervised_loss(model, *validation, device) if validation is not None else None
        history.append(dict(epoch=epoch, train_loss=loss_sum/len(x), validation_loss=val,
                            optimizer_steps=steps, seconds=time.monotonic()-started))
        progress('supervised_selection' if validation is not None else 'supervised_refit', history, 10)
        if validation is not None:
            if val < best_loss-SETTINGS['min_delta']:
                best_loss, best_epoch, stale = val, epoch, 0
            else:
                stale += 1
            if epoch >= SETTINGS['supervised_min_epochs'] and stale >= SETTINGS['supervised_patience']:
                break
    if validation is None:
        best_epoch = cap
    metadata = dict(kind='supervised_epoch_selection' if validation is not None else 'supervised_refit',
                    training_rows=len(x), validation_rows=0 if validation is None else len(validation[0]),
                    selected_epoch=best_epoch, last_epoch=len(history), optimizer_steps=steps,
                    cap_reached=bool(validation is not None and len(history) == cap),
                    best_epoch_at_cap=bool(validation is not None and best_epoch == cap),
                    history=history, elapsed_seconds=time.monotonic()-started,
                    parameter_count=sum(p.numel() for p in model.parameters()), **resource_snapshot(device))
    state = cpu_state(model) if validation is None else None
    del model, opt
    release(device)
    return state, best_epoch, metadata


class SCARFPredictor:
    def __init__(self, prep, state, metadata, settings, device):
        self.preprocessor, self.state = prep, state
        self.metadata, self.settings = metadata, copy.deepcopy(settings)
        self.device = str(device)

    def predict_proba(self, raw):
        # CPU weights remain portable; this inference fallback never changes training.
        device = 'cpu' if self.device == 'mps' and not torch.backends.mps.is_available() else self.device
        self.last_prediction_device = device
        x = self.preprocessor.transform(raw)
        model = Classifier(self.preprocessor.output_dim, self.settings)
        model.load_state_dict(self.state)
        model = model.to(device).eval()
        values = []
        try:
            with torch.no_grad():
                for start in range(0, len(x), self.settings['batch_size']):
                    values.append(model(torch.as_tensor(x[start:start+self.settings['batch_size']], device=device)).sigmoid().cpu().numpy())
            p = np.concatenate(values).astype(float)
            if not np.isfinite(p).all():
                raise RuntimeError('Invalid prediction probabilities.')
            return np.column_stack([1-p, p])
        finally:
            del model
            release(device)


class MeanPredictor:
    def __init__(self, members):
        self.members = members

    def predict_proba(self, raw):
        return np.mean([member.predict_proba(raw) for member in self.members], axis=0)


def dataset():
    global TEST_ROWS
    train, test, matrices, dev = BASE_DATASET()
    TEST_ROWS = np.arange(len(train), len(train)+len(test))
    return train, test, matrices, dev


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    raw, cats = matrices[False]
    if name == REFERENCE:
        changed = {False: (add_contrasts(raw), cats)}
        results = [spatial_expert.fit_spatial(REFERENCE, changed, train_idx, y, predict_idx, seed+offset)
                   for offset in [0, 100, 200]]
        return np.mean([p for p, _ in results], axis=0), MeanPredictor([ContrastPredictor(m) for _, m in results])
    if name not in [SCRATCH, SCARF]:
        raise ValueError(name)
    train_idx, predict_idx = np.asarray(train_idx), np.asarray(predict_idx)
    target = np.asarray(y, dtype=np.int64)
    if TEST_ROWS is None or len(target) != len(train_idx) or len(np.unique(train_idx)) != len(train_idx):
        raise ValueError('Initialize actual test rows and aligned unique training indices.')
    if np.intersect1d(train_idx, predict_idx).size or np.intersect1d(train_idx, TEST_ROWS).size:
        raise ValueError('Training labels must exclude prediction and actual test rows.')
    started = time.monotonic()
    a, b = train_test_split(np.arange(len(target)), test_size=SETTINGS['inner_fraction'],
                            stratify=target, random_state=seed)
    inner = Preprocessor().fit(raw.iloc[train_idx[a]])
    x, xv = inner.transform(raw.iloc[train_idx[a]]), inner.transform(raw.iloc[train_idx[b]])
    phases = []
    pretrain_epochs, encoder_state = None, None
    if name == SCARF:
        pool = np.concatenate([x, inner.transform(raw.iloc[TEST_ROWS])])
        _, pretrain_epochs, phase = pretrain(pool, inner.groups, seed, DEVICE)
        phases.append(phase)
        # Epoch selection and pretraining refit are distinct; no inner checkpoint shortcuts.
        encoder_state, _, phase = pretrain(pool, inner.groups, seed, DEVICE, pretrain_epochs)
        phases.append(phase)
    _, supervised_epochs, phase = supervise(x, target[a], seed, DEVICE, encoder_state,
                                             validation=(xv, target[b]))
    phases.append(phase)
    # Refit both preprocessing and networks from scratch on all outer-training rows.
    outer = Preprocessor().fit(raw.iloc[train_idx])
    all_x = outer.transform(raw.iloc[train_idx])
    encoder_state = None
    if name == SCARF:
        pool = np.concatenate([all_x, outer.transform(raw.iloc[TEST_ROWS])])
        encoder_state, _, phase = pretrain(pool, outer.groups, seed, DEVICE, pretrain_epochs)
        phases.append(phase)
    state, _, phase = supervise(all_x, target, seed, DEVICE, encoder_state,
                                fixed_epochs=supervised_epochs)
    phases.append(phase)
    metadata = dict(version=name, seed=int(seed), outer_training_rows=len(train_idx),
        inner_training_rows=len(a), inner_validation_rows=len(b), actual_test_attribute_rows=len(TEST_ROWS),
        inner_train_index_sha256=hashlib.sha256(train_idx[a].astype(np.int64).tobytes()).hexdigest(),
        inner_validation_index_sha256=hashlib.sha256(train_idx[b].astype(np.int64).tobytes()).hexdigest(),
        pretraining_epochs=pretrain_epochs, supervised_epochs=supervised_epochs, phases=phases,
        elapsed_training_seconds=time.monotonic()-started,
        total_optimizer_steps=sum(phase['optimizer_steps'] for phase in phases),
        epoch_transfer_note='Selected epoch counts transfer to larger refit pools; optimizer step counts change and are reported per phase.',
        training_adequacy='inspect curves and cap flags; one recipe does not establish family failure',
        resource_note='RSS is a process-lifetime maximum; MPS values are snapshots, not continuous peaks')
    predictor = SCARFPredictor(outer, state, metadata, SETTINGS, DEVICE)
    prediction_started = time.monotonic()
    p = predictor.predict_proba(raw.iloc[predict_idx])[:, 1]
    metadata['prediction_seconds'] = time.monotonic()-prediction_started
    if RUN_OUT is not None:
        key = hashlib.sha256(train_idx.astype(np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT/f'{name}_seed{seed}_{key}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('SCARF recipe', json.dumps({k: v for k, v in metadata.items() if k != 'phases'}), flush=True)
    return p, predictor


def environment_lock():
    return ''.join(f'{n}=={v}\n' for n, v in sorted(
        (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()))


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'scarf_policy' in protocol:
        return
    for filename in SOURCES:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out/'requirements-scarf.lock.txt').write_text(lock)
    protocol['scarf_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['scarf_training_device'] = DEVICE
    protocol['scarf_policy'] = json.loads((iterate.ROOT/'reports/scarf_protocol.json').read_text())
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def probe():
    """Root-only resource/API probe: one real epoch of each phase, no accuracy."""
    train, _, matrices, dev = dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    inner_a, _ = train_test_split(np.arange(len(a)), test_size=SETTINGS['inner_fraction'],
                                 stratify=y[a], random_state=iterate.SEEDS[0])
    raw = matrices[False][0]
    prep = Preprocessor().fit(raw.iloc[dev[a[inner_a]]])
    x = prep.transform(raw.iloc[dev[a[inner_a]]])
    pool = np.concatenate([x, prep.transform(raw.iloc[TEST_ROWS])])
    started = time.monotonic()
    encoder, _, pre = pretrain(pool, prep.groups, iterate.SEEDS[0], DEVICE,
                               selection_cap_override=1)
    state, _, sup = supervise(x, y[a[inner_a]], iterate.SEEDS[0], DEVICE,
                              encoder_state=encoder, fixed_epochs=1)
    predictor = SCARFPredictor(prep, state, {}, SETTINGS, DEVICE)
    query = raw.iloc[dev[b[:128]]]
    p = predictor.predict_proba(query)
    with tempfile.TemporaryDirectory(prefix='scarf-probe-') as folder:
        path = Path(folder)/'predictor.joblib'
        joblib.dump(predictor, path)
        restored = joblib.load(path)
        restored_p = restored.predict_proba(query)
        np.testing.assert_allclose(p, restored_p, atol=1e-6, rtol=0)
    result = dict(probe_only=True, accuracy_computed=False, actual_test_attribute_rows=len(TEST_ROWS),
        inner_training_rows=len(x), unlabeled_pool_rows=len(pool), query_rows=len(query),
        pretraining_phase=pre, supervised_phase=sup, total_seconds=time.monotonic()-started,
        serialization_max_absolute_difference=float(np.max(np.abs(p-restored_p))),
        note='Resource probe only. One-epoch overrides do not modify the frozen screening settings.',
        **resource_snapshot(DEVICE))
    print('SCARF probe', json.dumps(result), flush=True)
    return result


iterate.dataset, iterate.fit, iterate.evaluate = dataset, fit, evaluate


def main():
    global RUN_OUT, DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_037')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    DEVICE = args.device
    torch.set_num_threads(4)
    if DEVICE == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; do not silently change the training device.')
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT/'artifacts'/args.run
    if args.stage in ['confirm', 'export']:
        protocol = json.loads((RUN_OUT/'protocol.json').read_text())
        expected = protocol['scarf_environment_sha256']
        if protocol['scarf_training_device'] != DEVICE:
            raise RuntimeError('Training device changed since screening.')
        if (hashlib.sha256(environment_lock().encode()).hexdigest() != expected or
            hashlib.sha256((RUN_OUT/'requirements-scarf.lock.txt').read_bytes()).hexdigest() != expected):
            raise RuntimeError('SCARF environment changed since screening.')
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    sys.modules['scarf_candidate'] = sys.modules[__name__]
    for cls in [Preprocessor, SCARFPredictor, MeanPredictor]:
        cls.__module__ = 'scarf_candidate'
    main()
