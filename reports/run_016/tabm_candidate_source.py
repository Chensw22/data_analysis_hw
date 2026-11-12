"""Train one frozen TabM candidate with inner epoch selection and full-fold refit."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rtdl_num_embeddings
from tabm import TabM
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import iterate

BASE_FIT = iterate.fit
BASE_EVALUATE = iterate.evaluate
NAME = 'v41_TabM'
NUMERIC = ['Age', 'CabinNumber', *iterate.base.SPEND]
CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
SETTINGS = dict(n_bins=16, d_embedding=16, n_blocks=2, d_block=128, k=32,
                dropout=.1, learning_rate=.002, weight_decay=.0003,
                batch_size=256, gradient_clip=1., max_epochs=200, patience=20,
                inner_validation_fraction=.2, epoch_metric='log_loss',
                refit_policy='fresh_model_same_selected_epoch_count',
                device_policy='prefer_mps_cpu_fallback_for_backend_error')
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    NAME: dict(context=False, model='tabm', **SETTINGS),
}
iterate.BLEND = []
RUN_OUT = None


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


class TabMPreprocessor:
    """Training-only medians, category vocabulary, scaling and constant filtering."""

    def fit(self, x):
        raw = x[NUMERIC].astype(float)
        self.medians = raw.median().fillna(0.)
        numeric = self._impute_log(raw)
        self.keep = numeric.nunique().gt(1).to_numpy()
        if not self.keep.any():
            raise ValueError('TabM requires at least one nonconstant numerical feature.')
        self.numeric_columns = [c for c, keep in zip(NUMERIC, self.keep) if keep]
        self.scaler = StandardScaler().fit(numeric.iloc[:, self.keep])
        self.vocabularies = []
        for col in CATEGORICAL:
            # 0 is reserved for unknown, 1 for missing, even when absent in fit.
            values = x[col].astype('string').fillna('__MISSING__')
            known = sorted(set(values) - {'__MISSING__'})
            self.vocabularies.append({'__MISSING__': 1, **{v: j + 2 for j, v in enumerate(known)}})
        self.cat_cardinalities = [len(v) + 1 for v in self.vocabularies] + [2] * len(NUMERIC)
        return self

    def _impute_log(self, raw):
        values = raw.fillna(self.medians).copy()
        if values[iterate.base.SPEND].lt(0).any().any():
            raise ValueError('Spending must be nonnegative before log1p.')
        values[iterate.base.SPEND] = np.log1p(values[iterate.base.SPEND])
        return values

    def transform(self, x):
        raw = x[NUMERIC].astype(float)
        missing = raw.isna().to_numpy(dtype=np.int64)
        values = self._impute_log(raw)
        numeric = self.scaler.transform(values.iloc[:, self.keep]).astype(np.float32)
        columns = []
        for col, vocabulary in zip(CATEGORICAL, self.vocabularies):
            values = x[col].astype('string').fillna('__MISSING__')
            columns.append(values.map(vocabulary).fillna(0).to_numpy(dtype=np.int64))
        categorical = np.column_stack([*columns, *missing.T])
        if not np.isfinite(numeric).all():
            raise ValueError('Nonfinite transformed numerical values.')
        return numeric, categorical


def make_model(preprocessor, numeric_train, seed, device):
    seed_everything(seed)
    # Quantile computation stays on CPU. Duplicate quantiles are removed by the
    # library; constant columns were removed using training data only.
    bins = rtdl_num_embeddings.compute_bins(torch.from_numpy(numeric_train), n_bins=SETTINGS['n_bins'])
    embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bins, d_embedding=SETTINGS['d_embedding'], activation=False, version='B')
    model = TabM.make(
        n_num_features=numeric_train.shape[1], cat_cardinalities=preprocessor.cat_cardinalities,
        d_out=1, num_embeddings=embeddings, arch_type='tabm',
        n_blocks=SETTINGS['n_blocks'], d_block=SETTINGS['d_block'],
        dropout=SETTINGS['dropout'], k=SETTINGS['k'])
    return model.to(device)


def member_loss(logits, targets):
    """Each member learns independently; never average logits before the loss."""
    logits = logits.squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, targets[:, None].expand_as(logits))


def member_probabilities(logits):
    return logits.squeeze(-1).sigmoid().mean(dim=1)


def tensors(arrays, device):
    numeric, categorical = arrays
    return (torch.as_tensor(numeric, dtype=torch.float32, device=device),
            torch.as_tensor(categorical, dtype=torch.long, device=device))


def train_epoch(model, optimizer, data, targets, generator, max_batches=None):
    model.train()
    # CPU generator works on both CPU and MPS, with the same index permutation.
    order = torch.randperm(len(targets), generator=generator, device='cpu')
    losses = []
    for batch, idx_cpu in enumerate(order.split(SETTINGS['batch_size'])):
        if max_batches is not None and batch >= max_batches:
            break
        idx = idx_cpu.to(targets.device)
        optimizer.zero_grad(set_to_none=True)
        loss = member_loss(model(data[0][idx], data[1][idx]), targets[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('TabM produced a nonfinite training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.inference_mode()
def predict_arrays(model, data):
    model.eval()
    predictions = []
    for start in range(0, len(data[0]), 512):
        logits = model(data[0][start:start + 512], data[1][start:start + 512])
        predictions.append(member_probabilities(logits).cpu().numpy())
    if not predictions:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(predictions)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'],
                             weight_decay=SETTINGS['weight_decay'])


def select_epochs(x, y, seed, device):
    a, b = train_test_split(np.arange(len(y)), test_size=SETTINGS['inner_validation_fraction'],
                            stratify=y, random_state=seed)
    prep = TabMPreprocessor().fit(x.iloc[a])
    train_arrays = prep.transform(x.iloc[a])
    model = make_model(prep, train_arrays[0], seed, device)
    train_data, valid_data = tensors(train_arrays, device), tensors(prep.transform(x.iloc[b]), device)
    target = torch.as_tensor(y[a], dtype=torch.float32, device=device)
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    best_loss, best_epoch, stale = float('inf'), 1, 0
    history = []
    for epoch in range(1, SETTINGS['max_epochs'] + 1):
        train_epoch(model, optimizer, train_data, target, generator)
        score = float(log_loss(y[b], predict_arrays(model, valid_data), labels=[0, 1]))
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite inner validation loss.')
        history.append(score)
        if score < best_loss:
            best_loss, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'TabM inner epoch={epoch} logloss={score:.6f} best_epoch={best_epoch}', flush=True)
        if stale >= SETTINGS['patience']:
            break
    del model, optimizer, train_data, valid_data, target
    release_memory()
    return best_epoch, dict(inner_train_rows=len(a), inner_validation_rows=len(b),
                           inner_best_log_loss=best_loss, inner_loss_history=history)


class TabMPredictor:
    """CPU-portable fitted model; retain the chosen runtime device as metadata."""

    def __init__(self, preprocessor, model, device, selected_epochs, metadata):
        self.preprocessor = preprocessor
        self.model = model.cpu()
        self.device = str(device)
        self.selected_epochs = selected_epochs
        self.metadata = metadata

    def predict_proba(self, x):
        device = self.device if self.device != 'mps' or torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)
        try:
            p = predict_arrays(self.model, tensors(self.preprocessor.transform(x), device))
        finally:
            self.model.cpu()
        return np.column_stack([1 - p, p])


def fit_tabm(x, y, seed, device):
    selected_epochs, metadata = select_epochs(x, y, seed, device)
    print(f'TabM selected_epochs={selected_epochs}; fresh full-fold refit on {len(y)} rows', flush=True)
    # This is a fresh fit: no inner-trained network, optimizer, bins, medians,
    # scaler, or category vocabulary is carried into the full training fold.
    prep = TabMPreprocessor().fit(x)
    arrays = prep.transform(x)
    model = make_model(prep, arrays[0], seed, device)
    data = tensors(arrays, device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = optimizer_for(model)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(1, selected_epochs + 1):
        train_epoch(model, optimizer, data, targets, generator)
        if epoch == 1 or epoch % 10 == 0:
            print(f'TabM refit epoch={epoch}/{selected_epochs}', flush=True)
    metadata.update(seed=int(seed), device=str(device), selected_epochs=selected_epochs,
                    outer_train_rows=len(y), retained_numeric_columns=prep.numeric_columns)
    return TabMPredictor(prep, model, device, selected_epochs, metadata)


def backend_error(error):
    return isinstance(error, NotImplementedError) or (
        isinstance(error, RuntimeError) and any(s in str(error).lower()
        for s in ['mps', 'metal', 'not implemented', 'out of memory']))


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name != NAME:
        raise ValueError(f'Unknown TabM candidate: {name}')
    x, _ = matrices[False]
    target = np.asarray(y, dtype=np.int64)
    if len(target) != len(train_idx):
        raise ValueError('Target length must match the outer training indices.')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    started = time.monotonic()
    predictor = None
    fallback_reason = None
    try:
        predictor = fit_tabm(x.iloc[train_idx], target, seed, device)
        pred = predictor.predict_proba(x.iloc[predict_idx])[:, 1]
    except (RuntimeError, NotImplementedError) as error:
        if device != 'mps' or not backend_error(error):
            raise
        fallback_reason = str(error)
    if fallback_reason is not None:
        # Exit the except block first: its traceback otherwise retains the
        # failed training frames and their GPU tensors during the CPU retry.
        predictor = None
        release_memory()
        print(f'TabM MPS backend failed; restarting the frozen candidate on CPU: {fallback_reason}', flush=True)
        predictor = fit_tabm(x.iloc[train_idx], target, seed, 'cpu')
        predictor.metadata['mps_fallback_reason'] = fallback_reason
        pred = predictor.predict_proba(x.iloc[predict_idx])[:, 1]
    predictor.metadata['seconds'] = time.monotonic() - started
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT / f'tabm_training_seed{seed}_{key}.json').write_text(
            json.dumps(predictor.metadata, indent=2) + '\n')
    release_memory()
    return pred, predictor


def environment_lock():
    return ''.join(f'{name}=={version}\n' for name, version in sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'tabm_policy' in protocol:
        return
    source = Path(__file__).resolve()
    shutil.copy2(source, out / 'tabm_candidate_source.py')
    protocol['sources'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out / 'requirements-tabm.lock.txt').write_text(lock)
    protocol['tabm_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['tabm_policy'] = (
        'Compact13 plus seven original numeric missing flags; all medians/scaling/category vocabularies/'
        'constant filtering/quantile bins fit inside the applicable training split. Inner stratified20% '
        'selects epoch by logloss; a fresh full-outer-training preprocessing/model refit uses exactly '
        'that many epochs. Separate BCE loss per member; mean sigmoid probabilities at inference. '
        'No outer-validation labels for epoch selection, no pretrained weights or cloud inference.')
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def evaluate(name, matrices, train, dev, seed, out):
    # iterate.screen has created protocol.json by this point. Freeze the wrapper
    # and dependency snapshot before the first model executes, not afterwards.
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def check_environment(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    expected = protocol['tabm_environment_sha256']
    if hashlib.sha256(environment_lock().encode()).hexdigest() != expected:
        raise RuntimeError('TabM dependency environment changed since screening.')
    if hashlib.sha256((out / 'requirements-tabm.lock.txt').read_bytes()).hexdigest() != expected:
        raise RuntimeError('TabM saved dependency snapshot changed since screening.')


def probe():
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, _ = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    x = matrices[False][0].iloc[dev[a]]
    prep = TabMPreprocessor().fit(x)
    arrays = prep.transform(x)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    started = time.monotonic()
    fallback = None
    for attempt in range(2):
        model = data = target = optimizer = None
        if attempt:
            release_memory()
        try:
            model = make_model(prep, arrays[0], iterate.SEEDS[0], device)
            data = tensors(arrays, device)
            target = torch.as_tensor(y[a], dtype=torch.float32, device=device)
            optimizer = optimizer_for(model)
            losses = train_epoch(model, optimizer, data, target,
                                 torch.Generator().manual_seed(iterate.SEEDS[0]), max_batches=2)
            p = predict_arrays(model, (data[0][:32], data[1][:32]))
            assert p.shape == (32,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            break
        except (RuntimeError, NotImplementedError) as error:
            if attempt or device != 'mps' or not backend_error(error):
                raise
            fallback, device = str(error), 'cpu'
    result = dict(preprocessor_training_rows=len(a), training_batches=2,
                  training_batch_losses=losses, device=device, fallback_reason=fallback,
                  seconds=time.monotonic() - started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  parameters=sum(p.numel() for p in model.parameters()),
                  retained_numeric_columns=prep.numeric_columns,
                  valid_probabilities=True, accuracy_computed=False,
                  note='Only a resource/backpropagation probe, not CV or a completed training run.')
    if device == 'mps':
        result['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        result['mps_recommended_max_bytes'] = torch.mps.recommended_max_memory()
    print(json.dumps(result), flush=True)
    return result


iterate.fit = fit
iterate.evaluate = evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_016')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT / 'artifacts' / args.run
    if args.stage in ['confirm', 'export']:
        check_environment(RUN_OUT)
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    # Models exported by the CLI must be loadable outside this __main__ process.
    sys.modules['tabm_candidate'] = sys.modules[__name__]
    for cls in [TabMPreprocessor, TabMPredictor]:
        cls.__module__ = 'tabm_candidate'
    main()
