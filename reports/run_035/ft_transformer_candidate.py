"""Official FT-Transformer baseline, trained from scratch with nested epoch selection."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import iterate

BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
NAME = 'v67_FTTransformer_default'
NUMERIC = ['Age', 'CabinNumber', *iterate.base.SPEND]
CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
SETTINGS = dict(package='rtdl_revisiting_models', package_version='0.0.2',
    n_blocks=3, d_block=192, attention_n_heads=8, attention_dropout=.2,
    ffn_d_hidden_multiplier=4/3, ffn_dropout=.1, residual_dropout=0.,
    learning_rate=1e-4, weight_decay=1e-5, batch_size=256,
    max_epochs=500, patience=40, inner_validation_fraction=.2,
    epoch_metric='log_loss', refit_policy='fresh_model_same_selected_epoch_count',
    missing_flags='seven_binary_categorical_tokens',
    optimizer='official_make_default_optimizer', device_policy='mps_if_available_else_cpu',
    stage_budget_seconds=dict(probe=300, screen=3600, confirm=10800, export=3600))
PROTOCOL_FILE = 'reports/ft_transformer_protocol.json'
RUN_OUT = None
DEADLINE = None
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    NAME: dict(context=False, model='ft_transformer', **SETTINGS),
}
iterate.BLEND = []


def check_deadline():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise TimeoutError('FT-Transformer stage budget exhausted; incomplete candidate has no valid score.')


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device) == 'mps':
        torch.mps.manual_seed(seed)


def release_memory(device):
    gc.collect()
    if str(device) == 'mps':
        torch.mps.empty_cache()


def resources(device):
    result = dict(process_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if str(device) == 'mps':
        result.update(mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return result


class FTPreprocessor:
    """Same input/preprocessing contract as TabM, without importing its experiment."""
    def fit(self, x):
        raw = x[NUMERIC].astype(float)
        self.medians = raw.median().fillna(0.)
        numeric = self._impute_log(raw)
        self.keep = numeric.nunique().gt(1).to_numpy()
        if not self.keep.any():
            raise ValueError('At least one nonconstant numerical feature is required.')
        self.numeric_columns = [c for c, keep in zip(NUMERIC, self.keep) if keep]
        self.scaler = StandardScaler().fit(numeric.iloc[:, self.keep])
        self.vocabularies = []
        for col in CATEGORICAL:
            known = sorted(set(x[col].astype('string').fillna('__MISSING__')) - {'__MISSING__'})
            self.vocabularies.append({'__MISSING__': 1, **{v: j+2 for j, v in enumerate(known)}})
        self.cat_cardinalities = [len(v)+1 for v in self.vocabularies] + [2]*len(NUMERIC)
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
        numeric = self.scaler.transform(self._impute_log(raw).iloc[:, self.keep]).astype(np.float32)
        columns = [x[col].astype('string').fillna('__MISSING__').map(vocab).fillna(0).to_numpy(dtype=np.int64)
                   for col, vocab in zip(CATEGORICAL, self.vocabularies)]
        categorical = np.column_stack([*columns, *missing.T])
        if not np.isfinite(numeric).all():
            raise ValueError('Nonfinite transformed numerical values.')
        return numeric, categorical


def make_model(preprocessor, seed, device):
    from rtdl_revisiting_models import FTTransformer
    if importlib.metadata.version(SETTINGS['package']) != SETTINGS['package_version']:
        raise RuntimeError('FT-Transformer package version differs from frozen specification.')
    seed_everything(seed, device)
    kwargs = FTTransformer.get_default_kwargs(n_blocks=SETTINGS['n_blocks'])
    for key in ['n_blocks', 'd_block', 'attention_n_heads', 'attention_dropout',
                'ffn_d_hidden_multiplier', 'ffn_dropout', 'residual_dropout']:
        if kwargs[key] != SETTINGS[key]:
            raise RuntimeError(f'Official default changed: {key}')
    return FTTransformer(n_cont_features=len(preprocessor.numeric_columns),
                         cat_cardinalities=preprocessor.cat_cardinalities,
                         d_out=1, **kwargs).to(device)


def tensors(arrays, device):
    return (torch.as_tensor(arrays[0], dtype=torch.float32, device=device),
            torch.as_tensor(arrays[1], dtype=torch.long, device=device))


def train_epoch(model, optimizer, data, targets, generator):
    model.train()
    order = torch.randperm(len(targets), generator=generator, device='cpu')
    total, seen, steps = 0., 0, 0
    for cpu_idx in order.split(SETTINGS['batch_size']):
        check_deadline()
        idx = cpu_idx.to(targets.device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(data[0][idx], data[1][idx]).squeeze(-1), targets[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite FT-Transformer training loss.')
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())*len(idx)
        seen += len(idx)
        steps += 1
    return dict(training_bce=total/seen, optimizer_steps=steps)


@torch.inference_mode()
def predict_arrays(model, data):
    model.eval()
    predictions = []
    for start in range(0, len(data[0]), 512):
        check_deadline()
        logits = model(data[0][start:start+512], data[1][start:start+512]).squeeze(-1)
        predictions.append(logits.sigmoid().cpu().numpy())
    p = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise FloatingPointError('Invalid FT-Transformer probabilities.')
    return p


def select_epochs(x, y, seed, device):
    started = time.monotonic()
    a, b = train_test_split(np.arange(len(y)), test_size=SETTINGS['inner_validation_fraction'],
                           stratify=y, random_state=seed)
    prep = FTPreprocessor().fit(x.iloc[a])
    model = make_model(prep, seed, device)
    data = tensors(prep.transform(x.iloc[a]), device)
    valid = tensors(prep.transform(x.iloc[b]), device)
    targets = torch.as_tensor(y[a], dtype=torch.float32, device=device)
    optimizer = model.make_default_optimizer()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    best_loss, best_epoch, stale = float('inf'), None, 0
    history = []
    try:
        for epoch in range(1, SETTINGS['max_epochs']+1):
            epoch_started = time.monotonic()
            row = train_epoch(model, optimizer, data, targets, generator)
            loss = float(log_loss(y[b], predict_arrays(model, valid), labels=[0, 1]))
            if not np.isfinite(loss):
                raise FloatingPointError('Nonfinite inner validation loss.')
            history.append(dict(epoch=epoch, **row, validation_log_loss=loss,
                                seconds=time.monotonic()-epoch_started))
            if loss < best_loss:
                best_loss, best_epoch, stale = loss, epoch, 0
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f'FTT inner epoch={epoch} loss={loss:.6f} best={best_epoch}', flush=True)
            if stale >= SETTINGS['patience']:
                break
        stop_reason = 'patience' if stale >= SETTINGS['patience'] else 'epoch_cap'
        metadata = dict(inner_train_rows=len(a), inner_validation_rows=len(b),
            inner_best_log_loss=best_loss, inner_history=history, stop_reason=stop_reason,
            budget_capped=stop_reason == 'epoch_cap',
            inner_optimizer_steps=sum(row['optimizer_steps'] for row in history),
            inner_selected_epoch_steps=best_epoch*math.ceil(len(a)/SETTINGS['batch_size']),
            inner_parameters=sum(p.numel() for p in model.parameters()),
            inner_seconds=time.monotonic()-started, inner_resources=resources(device))
        return best_epoch, metadata
    finally:
        del model, optimizer, data, valid, targets
        release_memory(device)


class FTPredictor:
    """Preprocessor plus CPU model; serialized without GPU optimizer/state."""
    def __init__(self, preprocessor, model, device, selected_epochs, metadata):
        self.preprocessor, self.model = preprocessor, model.cpu()
        self.device, self.selected_epochs, self.metadata = str(device), selected_epochs, metadata

    def predict_proba(self, x):
        device = self.device if self.device != 'mps' or torch.backends.mps.is_available() else 'cpu'
        self.model.to(device)
        try:
            p = predict_arrays(self.model, tensors(self.preprocessor.transform(x), device))
        finally:
            self.model.cpu()
            release_memory(device)
        return np.column_stack([1-p, p])


def fit_ft(x, y, seed, device):
    selected_epochs, metadata = select_epochs(x, y, seed, device)
    started = time.monotonic()
    prep = FTPreprocessor().fit(x)  # Fresh medians, scaling, filtering and vocabulary.
    model = make_model(prep, seed, device)  # Fresh weights and optimizer.
    optimizer = model.make_default_optimizer()
    data = tensors(prep.transform(x), device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    history = []
    try:
        for epoch in range(1, selected_epochs+1):
            epoch_started = time.monotonic()
            row = train_epoch(model, optimizer, data, targets, generator)
            history.append(dict(epoch=epoch, **row, seconds=time.monotonic()-epoch_started))
            if epoch == 1 or epoch % 10 == 0:
                print(f'FTT full refit epoch={epoch}/{selected_epochs}', flush=True)
        metadata.update(seed=int(seed), device=str(device), selected_epochs=selected_epochs,
            outer_train_rows=len(y), retained_numeric_columns=prep.numeric_columns,
            cat_cardinalities=prep.cat_cardinalities, refit_history=history,
            parameters=sum(p.numel() for p in model.parameters()),
            refit_optimizer_steps=sum(row['optimizer_steps'] for row in history),
            refit_seconds=time.monotonic()-started, refit_resources=resources(device))
        return FTPredictor(prep, model, device, selected_epochs, metadata)
    finally:
        del model, optimizer, data, targets
        release_memory(device)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    check_deadline()
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name != NAME:
        raise ValueError(name)
    target = np.asarray(y, dtype=np.int64)
    if len(target) != len(train_idx):
        raise ValueError('Target length must match training indices.')
    x = matrices[False][0]
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    started = time.monotonic()
    predictor = fit_ft(x.iloc[train_idx], target, seed, device)
    p = predictor.predict_proba(x.iloc[predict_idx])[:, 1]
    predictor.metadata.update(seconds=time.monotonic()-started,
                              after_prediction_resources=resources(device))
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT/f'ftt_training_seed{seed}_{key}.json').write_text(json.dumps(predictor.metadata, indent=2)+'\n')
    return p, predictor


def environment_lock():
    return ''.join(f'{name}=={version}\n' for name, version in sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'ft_transformer_policy' in protocol:
        return
    declared = json.loads((iterate.ROOT/PROTOCOL_FILE).read_text())
    if declared['settings'] != SETTINGS:
        raise RuntimeError('Declared FT-Transformer protocol and implementation differ.')
    for filename in ['ft_transformer_candidate.py', 'test_ft_transformer_candidate.py', PROTOCOL_FILE]:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out/'requirements-ft-transformer.lock.txt').write_text(lock)
    protocol['ft_transformer_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['ft_transformer_policy'] = declared
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # Before the very first baseline fit.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def check_environment(out):
    expected = json.loads((out/'protocol.json').read_text())['ft_transformer_environment_sha256']
    if (hashlib.sha256(environment_lock().encode()).hexdigest() != expected
        or hashlib.sha256((out/'requirements-ft-transformer.lock.txt').read_bytes()).hexdigest() != expected):
        raise RuntimeError('FT-Transformer dependency environment changed.')


def probe():
    """Three full-context epochs and serialization only; never computes accuracy."""
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    x = matrices[False][0]
    prep = FTPreprocessor().fit(x.iloc[dev[a]])
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = make_model(prep, iterate.SEEDS[0], device)
    optimizer = model.make_default_optimizer()
    data = tensors(prep.transform(x.iloc[dev[a]]), device)
    target = torch.as_tensor(y[a], dtype=torch.float32, device=device)
    generator = torch.Generator(device='cpu').manual_seed(iterate.SEEDS[0])
    epoch_seconds = []
    started = time.monotonic()
    for _ in range(3):
        before = time.monotonic()
        train_epoch(model, optimizer, data, target, generator)
        epoch_seconds.append(time.monotonic()-before)
    snapshot = resources(device)
    parameters = sum(p.numel() for p in model.parameters())
    predictor = FTPredictor(prep, model, device, 3, {'probe_only': True})
    del model, optimizer, data, target
    release_memory(device)
    query = x.iloc[dev[b[:128]]]
    p = predictor.predict_proba(query)
    with tempfile.TemporaryDirectory(prefix='ftt-probe-') as folder:
        artifact = Path(folder)/'model.joblib'
        joblib.dump(predictor, artifact)
        restored_p = joblib.load(artifact).predict_proba(query)
        np.testing.assert_allclose(restored_p, p, atol=1e-6, rtol=0)
    result = dict(training_rows=len(a), query_rows=len(query), device=device,
        parameters=parameters, epoch_seconds=epoch_seconds, seconds=time.monotonic()-started,
        resources_after_training=snapshot, resources_after_release=resources(device),
        serialization_max_absolute_difference=float(np.max(np.abs(restored_p-p))),
        valid_probabilities=True, accuracy_computed=False,
        note='Resource and backpropagation probe only, not completed training or CV.')
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT, DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_035')
    args = parser.parse_args()
    torch.set_num_threads(4)
    DEADLINE = time.monotonic()+SETTINGS['stage_budget_seconds'][args.stage]
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT/'artifacts'/args.run
    if args.stage in ['confirm', 'export']:
        check_environment(RUN_OUT)
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    sys.modules['ft_transformer_candidate'] = sys.modules[__name__]
    for cls in [FTPreprocessor, FTPredictor]:
        cls.__module__ = 'ft_transformer_candidate'
    main()
