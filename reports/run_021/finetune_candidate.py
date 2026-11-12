"""Built with PriorLabs-TabPFN: local V2 fine-tuning with isolated validation."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_NO_BROWSER'] = '1'
os.environ['TABPFN_MODEL_VERSION'] = 'v2'
# Never share mutable trained module objects across independent CV folds.
os.environ['TABPFN_MODEL_CACHE_SIZE'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.model_loading import save_tabpfn_model

import iterate

BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
FROZEN = 'v47_TabPFN_frozen1'
TUNED = 'v48_TabPFN_finetuned1'
COMPACT = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
           'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNumber', 'Side']
WEIGHTS = iterate.ROOT / 'artifacts/pretrained/tabpfn-v2-classifier-v2_default.ckpt'
WEIGHT_HASH = 'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
FLAGS = dict(ENABLE_GPU_PREPROCESSING=False, TRANSFORM_TEXT=False,
             TRANSFORM_DATES=False, SUBSAMPLE_SAMPLES=None)
POLICY = dict(epochs=10, learning_rate=1e-5, weight_decay=.01,
              validation_split_ratio=.2, n_finetune_ctx_plus_query_samples=640,
              finetune_ctx_query_split_ratio=.2, n_inference_subsample_samples=None,
              early_stopping=True, early_stopping_patience=3, validation_frequency=1,
              min_delta=1e-4, grad_clip_value=1., use_lr_scheduler=False,
              n_estimators_finetune=1, n_estimators_validation=1,
              n_estimators_final_inference=1, use_activation_checkpointing=True,
              save_checkpoint_interval=None, use_fixed_preprocessing_seed=True,
              eval_metric='log_loss')
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    FROZEN: dict(context=False, model='tabpfn_frozen', columns=COMPACT, n_estimators=1,
                 inference_config=FLAGS, checkpoint_sha256=WEIGHT_HASH),
    TUNED: dict(context=False, model='tabpfn_finetuned', columns=COMPACT, policy=POLICY,
                inference_config=FLAGS, checkpoint_sha256=WEIGHT_HASH),
}
iterate.BLEND = []
RUN_OUT = None


def verify_weights():
    if not WEIGHTS.is_file() or hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHT_HASH:
        raise RuntimeError('Local pretrained V2 checkpoint missing or SHA256 mismatch.')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extra_kwargs(cats, path=WEIGHTS):
    return dict(model_path=path, n_preprocessing_jobs=1, inference_precision=torch.float32,
                categorical_features_indices=[COMPACT.index(c) for c in cats if c in COMPACT],
                inference_config=dict(FLAGS))


def frozen_model(seed, cats, path=WEIGHTS):
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=1, device='mps', random_state=seed,
        ignore_pretraining_limits=True, **extra_kwargs(cats, path))


class AuditLogger:
    """Local audit only; no cloud logger or modification of optimization."""
    def __init__(self, owner):
        self.owner = owner
        self.steps = []

    def setup(self, config):
        pass

    def log_step(self, metrics, step):
        if not np.isfinite(metrics['train/loss']):
            raise RuntimeError('Nonfinite fine-tuning loss.')
        model = self.owner.finetuned_estimator_.model_
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all().item()):
            raise RuntimeError('No gradients or nonfinite fine-tuning gradients.')
        if not self.steps:
            changed, delta = weight_change(model, self.owner.initial_parameters)
            if not changed:
                raise RuntimeError('The optimizer step did not change any trainable parameter.')
            self.owner.first_update_max_absolute_change = delta
        self.steps.append(dict(step=int(step), **{k: float(v) for k, v in metrics.items()}))

    def log_epoch(self, metrics, step):
        pass

    def finish(self):
        pass


def weight_change(model, initial):
    largest = 0.
    for name, value in model.named_parameters():
        if name in initial:
            current = value.detach().cpu()
            if not torch.isfinite(current).all():
                raise RuntimeError('Nonfinite model parameter.')
            largest = max(largest, float((current - initial[name]).abs().max()))
    return largest > 0, largest


class AuditedFinetuner(FinetunedTabPFNClassifier):
    """Keep the official loop; reject swallowed validation errors and audit steps."""
    def _setup_batch(self, batch):
        super()._setup_batch(batch)
        if not hasattr(self, 'initial_parameters'):
            self.initial_parameters = {
                n: p.detach().cpu().clone() for n, p in self.finetuned_estimator_.model_.named_parameters()
                if p.requires_grad}

    def _evaluate_model(self, *args, **kwargs):
        result = super()._evaluate_model(*args, **kwargs)
        if not np.isfinite(result.primary) or not np.isfinite(result.secondary['log_loss']):
            raise RuntimeError('Official fine-tuning validation returned nonfinite metrics.')
        return result

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):
        if not hasattr(self, 'validation_audit'):
            self.validation_audit = []
        self.validation_audit.append(dict(epoch=int(epoch + 1),
            log_loss=float(eval_result.secondary['log_loss']), primary=float(eval_result.primary)))
        super()._log_epoch_evaluation(epoch, eval_result, mean_train_loss)


def selected_epoch(history, min_delta=POLICY['min_delta']):
    """Mirror official improvement semantics, including the unfine-tuned epoch 0."""
    if not history or history[0]['epoch'] != 0:
        raise RuntimeError('Missing epoch-0 validation audit.')
    best = history[0]
    for row in history[1:]:
        if row['primary'] > best['primary'] + min_delta:
            best = row
    return best


def make_tuner(seed, cats, **overrides):
    tuner = AuditedFinetuner(model_version=ModelVersion.V2, device='mps', random_state=seed,
                            extra_classifier_kwargs=extra_kwargs(cats), **{**POLICY, **overrides})
    tuner.experiment_logger = AuditLogger(tuner)
    return tuner


class FinetunePredictor:
    """Portable artifact embeds tuned weights, their digest, and full training context."""
    def __init__(self, model, context_x, context_y, seed, cats, metadata):
        self.model = model
        self.context_x = context_x.copy()
        self.context_y = np.asarray(context_y).copy()
        self.seed, self.cats, self.metadata = seed, cats, metadata
        self.columns = list(COMPACT)

    def predict_proba(self, x):
        return self.model.predict_proba(x[self.columns])

    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items() if k != 'model'}
        with tempfile.TemporaryDirectory(prefix='tabpfn-export-') as folder:
            path = Path(folder) / 'tuned.ckpt'
            save_tabpfn_model(self.model, path)
            state['checkpoint_bytes'] = path.read_bytes()
        state['checkpoint_sha256'] = hashlib.sha256(state['checkpoint_bytes']).hexdigest()
        state['inference_params'] = self.model.get_params(deep=False).copy()
        state['inference_params'].pop('model_path', None)
        return state

    def __setstate__(self, state):
        content = state.pop('checkpoint_bytes')
        expected = state.pop('checkpoint_sha256')
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError('Embedded fine-tuned checkpoint SHA256 mismatch.')
        params = state.pop('inference_params')
        self.__dict__.update(state)
        with tempfile.TemporaryDirectory(prefix='tabpfn-restore-') as folder:
            path = Path(folder) / 'tuned.ckpt'
            path.write_bytes(content)
            self.model = TabPFNClassifier(model_path=path, **params)
            self.model.fit(self.context_x, self.context_y)


def fit_candidate(name, x, y, cats, seed, checkpoint_dir):
    verify_weights()
    if not torch.backends.mps.is_available():
        raise RuntimeError('This frozen experiment requires MPS; no automatic CPU/cloud retraining.')
    seed_all(seed)
    if name == FROZEN:
        model = frozen_model(seed, cats)
        model.fit(x, y)
        metadata = dict(method='frozen', selected_epoch=0, optimizer_steps=0)
    elif name == TUNED:
        if checkpoint_dir.exists():
            raise RuntimeError('Fine-tuning checkpoint directory must be fresh; auto-resume forbidden.')
        tuner = make_tuner(seed, cats)
        tuner.fit(x, y, output_dir=checkpoint_dir)
        if not tuner.experiment_logger.steps:
            raise RuntimeError('No fine-tuning optimizer steps were executed.')
        best = selected_epoch(tuner.validation_audit)
        changed, delta = weight_change(tuner.finetuned_estimator_.model_, tuner.initial_parameters)
        # Official finalization includes all supplied outer-training rows as context.
        if len(tuner.X_) != len(x) or len(tuner.y_) != len(y):
            raise RuntimeError('Final inference context omitted outer training rows.')
        model = tuner.finetuned_inference_classifier_
        metadata = dict(method='80pct_gradient_20pct_earlystop_full_outer_context',
            selected_epoch=best['epoch'], selected_inner_log_loss=best['log_loss'],
            inner_validation_rows=int(len(y) * POLICY['validation_split_ratio']),
            inner_gradient_rows=len(y)-int(len(y) * POLICY['validation_split_ratio']),
            optimizer_steps=len(tuner.experiment_logger.steps),
            first_update_max_absolute_change=tuner.first_update_max_absolute_change,
            selected_weights_changed=changed, selected_max_absolute_change=delta,
            finite_gradients=True, validation_history=tuner.validation_audit,
            training_history=tuner.experiment_logger.steps)
        tuner.experiment_logger.owner = None
        del tuner
    else:
        raise ValueError(name)
    metadata.update(seed=int(seed), outer_train_rows=len(y), inference_context_rows=len(y),
                    pretrained_sha256=WEIGHT_HASH, device='mps')
    return FinetunePredictor(model, x, y, seed, cats, metadata)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    raw, cats = matrices[False]
    x = raw[COMPACT]
    y = np.asarray(y, dtype=np.int64)
    if len(y) != len(train_idx):
        raise ValueError('Labels must match outer training rows.')
    started = time.monotonic()
    key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
    # A unique temporary directory prevents official checkpoint auto-resume.
    with tempfile.TemporaryDirectory(prefix='tabpfn-finetune-') as folder:
        predictor = fit_candidate(name, x.iloc[train_idx], y, cats, seed, Path(folder)/'checkpoints')
    p = predictor.predict_proba(raw.iloc[predict_idx])[:, 1]
    if not np.isfinite(p).all():
        raise RuntimeError('Nonfinite outer predictions.')
    predictor.metadata['seconds'] = time.monotonic() - started
    print(name, json.dumps(predictor.metadata), flush=True)
    if RUN_OUT is not None:
        (RUN_OUT / f'{name}_seed{seed}_{key}.json').write_text(json.dumps(predictor.metadata, indent=2)+'\n')
    gc.collect()
    torch.mps.empty_cache()
    return p, predictor


def environment_lock():
    return ''.join(f'{n}=={v}\n' for n, v in sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'finetune_policy' in protocol:
        return
    verify_weights()
    source = Path(__file__).resolve()
    shutil.copy2(source, out/source.name)
    protocol['sources'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out/'requirements-finetune.lock.txt').write_text(lock)
    protocol['finetune_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['pretrained_sha256'] = WEIGHT_HASH
    protocol['finetune_policy'] = ('Official V2 full-parameter fine-tuning on inner80%, inner20% logloss early '
        'stopping including epoch0, then all outertrain rows as inference context. No full-data gradient refit. '
        'No outer-validation label access. Same compact13, 1estimator and preprocessing flags for frozen control. '
        '640 is the official target chunk size, not a strict maximum; actual chunks can be about742 rows. '
        'No automatic CPU/cloud training fallback. Built with PriorLabs-TabPFN.')
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def check_environment(out):
    verify_weights()
    protocol = json.loads((out/'protocol.json').read_text())
    expected = protocol['finetune_environment_sha256']
    if (hashlib.sha256(environment_lock().encode()).hexdigest() != expected
        or hashlib.sha256((out/'requirements-finetune.lock.txt').read_bytes()).hexdigest() != expected):
        raise RuntimeError('Fine-tuning environment changed since screening.')
    if protocol['pretrained_sha256'] != WEIGHT_HASH:
        raise RuntimeError('Pretrained checkpoint digest changed since screening.')


def probe():
    verify_weights()
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    raw, cats = matrices[False]
    outer_x, outer_y = raw.iloc[dev[a]][COMPACT], y[a]
    rows, _ = train_test_split(np.arange(len(a)), train_size=742,
                              stratify=outer_y, random_state=iterate.SEEDS[0])
    seed_all(iterate.SEEDS[0])
    started = time.monotonic()
    tuner = make_tuner(iterate.SEEDS[0], cats, epochs=1, validation_split_ratio=0,
                       early_stopping=False)
    with tempfile.TemporaryDirectory(prefix='tabpfn-backward-probe-') as folder:
        tuner.fit(outer_x.iloc[rows], outer_y[rows], output_dir=Path(folder)/'checkpoints')
    changed, delta = weight_change(tuner.finetuned_estimator_.model_, tuner.initial_parameters)
    if not changed or len(tuner.experiment_logger.steps) != 1:
        raise RuntimeError('Expected one actual optimizer update in 742-row probe.')
    model = tuner.finetuned_inference_classifier_
    model.fit(outer_x, outer_y)
    predictor = FinetunePredictor(model, outer_x, outer_y, iterate.SEEDS[0], cats, {})
    query = raw.iloc[dev[b[:128]]]
    p = predictor.predict_proba(query)
    assert p.shape == (128, 2) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    with tempfile.TemporaryDirectory(prefix='tabpfn-roundtrip-') as folder:
        path = Path(folder)/'model.joblib'
        joblib.dump(predictor, path)
        restored = joblib.load(path)
        restored_p = restored.predict_proba(query)
        np.testing.assert_allclose(restored_p, p, atol=1e-6, rtol=0)
    result = dict(training_probe_rows=742, optimizer_steps=1, finite_gradients=True,
                  weights_changed=True, max_absolute_weight_change=delta,
                  full_context_rows=len(a), prediction_rows=128, valid_probabilities=True,
                  serialization_max_absolute_difference=float(np.max(np.abs(restored_p-p))),
                  seconds=time.monotonic()-started, accuracy_computed=False,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                  mps_recommended_max_bytes=torch.mps.recommended_max_memory(),
                  probe_note='Compatibility/backpropagation only; weights are never reused for CV.')
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_021')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT/'artifacts'/args.run
    if args.stage in ['confirm', 'export']:
        check_environment(RUN_OUT)
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    sys.modules['finetune_candidate'] = sys.modules[__name__]
    FinetunePredictor.__module__ = 'finetune_candidate'
    main()
