"""Built with PriorLabs-TabPFN: frozen ensemble of two V2 architecture families."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_NO_BROWSER'] = '1'
os.environ['TABPFN_MODEL_VERSION'] = 'v2'
os.environ['TABPFN_MODEL_CACHE_SIZE'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
import torch
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import iterate

BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
NAME = 'v54_TabPFN_architecture_ensemble'
CONTROL = 'TabPFN_v2_compact8_matched'
REVISION = '275f19d2add9c2c0de38f9ebaded3856599fa2e7'
COMPACT = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
           'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNumber', 'Side']
FLAGS = dict(ENABLE_GPU_PREPROCESSING=False, TRANSFORM_TEXT=False,
             TRANSFORM_DATES=False, SUBSAMPLE_SAMPLES=None)
CHECKPOINTS = {
    'default': dict(filename='tabpfn-v2-classifier-v2_default.ckpt', bytes=29009539,
        sha256='cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49',
        features_per_group=2, emsize=192, nhead=6, nlayers=12),
    'gn2p4bpt': dict(filename='tabpfn-v2-classifier-gn2p4bpt.ckpt', bytes=12926661,
        sha256='de01a095da7a083b25a74d1f7d85de524e8bd6137936092feab9790673ad9215',
        features_per_group=1, emsize=128, nhead=4, nlayers=12),
    'vutqq28w': dict(filename='tabpfn-v2-classifier-vutqq28w.ckpt', bytes=12926661,
        sha256='c060b5f61480e8a6ec8cbf909a0fc236563eb4628df75b7b8c2d1f8fc1b9cf4e',
        features_per_group=1, emsize=128, nhead=4, nlayers=12),
    'znskzxi4': dict(filename='tabpfn-v2-classifier-znskzxi4.ckpt', bytes=12926661,
        sha256='fadb46928ddd70a53c6e1b205772e78784bf7bea5132ba6adbd48f201d8e9076',
        features_per_group=1, emsize=128, nhead=4, nlayers=12),
}
ENSEMBLE = [dict(checkpoint=k, n_estimators=2, weight=.5 if k == 'default' else 1/6)
            for k in CHECKPOINTS]
CONTROL_COMPONENTS = [dict(checkpoint='default', n_estimators=8, weight=1.)]
MANIFEST = 'reports/pretrained_architecture_sources.json'
RUN_OUT = None
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    CONTROL: dict(context=False, model='tabpfn_architecture', components=CONTROL_COMPONENTS,
                  columns=COMPACT, inference_config=FLAGS, checkpoints=CHECKPOINTS),
    NAME: dict(context=False, model='tabpfn_architecture', components=ENSEMBLE,
               columns=COMPACT, inference_config=FLAGS, checkpoints=CHECKPOINTS),
}
iterate.BLEND = []


def checkpoint_path(spec):
    return iterate.ROOT / 'artifacts/pretrained' / spec['filename']


def verify_checkpoint(spec):
    path = checkpoint_path(spec)
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise RuntimeError(f'Pretrained checkpoint missing/wrong size: {spec["filename"]}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise RuntimeError(f'Pretrained checkpoint SHA256 mismatch: {spec["filename"]}')
    return path


def verify_all():
    for spec in CHECKPOINTS.values():
        verify_checkpoint(spec)


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def make_model(spec, seed, n_estimators, categorical_indices, flags):
    path = verify_checkpoint(spec)
    if not torch.backends.mps.is_available():
        raise RuntimeError('This frozen ensemble requires local MPS; no automatic cloud/CPU fallback.')
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, model_path=path, n_estimators=n_estimators,
        device='mps', random_state=seed, n_preprocessing_jobs=1,
        inference_precision=torch.float32, ignore_pretraining_limits=True,
        categorical_features_indices=categorical_indices, inference_config=flags)


class ArchitecturePredictor:
    """Only immutable checkpoint references and context are retained, never GPU engines."""
    def __init__(self, x, y, cats, seed, components):
        self.columns = list(COMPACT)
        self.context_x = x[self.columns].copy()
        self.context_y = np.asarray(y, dtype=np.int64).copy()
        self.seed = int(seed)
        self.components = copy.deepcopy(components)
        self.checkpoints = copy.deepcopy(CHECKPOINTS)
        self.flags = dict(FLAGS)
        self.categorical_indices = [self.columns.index(c) for c in cats if c in self.columns]
        self.validate()

    def validate(self):
        if len(self.context_x) != len(self.context_y):
            raise ValueError('Training context and target lengths differ.')
        weights = [c['weight'] for c in self.components]
        if not weights or not np.isfinite(weights).all() or min(weights) <= 0 or not np.isclose(sum(weights), 1.):
            raise ValueError('The predeclared component weights must be positive and sum to one.')
        for component in self.components:
            verify_checkpoint(self.checkpoints[component['checkpoint']])

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.validate()

    def predict_proba(self, raw):
        query = raw[self.columns]
        total = np.zeros((len(query), 2), dtype=np.float64)
        resources = []
        for component in self.components:
            spec = self.checkpoints[component['checkpoint']]
            model = None
            started = time.monotonic()
            try:
                model = make_model(spec, self.seed, component['n_estimators'],
                                   self.categorical_indices, self.flags)
                model.fit(self.context_x, self.context_y)
                config = model.configs_[0]
                actual = {key: int(getattr(config, key)) for key in
                          ['features_per_group', 'emsize', 'nhead', 'nlayers']}
                if any(actual[key] != spec[key] for key in actual):
                    raise RuntimeError('Loaded model architecture differs from pinned metadata.')
                p = model.predict_proba(query)
                if (p.shape != total.shape or not np.isfinite(p).all()
                    or not ((p >= 0) & (p <= 1)).all()
                    or not np.allclose(p.sum(axis=1), 1., atol=1e-6)):
                    raise RuntimeError('Invalid component probabilities; do not drop/reweight a failed member.')
                total += component['weight'] * p
                resources.append(dict(**component, sha256=spec['sha256'], architecture=actual,
                    seconds=time.monotonic()-started,
                    mps_driver_allocated_bytes=torch.mps.driver_allocated_memory()))
                print('Architecture component', json.dumps(resources[-1]), flush=True)
            finally:
                # No estimator is stored on self; next checkpoint starts after release.
                model = None
                release_memory()
        self.last_resources = resources
        return total


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in [CONTROL, NAME]:
        raise ValueError(name)
    x, cats = matrices[False]
    predictor = ArchitecturePredictor(x.iloc[train_idx], y, cats, seed,
                                      iterate.CONFIGS[name]['components'])
    p = predictor.predict_proba(x.iloc[predict_idx])[:, 1]
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT/f'{name}_seed{seed}_{key}.json').write_text(json.dumps(dict(
            training_rows=len(train_idx), query_rows=len(predict_idx),
            components=predictor.last_resources), indent=2)+'\n')
    return p, predictor


def environment_lock():
    return ''.join(f'{n}=={v}\n' for n, v in sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'architecture_policy' in protocol:
        return
    verify_all()
    for filename in ['tabpfn_architecture.py', MANIFEST]:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out/'requirements-architecture.lock.txt').write_text(lock)
    protocol['architecture_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['architecture_policy'] = (
        'One fixed family-balanced ensemble: default two-feature-token model weight .5; '
        'all three official one-feature-token models each weight 1/6. Each model has two '
        'preprocessing views. Control is default-only eight views with identical compact13 '
        'and preprocessing settings. No member/weight tuning or prediction cache. Sequential '
        'MPS execution; any component failure aborts the whole candidate. Exact checkpoint '
        'bytes pinned by SHA256 and official HF revision. Author-reported synthetic pretraining, '
        'not an independent audit of each checkpoint training history. Built with PriorLabs-TabPFN.')
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def check_environment(out):
    verify_all()
    expected = json.loads((out/'protocol.json').read_text())['architecture_environment_sha256']
    if (hashlib.sha256(environment_lock().encode()).hexdigest() != expected
        or hashlib.sha256((out/'requirements-architecture.lock.txt').read_bytes()).hexdigest() != expected):
        raise RuntimeError('Architecture experiment dependency environment changed.')


def probe():
    verify_all()
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    x, cats = matrices[False]
    predictor = ArchitecturePredictor(x.iloc[dev[a]], y[a], cats, iterate.SEEDS[0], ENSEMBLE)
    query = x.iloc[dev[b[:128]]]
    started = time.monotonic()
    p = predictor.predict_proba(query)
    first_resources = predictor.last_resources
    with tempfile.TemporaryDirectory(prefix='architecture-probe-') as folder:
        path = Path(folder)/'model.joblib'
        joblib.dump(predictor, path)
        restored = joblib.load(path)
        restored_p = restored.predict_proba(query)
        np.testing.assert_allclose(restored_p, p, atol=1e-6, rtol=0)
    result = dict(full_training_context_rows=len(a), query_rows=128,
                  components=first_resources, valid_probabilities=True,
                  serialization_max_absolute_difference=float(np.max(np.abs(p-restored_p))),
                  seconds=time.monotonic()-started, accuracy_computed=False,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  mps_driver_after_release_bytes=torch.mps.driver_allocated_memory(),
                  mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_026')
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
    sys.modules['tabpfn_architecture'] = sys.modules[__name__]
    ArchitecturePredictor.__module__ = 'tabpfn_architecture'
    main()
