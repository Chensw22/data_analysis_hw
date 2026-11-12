"""One fixed four-view preprocessing mirror mean of the same Mitra-v2 weights.

Only screen is exposed. A/S eligibility and any export are separate root tasks.
"""
import argparse
import gc
import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

import iterate
BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
import mitra_candidate as mitra

ROOT = Path(__file__).resolve().parent
CONTROL = 'v71_Mitra2_zeroshot'
NAME = 'v74_Mitra_mirror4'
OFFSETS = {CONTROL: (0,), NAME: (0, 100, 200, 300)}
REVIEW = 'reports/top5_mirror_ensemble_review.json'
VALIDATOR_SOURCE = 'top5_validate.py'
SETTINGS = dict(query_chunk=2318, device='mps', threshold=.5,
    full_support=True, mirror_offsets=[0, 100, 200, 300],
    probability_weights=[.25, .25, .25, .25], probability_mean_dtype='float64',
    checkpoint_sha256=mitra.WEIGHT_SHA, screen_budget_seconds=1800,
    stability_budget_seconds=3600,
    weights_and_context_dtype='float32', mps_autocast_dtype='bfloat16',
    mps_memory_fraction=.65, sequential_views=True, gradient_updates=0)
RUN_OUT = None


def hash_array(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def write_json(path, value):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def verify_reference_contract():
    history = ROOT/'artifacts/run_038'
    protocol = json.loads((history/'protocol.json').read_text())
    for relative, expected in protocol['sources'].items():
        if mitra.sha256(ROOT/relative) != expected:
            raise RuntimeError(f'Frozen Mitra dependency changed: {relative}')
    expected = protocol['mitra_environment_sha256']
    if hashlib.sha256(mitra.environment_lock().encode()).hexdigest() != expected or mitra.sha256(history/'requirements-mitra.lock.txt') != expected:
        raise RuntimeError('Use the unchanged run038 Mitra environment.')
    for relative, expected in json.loads((ROOT/REVIEW).read_text())['sources_sha256'].items():
        if mitra.sha256(ROOT/relative) != expected:
            raise RuntimeError(f'Prespecified mirror review source changed: {relative}')
    mitra.verify_checkpoint(mitra.CHECKPOINT_DIR)
    return protocol


class MirrorContextPredictor:
    """CPU raw context and recipe only; no model tensors or four-state cache.

    Prediction rebuilds each view from the SHA-pinned external checkpoint,
    releases it fully, then averages all required results. The checkpoint and
    frozen Python dependencies must remain available; no automatic downloads.
    """
    def __init__(self, training, labels, train_ids, seed, version, device='mps'):
        if version not in OFFSETS:
            raise ValueError('Only the fixed matched control and four-view recipe are allowed.')
        self.columns = list(mitra.COMPACT)
        self.training = training[self.columns].copy(deep=True)
        self.labels = mitra.binary_labels(labels, len(training))
        self.train_ids = np.array(train_ids, dtype=np.int64, copy=True)
        if len(self.train_ids) != len(training) or len(np.unique(self.train_ids)) != len(self.train_ids):
            raise ValueError('Training row IDs must be unique and aligned.')
        self.seed = int(seed)
        self.version = version
        self.offsets = tuple(OFFSETS[version])
        self.query_chunk = SETTINGS['query_chunk']
        self.checkpoint_dir = str(mitra.CHECKPOINT_DIR)
        self.checkpoint_sha256 = mitra.WEIGHT_SHA
        self.device = device
        self.expected_mirror_hashes = {}
        self.metadata = {}

    def validate_recipe(self):
        if self.offsets != OFFSETS.get(self.version) or self.query_chunk != 2318:
            raise RuntimeError('Serialized mirror recipe changed.')
        if self.checkpoint_sha256 != mitra.WEIGHT_SHA:
            raise RuntimeError('Serialized checkpoint identity changed.')
        if len(self.training) != len(self.train_ids) or len(self.labels) != len(self.train_ids):
            raise RuntimeError('Serialized full context is misaligned.')

    def predict_proba(self, query, prediction_ids=None, log_path=None, prediction_path=None):
        self.validate_recipe()
        if prediction_ids is not None:
            prediction_ids = np.asarray(prediction_ids, dtype=np.int64)
            if len(prediction_ids) != len(query) or len(np.unique(prediction_ids)) != len(query):
                raise ValueError('Prediction IDs must be unique and aligned.')
            if np.intersect1d(self.train_ids, prediction_ids).size:
                raise ValueError('Training and prediction IDs must be disjoint.')
        self.metadata = dict(status='incomplete', version=self.version,
            base_fit_seed=self.seed, offsets=list(self.offsets),
            weights=[1/len(self.offsets)]*len(self.offsets), checkpoint_sha256=self.checkpoint_sha256,
            training_rows=len(self.training), query_rows=len(query),
            train_ids_sha256=hash_array(self.train_ids, np.int64),
            query_ids_verified=prediction_ids is not None,
            query_ids_sha256=hash_array(prediction_ids, np.int64) if prediction_ids is not None else None,
            full_support=True, query_chunk=self.query_chunk,
            interpretation='Preprocessing mirror views of identical pretrained weights, not separately trained networks.',
            components=[])
        write_json(log_path, self.metadata)
        started = time.monotonic()
        predictions = []
        try:
            for offset in self.offsets:
                mitra.check_deadline()
                component = None
                tick = time.monotonic()
                try:
                    component = mitra.fit_mitra(self.training, self.labels, self.train_ids,
                        self.seed+offset, self.device, 'zero-shot',
                        checkpoint_dir=Path(self.checkpoint_dir), query_chunk=self.query_chunk)
                    if len(component.support) != len(self.training):
                        raise RuntimeError('A mirror view truncated the full training context.')
                    mask = np.asarray(component.preprocessor.upstream.mirror, dtype=np.int8)
                    mask_hash = hash_array(mask, np.int8)
                    if offset in self.expected_mirror_hashes and self.expected_mirror_hashes[offset] != mask_hash:
                        raise RuntimeError('Rebuilt preprocessing mirror differs from the persisted recipe.')
                    prediction_started = time.monotonic()
                    probability = np.asarray(component.predict_proba(query), dtype=np.float64)
                    if (probability.shape != (len(query), 2) or not np.isfinite(probability).all()
                            or (probability < 0).any() or (probability > 1).any()
                            or not np.allclose(probability.sum(axis=1), 1., atol=1e-6, rtol=0)):
                        raise FloatingPointError('Invalid mirror component probabilities.')
                    predictions.append(probability.copy())
                    self.expected_mirror_hashes[offset] = mask_hash
                    self.metadata['components'].append(dict(offset=offset, seed=self.seed+offset,
                        mirror_mask=mask.reshape(-1).tolist(), mirror_mask_shape=list(mask.shape),
                        mirror_hash_encoding='int8_row_major', mirror_sha256=mask_hash,
                        resolved_full_support_rows=len(component.support),
                        offset0_is_original_v71=offset == 0,
                        probabilities_sha256=hash_array(probability, np.float64),
                        prediction_seconds=time.monotonic()-prediction_started,
                        fit_predict_seconds=time.monotonic()-tick,
                        resources_before_release=mitra.snapshot(self.device),
                        original_fit_metadata=component.metadata))
                finally:
                    if component is not None:
                        component.release()
                    del component
                    gc.collect()
                    mitra.release_memory()
                self.metadata['components'][-1]['resources_after_release'] = mitra.snapshot(self.device)
                write_json(log_path, self.metadata)
            if len(predictions) != len(self.offsets):
                raise RuntimeError('Incomplete ensemble: no partial mean is allowed.')
            stacked = np.stack(predictions, axis=0)
            mean = np.mean(stacked, axis=0, dtype=np.float64)
            self.metadata.update(status='complete', completed_views=len(predictions),
                seconds=time.monotonic()-started, mean_probabilities_sha256=hash_array(mean, np.float64),
                mean_class1_probability=float(mean[:,1].mean()) if len(mean) else None,
                per_query_class1_view_variance_mean=float(np.var(stacked[:,:,1],axis=0).mean()) if len(mean) else None)
            if prediction_path is not None:
                np.savez(prediction_path, component_probabilities=stacked, mean_probabilities=mean,
                    offsets=np.asarray(self.offsets), train_ids=self.train_ids,
                    prediction_ids=prediction_ids if prediction_ids is not None else np.array([],dtype=np.int64))
            write_json(log_path, self.metadata)
            return mean
        except Exception as error:
            self.metadata.update(status='failed_incomplete_ensemble', error_type=type(error).__name__,
                error=str(error), completed_views=len(predictions), seconds=time.monotonic()-started)
            write_json(log_path, self.metadata)
            raise

    def __getstate__(self):
        self.validate_recipe()
        return dict(self.__dict__)  # CPU DataFrame/arrays/scalars only; no fitted model.

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.validate_recipe()


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in OFFSETS:
        raise ValueError(name)
    train_idx, predict_idx = np.asarray(train_idx), np.asarray(predict_idx)
    if np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Training and prediction IDs must be disjoint.')
    x = matrices[False][0]
    artifact = MirrorContextPredictor(x.iloc[train_idx], y, train_idx, seed, name)
    key = hash_array(train_idx, np.int64)[:12]
    log = RUN_OUT/f'{name}_seed{seed}_{key}.json' if RUN_OUT is not None else None
    saved = RUN_OUT/f'{name}_seed{seed}_{key}_views.npz' if RUN_OUT is not None else None
    probability = artifact.predict_proba(x.iloc[predict_idx], predict_idx, log, saved)[:,1]
    return probability, artifact


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'mirror_ensemble_policy' in protocol:
        return
    original = verify_reference_contract()
    review = json.loads((ROOT/REVIEW).read_text())
    names = set(original['sources']) | set(review['sources_sha256']) | {
        'mitra_mirror_ensemble.py', 'test_mitra_mirror_ensemble.py', REVIEW, VALIDATOR_SOURCE,
        'audit_top5_references.py', 'test_top5_validate.py',
        'test_audit_top5_references.py', 'reports/top5_validation_review.json',
        'reports/top5_authorization.json'}
    for name in sorted(names):
        protocol['sources'][name] = mitra.sha256(ROOT/name)
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    lock = mitra.environment_lock()
    (out/'mirror_environment.lock.txt').write_text(lock)
    protocol.update(mirror_ensemble_policy=dict(settings=SETTINGS,
        recipes={key:list(value) for key,value in OFFSETS.items()}, review=review,
        persistence='CPU raw training context/labels/IDs/recipe plus external SHA-pinned checkpoint; no model-state cache.',
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest()),
        gate='Screen evidence only. Separate root A/S eligibility review required; this CLI has no confirm/export or submission authorization.')
    write_json(path, protocol)


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # Before the very first v6 baseline model.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:
        return
    if iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE:
        raise RuntimeError('A different wrapper already patched iterate; use a fresh process.')
    if BASE_FIT.__module__ != 'iterate':
        raise RuntimeError('Original iterate baseline was not captured.')
    iterate.CONFIGS = {iterate.BASELINE:dict(context=False, model='catboost', **iterate.CAT)}
    for name, offsets in OFFSETS.items():
        iterate.CONFIGS[name] = {**SETTINGS, 'context':False, 'model':'mitra_mirror',
            'offsets':list(offsets), 'mirror_offsets':list(offsets),
            'probability_weights':[1/len(offsets)]*len(offsets)}
    iterate.BLEND = []
    iterate.fit, iterate.evaluate = fit, evaluate


def configure_runtime(out, stage='screen'):
    global RUN_OUT
    if stage not in ('screen', 'stability'):
        raise ValueError('Only screen and separate stability evaluation are supported.')
    configure()
    RUN_OUT = Path(out)
    if not mitra.torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS backend unavailable; no automatic fallback.')
    mitra.torch.set_num_threads(4)
    mitra.torch.mps.set_per_process_memory_fraction(SETTINGS['mps_memory_fraction'])
    mitra.DEADLINE = time.monotonic()+SETTINGS[f'{stage}_budget_seconds']


def assert_frozen(out):
    out = Path(out)
    iterate.assert_frozen(out)
    protocol = json.loads((out/'protocol.json').read_text())
    expected = protocol['mirror_ensemble_policy']['environment_sha256']
    if hashlib.sha256(mitra.environment_lock().encode()).hexdigest() != expected or mitra.sha256(out/'mirror_environment.lock.txt') != expected:
        raise RuntimeError('Frozen mirror environment changed.')
    verify_reference_contract()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen'])
    parser.add_argument('--run', default='run_040')
    args = parser.parse_args()
    configure_runtime(ROOT/'artifacts'/args.run)
    def timeout(*_):
        raise TimeoutError('Mirror screen deadline exceeded; no partial ensemble.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, SETTINGS['screen_budget_seconds'])
    try:
        iterate.screen(RUN_OUT)
        selection = json.loads((RUN_OUT/'selection.json').read_text())
        selection.update(stage='screen_only_pending_separate_A_S_review', automatic_eligibility=False,
                         confirm_export_supported=False)
        write_json(RUN_OUT/'selection.json', selection)
    except Exception as error:
        if RUN_OUT.exists():
            write_json(RUN_OUT/'screen_failure.json', dict(status='failed_inconclusive', error=repr(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        mitra.DEADLINE = None


if __name__ == '__main__':
    sys.modules['mitra_mirror_ensemble'] = sys.modules[__name__]
    MirrorContextPredictor.__module__ = 'mitra_mirror_ensemble'
    main()
