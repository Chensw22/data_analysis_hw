"""Built with PriorLabs-TabPFN: one frozen default-checkpoint view-budget test."""
import argparse
import copy
import dataclasses
import hashlib
import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import tabpfn_architecture as architecture
import iterate
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

BASE_FIT, BASE_EVALUATE = architecture.BASE_FIT, architecture.BASE_EVALUATE
NAME = 'v65_TabPFN_default32'
CONTROL = architecture.CONTROL
COMPACT, FLAGS = list(architecture.COMPACT), dict(architecture.FLAGS)
COMPONENTS = [dict(checkpoint='default', n_estimators=32, weight=1.)]
CONTROL_COMPONENTS = copy.deepcopy(architecture.CONTROL_COMPONENTS)
RUN_OUT = None
POLICY = (
    'Only n_estimators changes: default checkpoint 32 vs independently fitted matched 8. '
    'Identical compact13, flags, seed, float32 MPS and probability averaging. '
    'Views are NOT prefix stable: n_estimators changes feature/class permutation generation; '
    'never reuse 8-view predictions or truncate 32 views to reconstruct the control. '
    'No architecture, view, temperature, weight or threshold selection. '
    'MPS executes one view per forward; CPU preprocessing caches scale with view count. '
    'Resource snapshots are not continuous GPU peak measurements. '
    'A failure aborts the candidate, with no partial ensemble. Built with PriorLabs-TabPFN.')

iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    **{name: dict(context=False, model='tabpfn_views', components=components,
                  columns=COMPACT, inference_config=FLAGS,
                  checkpoints={'default': architecture.CHECKPOINTS['default']})
       for name, components in [(CONTROL, CONTROL_COMPONENTS), (NAME, COMPONENTS)]},
}
iterate.BLEND = []


def json_value(value):
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


class ViewPredictor(architecture.ArchitecturePredictor):
    """Same context/checkpoint serialization contract, with resolved-view diagnostics."""
    def validate(self):
        super().validate()
        if self.components not in [COMPONENTS, CONTROL_COMPONENTS]:
            raise ValueError('Only the frozen default8/default32 recipes are allowed.')

    def predict_proba(self, raw):
        component = self.components[0]
        spec = self.checkpoints['default']
        model = None
        started = time.monotonic()
        try:
            model = architecture.make_model(spec, self.seed, component['n_estimators'],
                                            self.categorical_indices, self.flags)
            model.fit(self.context_x, self.context_y)
            actual = {key: int(getattr(model.configs_[0], key)) for key in
                      ['features_per_group', 'emsize', 'nhead', 'nlayers']}
            if any(actual[key] != spec[key] for key in actual):
                raise RuntimeError('Loaded architecture differs from pinned metadata.')
            views = model.ensemble_configs_
            resolved = int(model.n_estimators_)
            if resolved != component['n_estimators'] or len(views) != resolved:
                raise RuntimeError('Resolved view count differs from the frozen budget.')
            diagnostics = dict(**component, actual_resolved_views=resolved,
                sha256=spec['sha256'], architecture=actual,
                fit_mode=model.fit_mode,
                ensemble_configs=json_value(views),
                fit_seconds=time.monotonic()-started,
                mps_driver_after_fit_bytes=torch.mps.driver_allocated_memory())
            prediction_started = time.monotonic()
            p = np.asarray(model.predict_proba(raw[self.columns]), dtype=np.float64)
            if (p.shape != (len(raw), 2) or not np.isfinite(p).all()
                or not ((p >= 0) & (p <= 1)).all()
                or not np.allclose(p.sum(axis=1), 1., atol=1e-6)):
                raise RuntimeError('Invalid probabilities; do not use a partial ensemble.')
            diagnostics.update(predict_seconds=time.monotonic()-prediction_started,
                seconds=time.monotonic()-started,
                mps_driver_after_predict_bytes=torch.mps.driver_allocated_memory(),
                process_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        finally:
            model = None
            architecture.release_memory()
        diagnostics['mps_driver_after_release_bytes'] = torch.mps.driver_allocated_memory()
        self.last_resources = [diagnostics]
        print('TabPFN views', json.dumps({k: v for k, v in diagnostics.items()
                                        if k != 'ensemble_configs'}), flush=True)
        return p


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in [CONTROL, NAME]:
        raise ValueError(name)
    x, cats = matrices[False]
    predictor = ViewPredictor(x.iloc[train_idx], y, cats, seed,
                              iterate.CONFIGS[name]['components'])
    p = predictor.predict_proba(x.iloc[predict_idx])[:, 1]
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT/f'{name}_seed{seed}_{key}.json').write_text(json.dumps(dict(
            training_rows=len(train_idx), query_rows=len(predict_idx),
            components=predictor.last_resources, views_prefix_stable=False), indent=2)+'\n')
    return p, predictor


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'views_policy' in protocol:
        return
    architecture.verify_checkpoint(architecture.CHECKPOINTS['default'])
    for filename in ['tabpfn_views.py', 'test_tabpfn_views.py', 'tabpfn_architecture.py',
                     architecture.MANIFEST]:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = architecture.environment_lock()
    (out/'requirements-views.lock.txt').write_text(lock)
    protocol['views_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['views_policy'] = POLICY
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # Includes source/environment freeze before the first v6 fit.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def check_environment(out):
    architecture.verify_checkpoint(architecture.CHECKPOINTS['default'])
    expected = json.loads((out/'protocol.json').read_text())['views_environment_sha256']
    if (hashlib.sha256(architecture.environment_lock().encode()).hexdigest() != expected
        or hashlib.sha256((out/'requirements-views.lock.txt').read_bytes()).hexdigest() != expected):
        raise RuntimeError('View experiment dependency environment changed.')


def probe():
    """Only run after exclusive MPS permission; does not calculate target accuracy."""
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    x, cats = matrices[False]
    predictor = ViewPredictor(x.iloc[dev[a]], y[a], cats, iterate.SEEDS[0], COMPONENTS)
    query = x.iloc[dev[b[:128]]]
    started = time.monotonic()
    p = predictor.predict_proba(query)
    first_resources = predictor.last_resources
    with tempfile.TemporaryDirectory(prefix='views-probe-') as folder:
        path = Path(folder)/'model.joblib'
        joblib.dump(predictor, path)
        restored = joblib.load(path)
        restored_p = restored.predict_proba(query)
        np.testing.assert_allclose(restored_p, p, atol=1e-6, rtol=0)
    result = dict(full_training_context_rows=len(a), query_rows=len(query),
        components=first_resources, valid_probabilities=True, accuracy_computed=False,
        serialization_max_absolute_difference=float(np.max(np.abs(p-restored_p))),
        seconds=time.monotonic()-started,
        process_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mps_driver_after_release_bytes=torch.mps.driver_allocated_memory(),
        mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_033')
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
    sys.modules['tabpfn_views'] = sys.modules[__name__]
    ViewPredictor.__module__ = 'tabpfn_views'
    main()
