"""Fixed CPU multilayer bagging; all AutoGluon fitting stays inside outertrain."""
import os
import sys
import importlib.util
from pathlib import Path


def ensure_openmp():
    """Reuse this environment's sklearn libomp; no global environment edits."""
    if sys.platform == 'darwin':
        spec = importlib.util.find_spec('sklearn')
        library = Path(spec.origin).parent/'.dylibs'
        if (library/'libomp.dylib').exists() and str(library) not in os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '').split(':'):
            env = {**os.environ, 'DYLD_FALLBACK_LIBRARY_PATH': str(library)+':'+os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')}
            os.execve(sys.executable, [sys.executable, *sys.argv], env)


if __name__ == '__main__':
    ensure_openmp()

import argparse
import copy
import hashlib
import importlib.metadata
import json
import resource
import shutil
import subprocess
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
pd.options.future.infer_string = True  # pandas3-compatible frozen feature-builder semantics.
from autogluon.tabular import TabularPredictor
from threadpoolctl import threadpool_limits

import iterate

# Capture the original baseline before any monkeypatch or CONFIGS replacement.
BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
ROOT = Path(__file__).resolve().parent
NAME = 'v73_AutoGluon_multilayer'
LABEL = '__target__'
PROTOCOL = 'reports/automl_method_review.json'
HYPERPARAMETERS = {
    'CAT': dict(iterations=700, depth=6, learning_rate=.045, l2_leaf_reg=5,
                **{'ag.early_stop': 50}, ag_args_fit={'stopping_metric': 'log_loss'}),
    'GBM': dict(num_boost_round=500, num_leaves=31, learning_rate=.05,
                **{'ag.early_stop': 50}, ag_args_fit={'stopping_metric': 'log_loss'}),
    'RF': dict(n_estimators=300, criterion='gini', n_jobs=4),
    'XT': dict(n_estimators=300, criterion='gini', n_jobs=4),
}
ENSEMBLE_ARGS = dict(use_child_oof=False, use_orig_features=True,
    fold_fitting_strategy='sequential_local', save_bag_folds=True,
    max_base_models=0, max_base_models_per_type=0)
SETTINGS = dict(num_bag_folds=3, num_bag_sets=1, num_stack_levels=1,
    num_cpus=4, num_gpus=0, eval_metric='accuracy', threshold=.5,
    dynamic_stacking=False, hyperparameter_tune_kwargs=None,
    calibrate=False, calibrate_decision_threshold=False,
    fit_weighted_ensemble=True, fit_full_last_level_weighted_ensemble=True, fit_strategy='sequential',
    use_bag_holdout=False, time_limit=None, refit_full=False,
    input='original_engineered_no_ids_30_columns',
    inner_validation='stock bag fold early stopping; no extra holdout',
    stage_budget_seconds={'probe': 600, 'screen': 3600, 'confirm': 10800, 'export': 3600})
FAMILIES = {'CatBoostModel': 'CAT', 'LGBModel': 'GBM', 'RFModel': 'RF', 'XTModel': 'XT'}
RUN_OUT = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, tuple, set)):
        return list(value)
    return str(value)


def validate_features(x):
    forbidden = {'PassengerId', 'Name', 'GroupNumber', 'GroupMember', 'Transported', LABEL}
    if forbidden & set(x.columns) or len(x.columns) != 30 or not x.columns.is_unique:
        raise ValueError('Expected exactly30 existing engineered_no_ids columns without identifiers/targets.')


def fit_kwargs():
    return dict(hyperparameters=copy.deepcopy(HYPERPARAMETERS),
        num_bag_folds=3, num_bag_sets=1, num_stack_levels=1,
        ag_args_ensemble=copy.deepcopy(ENSEMBLE_ARGS),
        num_cpus=4, num_gpus=0, fit_strategy='sequential',
        dynamic_stacking=False, hyperparameter_tune_kwargs=None,
        calibrate=False, calibrate_decision_threshold=False,
        fit_weighted_ensemble=True, fit_full_last_level_weighted_ensemble=True, use_bag_holdout=False,
        time_limit=None, refit_full=False, save_bag_folds=True)


def assert_coverage(predictor):
    """Reject silently skipped families, OOB shortcuts or incomplete bag folds."""
    leaderboard = predictor.leaderboard(silent=True)
    levels = dict(zip(leaderboard['model'], leaderboard['stack_level']))
    info = predictor.info()
    l1_names = {name for name, details in info['model_info'].items()
                if levels[name] == 1 and details.get('bagged_info',{}).get('child_model_type') in FAMILIES}
    coverage = []
    counts = {(family, level): 0 for family in HYPERPARAMETERS for level in [1, 2]}
    for name, details in info['model_info'].items():
        bag = details.get('bagged_info', {})
        family = FAMILIES.get(bag.get('child_model_type'))
        if family is None:
            continue  # Official weighted ensemble is additional to these24 children.
        level = int(levels[name])
        if level not in [1, 2] or bag['num_child_models'] != 3 or not bag['bagged_mode']:
            raise RuntimeError(f'Incomplete3fold bag: {name}')
        if bag['_k_per_n_repeat'] != [3] or bag['_n_repeats'] != 1:
            raise RuntimeError(f'Unexpected bag repeats/folds: {name}')
        if not details.get('can_infer', True):
            raise RuntimeError(f'Model not inferable: {name}')
        params = details.get('hyperparameters', {})
        if params.get('use_child_oof') is not False or params.get('use_orig_features') is not True:
            raise RuntimeError(f'Unexpected OOB shortcut or original-feature exclusion: {name}')
        bases = details.get('stacker_info', {}).get('base_model_names', [])
        features = details['features']
        original_features = []
        if level == 2:
            original_features = sorted(set(features) & set(predictor.feature_metadata.get_features()))
            if set(bases) != l1_names or len(bases) != 4 or not set(bases).issubset(features) or not original_features:
                raise RuntimeError(f'L2 must combine all four L1 predictions and original features: {name}')
        children = details['children_info']
        if len(children) != 3:
            raise RuntimeError(f'Missing resolved child metadata: {name}')
        for child_name, child in children.items():
            trained = child.get('hyperparameters_fit', {})
            if family in ['RF','XT']:
                if trained.get('n_estimators') != 300 or child['hyperparameters'].get('n_jobs') != 4:
                    raise RuntimeError(f'Forest size or CPU budget changed: {name}/{child_name}')
            elif (bag['child_ag_args_fit'].get('early_stop') != 50 or child.get('stopping_metric') != 'log_loss'):
                raise RuntimeError(f'Boosting stopping recipe changed: {name}/{child_name}')
        counts[(family, level)] += 1
        coverage.append(dict(model=name, family=family, layer=level, children=3,
            base_models=bases, features=features, retained_original_features=original_features,
            resolved_hyperparameters=bag['child_hyperparameters'],
            child_trained_parameters=bag['child_hyperparameters_fit'],
            child_fit_args=bag['child_ag_args_fit']))
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(f'Incomplete or unexpected family/layer coverage: {counts}')
    return coverage, info


class AutoGluonArtifact:
    """On-disk official predictor plus explicit input schema and audit metadata."""
    def __init__(self, path, columns, metadata):
        self.path = str(Path(path).resolve())
        self.columns = list(columns)
        self.metadata = metadata
        self._predictor = None

    def predictor(self):
        if self._predictor is None:
            self._predictor = TabularPredictor.load(self.path)
        return self._predictor

    def predict_proba(self, x):
        validate_features(x)
        values = self.predictor().predict_proba(x[self.columns], as_pandas=True)
        if set(values.columns) != {0, 1}:
            raise ValueError('Unexpected target class mapping.')
        result = values[[0, 1]].to_numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite AutoGluon probabilities.')
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_predictor'] = None
        return state

    def relocate(self, new_predictor_path):
        """Explicit path rebinding after moving the complete official predictor directory."""
        self.path = str(Path(new_predictor_path).resolve())
        self._predictor = None


def fit_outer(x, y, ids, path, seed):
    validate_features(x)
    y, ids = np.asarray(y), np.asarray(ids, dtype=np.int64)
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or len(ids) != len(x) or len(np.unique(ids)) != len(ids):
        raise ValueError('Aligned unique training IDs and binary targets required.')
    path = Path(path)
    if path.exists():
        raise RuntimeError('Predictor directory already exists; do not resume partial fits.')
    train = x.copy()
    train.index = pd.Index(ids, name='outer_training_row_id')
    train[LABEL] = y.astype(int)
    started = time.monotonic()
    predictor = TabularPredictor(label=LABEL, problem_type='binary', eval_metric='accuracy',
        path=str(path), verbosity=2, learner_kwargs={'random_state': int(seed)},
        log_to_file=True)
    # Sole data argument contains outer-training rows. No tuning_data, pseudo
    # labels, validation feature rows or external target array enters this call.
    with threadpool_limits(limits=4):
        predictor.fit(train_data=train, **fit_kwargs())
    coverage, info = assert_coverage(predictor)
    saved_x, saved_y = predictor.load_data_internal(data='train')
    if not np.array_equal(saved_x.index.to_numpy(), ids) or not np.array_equal(saved_y.to_numpy(), y):
        raise RuntimeError('AutoGluon internal training membership differs from outertrain.')
    predictor.save()
    metadata = dict(status='complete', seed=int(seed), rows=len(y),
        train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        input_columns=list(x.columns), seconds=time.monotonic()-started,
        completed_bag_children=24, coverage=coverage, best_model=predictor.model_best,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        target_policy='Only outertrain targets used; stock innerfold targets may govern early stopping.',
        source_of_selection='AutoGluon internal bagOOF, not independent external accuracy')
    metadata['artifact_contract'] = 'Keep the complete predictors directory with wrapper.joblib; wrapper alone is not self-contained. After moving, call artifact.relocate(new_predictor_directory).'
    (path/'outer_audit.json').write_text(json.dumps(metadata, indent=2, default=json_default)+'\n')
    (path/'model_info.json').write_text(json.dumps(info, indent=2, default=json_default)+'\n')
    artifact = AutoGluonArtifact(path, x.columns, metadata)
    artifact._predictor = predictor
    return artifact


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name != NAME:
        raise ValueError(name)
    train_idx, predict_idx = np.asarray(train_idx), np.asarray(predict_idx)
    if np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Outer training/prediction rows must be disjoint.')
    if RUN_OUT is None:
        raise RuntimeError('A persistent run directory is required for bagged predictors.')
    key = hashlib.sha256(train_idx.astype(np.int64).tobytes()).hexdigest()[:12]
    path = RUN_OUT/'predictors'/f'seed{seed}_{key}'
    x = matrices[False][0]
    artifact = fit_outer(x.iloc[train_idx], y, train_idx, path, seed)
    probabilities = artifact.predict_proba(x.iloc[predict_idx])[:, 1]
    artifact._predictor = None
    return probabilities, artifact


def environment_lock():
    return ''.join(f'{name}=={version}\n' for name, version in sorted(
        (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()))


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'autogluon_policy' in protocol:
        return
    for name in ['autogluon_candidate.py', 'test_autogluon_candidate.py',
                 'requirements-autogluon.lock.txt', PROTOCOL]:
        protocol['sources'][name] = digest(ROOT/name)
        shutil.copy2(ROOT/name, out/Path(name).name)
    lock = environment_lock()
    (out/'autogluon_environment.lock.txt').write_text(lock)
    protocol.update(autogluon_policy=dict(settings=SETTINGS, hyperparameters=HYPERPARAMETERS,
        ensemble_args=ENSEMBLE_ARGS, version=importlib.metadata.version('autogluon.tabular')),
        autogluon_environment_sha256=hashlib.sha256(lock.encode()).hexdigest())
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # First call is baseline: freeze before the first fit.
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def configure():
    if importlib.metadata.version('autogluon.tabular') != '1.5.0':
        raise RuntimeError('Pinned AutoGluon1.5.0 required.')
    iterate.CONFIGS = {iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
        NAME: dict(context=False, model='autogluon', settings=SETTINGS,
                   hyperparameters=HYPERPARAMETERS, ensemble_args=ENSEMBLE_ARGS)}
    iterate.BLEND = []
    iterate.fit, iterate.evaluate = fit, evaluate


def synthetic_features(n=180, seed=7):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(dict(PassengerId=[f'{i+1:04d}_01' for i in range(n)],
        Name=[f'Person{i} Surname' for i in range(n)], HomePlanet=rng.choice(['Earth','Europa','Mars'], n),
        CryoSleep=rng.choice([True,False], n), Destination=rng.choice(['A','B','C'], n),
        Age=rng.uniform(1,80,n), VIP=rng.choice([True,False], n),
        Cabin=[f'{rng.choice(list("ABCD"))}/{i}/{rng.choice(["P","S"])}' for i in range(n)]))
    for col in iterate.base.SPEND:
        raw[col] = rng.exponential(200, n)
        raw.loc[rng.choice(n, max(1,n//12), replace=False), col] = np.nan
    return iterate.build_features(raw, False)[0]


def probe(out):
    if out.exists():
        raise RuntimeError('Probe output exists.')
    out.mkdir(parents=True)
    for name in ['autogluon_candidate.py', 'test_autogluon_candidate.py',
                 'requirements-autogluon.lock.txt', PROTOCOL]:
        shutil.copy2(ROOT/name, out/Path(name).name)
    (out/'source_hashes.json').write_text(json.dumps({name:digest(ROOT/name) for name in
        ['autogluon_candidate.py','test_autogluon_candidate.py','requirements-autogluon.lock.txt',PROTOCOL]}, indent=2)+'\n')
    x = synthetic_features(180)
    y = ((x['Age'] > 35) ^ x['CryoSleep'].eq('True')).to_numpy(dtype=int)
    arbitrary_ids = np.arange(150)*3+101
    artifact = fit_outer(x.iloc[:150], y[:150], arbitrary_ids, out/'predictor', 17)
    query = x.iloc[150:]
    expected = artifact.predict_proba(query)
    joblib.dump(artifact, out/'wrapper.joblib')
    restored = joblib.load(out/'wrapper.joblib')
    actual = restored.predict_proba(query)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    result = dict(status='complete', synthetic_only=True, external_accuracy_computed=False,
        internal_synthetic_oof_accuracy_used=True,
        rows=150, input_columns=len(x.columns), query_rows=len(query),
        training_ids='nonconsecutive_101_plus_3_times_row',
        training_seconds=artifact.metadata['seconds'], max_rss_bytes=artifact.metadata['max_rss_bytes'],
        completed_children=artifact.metadata['completed_bag_children'],
        reload_max_abs_difference=float(np.max(np.abs(actual-expected))),
        coverage=artifact.metadata['coverage'])
    (out/'probe_result.json').write_text(json.dumps(result, indent=2, default=json_default)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='coverage'}), flush=True)


def worker(args):
    global RUN_OUT
    configure()
    RUN_OUT = args.output if args.stage == 'probe' else ROOT/'artifacts'/args.run
    if args.stage == 'probe':
        probe(RUN_OUT)
        return
    if args.stage in ['confirm','export']:
        iterate.assert_frozen(RUN_OUT)
        expected = json.loads((RUN_OUT/'protocol.json').read_text())['autogluon_environment_sha256']
        if hashlib.sha256(environment_lock().encode()).hexdigest() != expected or digest(RUN_OUT/'autogluon_environment.lock.txt') != expected:
            raise RuntimeError('Frozen AutoGluon environment changed.')
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](RUN_OUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['probe','screen','confirm','export'])
    parser.add_argument('--run', default='run_039')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == 'probe' and args.output is None:
        parser.error('Probe requires a new --output directory.')
    if args.worker:
        worker(args)
        return
    out = args.output if args.stage == 'probe' else ROOT/'artifacts'/args.run
    # Parent process enforces wall-clock deadline even inside native learners.
    # sequential_local forbids Ray/fold subprocesses; killing the worker cannot
    # leave a distributed partial training job running.
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'])
    try:
        code = process.wait(timeout=SETTINGS['stage_budget_seconds'][args.stage])
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        out.mkdir(parents=True, exist_ok=True)
        (out/f'{args.stage}_failure.json').write_text(json.dumps(dict(
            status='failed_inconclusive', reason='hard_stage_timeout', completed_candidate=False), indent=2)+'\n')
        raise SystemExit(124)
    if code:
        if out.exists():
            (out/f'{args.stage}_failure.json').write_text(json.dumps(dict(
                status='failed_inconclusive', reason='worker_failed', exit_code=code), indent=2)+'\n')
        raise SystemExit(code)


if __name__ == '__main__':
    sys.modules['autogluon_candidate'] = sys.modules[__name__]
    AutoGluonArtifact.__module__ = 'autogluon_candidate'
    main()
