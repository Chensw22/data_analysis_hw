"""Strict inner selection of constant versus regularized linear LightGBM leaves."""
import model_families
if __name__ == '__main__':
    model_families.ensure_openmp()

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import iterate
import contrast_features
import spatial_expert

# Importing the existing OpenMP helper also patches iterate.fit. Undo that
# dependency capture before using the original CatBoost controls.
BASE_FIT = model_families.ORIGINAL_FIT
spatial_expert.ORIGINAL_FIT = BASE_FIT
contrast_features.BASE_FIT = BASE_FIT
BASE_EVALUATE = iterate.evaluate
CONTROL = 'v55_constant_leaf_spatial'
NAME = 'v56_nested_linear_leaf'
PARAMS = dict(n_estimators=700, learning_rate=.035, min_child_samples=30,
              reg_lambda=5., colsample_bytree=.9, subsample=.85,
              subsample_freq=1, num_leaves=15, objective='binary',
              device_type='cpu', tree_learner='serial', n_jobs=4,
              verbosity=-1, force_col_wise=True, zero_as_missing=False)
OPTIONS = (dict(linear_tree=False), dict(linear_tree=True, linear_lambda=1.),
           dict(linear_tree=True, linear_lambda=10.), dict(linear_tree=True, linear_lambda=100.))
SPATIAL = dict(knots=5, C=.1, weight=.5)
STAGE_BUDGET_SECONDS = 1200
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    'v34_contrasts_spatial': dict(context=False, model='spatial', **SPATIAL),
    CONTROL: dict(context=False, model='constant_leaf_spatial', params=PARAMS, spatial=SPATIAL),
    NAME: dict(context=False, model='nested_linear_leaf', params=PARAMS,
               options=list(OPTIONS), inner_folds=3, inner_seed_offset=10000, spatial=SPATIAL),
}
iterate.BLEND = []
RUN_OUT = None
DEADLINE = None


class StageBudgetExceeded(TimeoutError):
    pass


def check_budget():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise StageBudgetExceeded('Whole-stage 1200-second budget exceeded; incomplete.')


class BudgetCallback:
    order = -100
    before_iteration = True

    def __call__(self, env):
        check_budget()


def alarm_handler(signum, frame):
    raise StageBudgetExceeded('Whole-stage 1200-second budget exceeded; incomplete.')


class TrainingPreprocessor:
    """Training-only scaling and categorical schema; preserve numerical NaNs."""
    def fit(self, x, cats):
        self.columns = list(x.columns)
        self.cats = list(cats)
        self.numeric = [col for col in self.columns if col not in self.cats]
        self.scaler = StandardScaler().fit(x[self.numeric])
        self.vocabularies = {col: sorted(x[col].dropna().astype(str).unique().tolist()) for col in self.cats}
        return self

    def transform(self, x):
        result = x[self.columns].copy()
        result[self.numeric] = self.scaler.transform(x[self.numeric])
        for col in self.cats:
            known = x[col].where(x[col].isin(self.vocabularies[col]))
            result[col] = pd.Categorical(known, categories=self.vocabularies[col])
        return result


def new_model(option, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**PARAMS, **option, random_state=seed)


def fit_model(x, y, option, seed):
    check_budget()
    model = new_model(option, seed)
    with threadpool_limits(limits=4):
        model.fit(x, y, callbacks=[BudgetCallback()])
    check_budget()
    return model


def fit_spatial(x, y, seed):
    check_budget()
    mask = spatial_expert.SpatialPredictor.eligible(x)
    specialist = x.loc[mask]
    spline = make_pipeline(SimpleImputer(strategy='median'),
                           SplineTransformer(n_knots=5, degree=3, include_bias=False))
    side = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    spline.fit(specialist[['CabinNumber']])
    side.fit(specialist[['Side']])
    expert = LogisticRegression(C=.1, max_iter=1500, random_state=seed)
    spatial = spatial_expert.SpatialPredictor(None, spline, side, expert, .5)
    expert.fit(spatial.design(specialist), np.asarray(y)[mask.to_numpy()])
    check_budget()
    return spatial


def spatial_predictions(spatial, x):
    mask = spatial.eligible(x).to_numpy()
    p = spatial.expert.predict_proba(spatial.design(x.loc[mask]))[:, 1] if mask.any() else np.empty(0)
    return mask, p


def combine_spatial(p, mask, expert):
    p = np.asarray(p).copy()
    p[mask] = .5 * p[mask] + .5 * expert
    return p


class LinearLeafPredictor:
    def __init__(self, prep, model, spatial, metadata):
        self.preprocessor = prep
        self.model = model
        self.spatial = spatial
        self.metadata = metadata

    def predict_proba(self, prepared):
        x = contrast_features.add_contrasts(prepared)
        p = self.model.predict_proba(self.preprocessor.transform(x))[:, 1]
        mask, expert = spatial_predictions(self.spatial, x)
        p = combine_spatial(p, mask, expert)
        return np.column_stack([1 - p, p])

    def predict_proba_raw(self, raw):
        prepared, _ = iterate.base.features(raw.drop(columns='Transported', errors='ignore'), 'engineered_no_ids')
        return self.predict_proba(prepared)


def select_option(x, y, cats, seed):
    """Only outer-training rows are accepted, including all preprocessing fits."""
    y = np.asarray(y)
    oof = np.full((len(y), len(OPTIONS)), np.nan)
    folds = []
    for inner, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed + 10000).split(x, y)):
        check_budget()
        prep = TrainingPreprocessor().fit(x.iloc[a], cats)
        xa, xb = prep.transform(x.iloc[a]), prep.transform(x.iloc[b])
        spatial = fit_spatial(x.iloc[a], y[a], seed + inner)
        mask, expert = spatial_predictions(spatial, x.iloc[b])
        for index, option in enumerate(OPTIONS):
            model = fit_model(xa, y[a], option, seed + inner)
            p = model.predict_proba(xb)[:, 1]
            oof[b, index] = combine_spatial(p, mask, expert)
            del model
        folds.append(dict(inner_fold=inner, train_rows=len(a), validation_rows=len(b),
                          train_index_sha256=hashlib.sha256(np.asarray(x.index[a], dtype=np.int64).tobytes()).hexdigest(),
                          validation_index_sha256=hashlib.sha256(np.asarray(x.index[b], dtype=np.int64).tobytes()).hexdigest()))
        print(f'Linear-leaf selector seed={seed} inner_fold={inner + 1}/3 complete', flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError('Incomplete or invalid inner OOF; no option can be selected.')
    table = []
    for index, option in enumerate(OPTIONS):
        correct = int(np.sum((oof[:, index] >= .5) == y))
        table.append(dict(configuration_index=index, option=option, correct_count=correct,
                          accuracy=correct / len(y), log_loss=float(log_loss(y, oof[:, index], labels=[0, 1]))))
    winner = min(table, key=lambda row: (-row['correct_count'], row['log_loss'], row['configuration_index']))
    check_budget()
    return dict(winner['option']), dict(inner_table=table, inner_folds=folds,
                                       selected_configuration_index=winner['configuration_index'],
                                       selected_option=winner['option'])


def fit(name, matrices, train_idx, y, predict_idx, seed):
    check_budget()
    if name == iterate.BASELINE:
        result = BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
        check_budget()
        return result
    raw, cats = matrices[False]
    x = contrast_features.add_contrasts(raw)
    if name == 'v34_contrasts_spatial':
        p, model = spatial_expert.fit_spatial(name, {False: (x, cats)}, train_idx, y, predict_idx, seed)
        check_budget()
        return p, contrast_features.ContrastPredictor(model)
    if name not in [CONTROL, NAME]:
        raise ValueError(f'Unknown linear-leaf candidate: {name}')
    started = time.monotonic()
    if len(y) != len(train_idx) or len(np.unique(train_idx)) != len(train_idx) or np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Unique training rows/targets and disjoint prediction rows are required.')
    training, target = x.iloc[train_idx], np.asarray(y)
    if name == NAME:
        option, metadata = select_option(training, target, cats, seed)
    else:
        option = dict(OPTIONS[0])
        metadata = dict(selected_option=option, matched_constant_control=True)
    # A fresh preprocessor, spline and learner use every outer-training row.
    prep = TrainingPreprocessor().fit(training, cats)
    spatial = fit_spatial(training, target, seed)
    model = fit_model(prep.transform(training), target, option, seed)
    predictor = LinearLeafPredictor(prep, model, spatial, metadata)
    p = predictor.predict_proba(raw.iloc[predict_idx])[:, 1]
    check_budget()
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError('Invalid outer prediction; refusing candidate result.')
    metadata.update(seed=int(seed), outer_train_rows=len(target), prediction_rows=len(predict_idx),
                    elapsed_seconds=time.monotonic() - started, fixed_tree_budget=PARAMS['n_estimators'])
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT / f'{name}_seed{seed}_{key}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return p, predictor


def environment_lock():
    return ''.join(f'{name}=={version}\n' for name, version in sorted(
        (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()))


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'linear_leaf_policy' in protocol:
        return
    for filename in ['linear_leaf_candidate.py', 'contrast_features.py', 'spatial_expert.py', 'model_families.py', 'requirements-models.txt']:
        source = iterate.ROOT / filename
        shutil.copy2(source, out / filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = environment_lock()
    (out / 'requirements-linear-leaf.lock.txt').write_text(lock)
    protocol['linear_leaf_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['linear_leaf_policy'] = (
        'Same v34 input features and original Earth/CryoSleep spatial expert. Fixed run005 LGB15 '
        'parameters throughout; v55 is the matched scaled/categorical-schema constant-leaf control. '
        'v56 uses fresh inner3 OOF to select constant, linear_lambda1,10,100 in that frozen tie order '
        'by pooled correct count, then Logloss, then configuration index. StandardScaler, category '
        'vocabulary and spatial spline fit only the respective inner training rows; the spline '
        'may be shared across the four options only within that same split. Numerical NaNs stay NaN; '
        'unknown categories become missing. A fresh complete outer-training pipeline refits the chosen '
        'option for exactly700 rounds. No outer-label early stopping, threshold or blend search. '
        '1200-second whole-stage deadline; incomplete search fails closed without candidate scores. '
        'Existing gate compares v6, so v44 stability checks remain necessary before online approval.')
    protocol['linear_leaf_options'] = list(OPTIONS)
    protocol['linear_leaf_stage_budget_seconds'] = STAGE_BUDGET_SECONDS
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def check_environment(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    expected = protocol['linear_leaf_environment_sha256']
    if hashlib.sha256(environment_lock().encode()).hexdigest() != expected:
        raise RuntimeError('Environment changed since frozen screening.')
    if hashlib.sha256((out / 'requirements-linear-leaf.lock.txt').read_bytes()).hexdigest() != expected:
        raise RuntimeError('Saved environment snapshot changed since screening.')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    check_budget()
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def probe():
    train, _, matrices, dev = iterate.dataset()
    raw, cats = matrices[False]
    rng = np.random.default_rng(iterate.SEEDS[0])
    chosen = rng.choice(dev, size=512, replace=False)
    a, b = chosen[:384], chosen[384:]
    x = contrast_features.add_contrasts(raw)
    target = train.Transported.iloc[a].astype(int).to_numpy()
    prep = TrainingPreprocessor().fit(x.iloc[a], cats)
    xa = prep.transform(x.iloc[a])
    spatial = fit_spatial(x.iloc[a], target, iterate.SEEDS[0])
    rows = []
    for option in OPTIONS:
        started = time.monotonic()
        model = fit_model(xa, target, option, iterate.SEEDS[0])
        predictor = LinearLeafPredictor(prep, model, spatial, {'option': option})
        p = predictor.predict_proba(raw.iloc[b])
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        with tempfile.TemporaryDirectory(prefix='linear-leaf-probe-') as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(predictor, path)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(raw.iloc[b]), p)
        dump = model.booster_.dump_model()
        def count_linear(node):
            if 'leaf_index' in node or 'leaf_value' in node:
                return int(bool(node.get('leaf_features', [])))
            return count_linear(node['left_child']) + count_linear(node['right_child'])
        linear_leaves = sum(count_linear(tree['tree_structure']) for tree in dump['tree_info'])
        if option['linear_tree'] and linear_leaves == 0:
            raise RuntimeError('Probe requested linear leaves but none were fitted.')
        rows.append(dict(option=option, seconds=time.monotonic() - started,
                         fitted_linear_leaves=linear_leaves, serialization_exact=True))
    result = dict(training_rows=len(a), prediction_rows=len(b), options=rows,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  accuracy_computed=False, note='Small resource/API probe only, not cross-validation.')
    print(json.dumps(result), flush=True)
    return result


def run_stage(stage, out):
    global RUN_OUT, DEADLINE
    RUN_OUT = None if stage == 'probe' else out
    if stage in ['confirm', 'export']:
        check_environment(out)
        if json.loads((out / 'screen_status.json').read_text())['status'] != 'completed':
            raise RuntimeError('Screening did not complete; partial results cannot be used.')
    started = time.monotonic()
    DEADLINE = started + STAGE_BUDGET_SECONDS
    guarded = {'screen': ['selection.json', 'screen_results.csv'], 'confirm': ['gate.json'],
               'export': ['submission.csv', 'test_probabilities.csv'], 'probe': []}[stage]
    existing = {name for name in guarded if (out / name).exists()}
    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, STAGE_BUDGET_SECONDS)
    status = 'failed'
    try:
        if stage == 'probe':
            probe()
        else:
            {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[stage](out)
        check_budget()
        status = 'completed'
    except BaseException as error:
        timeout = isinstance(error, StageBudgetExceeded) or time.monotonic() >= DEADLINE
        status = 'incomplete_timeout' if timeout else 'failed'
        # Even a non-timeout failure must not leave a partial search winner.
        for name in guarded:
            if name not in existing:
                (out / name).unlink(missing_ok=True)
        if timeout and not isinstance(error, StageBudgetExceeded):
            raise StageBudgetExceeded('Deadline reached inside native learner; incomplete.') from error
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        DEADLINE = None
        if stage != 'probe' and out.exists():
            (out / f'{stage}_status.json').write_text(json.dumps(dict(
                stage=stage, status=status, elapsed_seconds=time.monotonic() - started,
                budget_seconds=STAGE_BUDGET_SECONDS, partial_results_eligible=False), indent=2) + '\n')


iterate.fit = fit
iterate.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_027')
    args = parser.parse_args()
    run_stage(args.stage, iterate.ROOT / 'artifacts' / args.run)


if __name__ == '__main__':
    sys.modules['linear_leaf_candidate'] = sys.modules[__name__]
    for cls in [TrainingPreprocessor, LinearLeafPredictor]:
        cls.__module__ = 'linear_leaf_candidate'
    main()
