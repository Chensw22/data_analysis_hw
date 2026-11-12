"""One nested learning procedure: choose CatBoost structure/budget inside training only."""
import argparse
import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from threadpoolctl import threadpool_limits

import iterate
import spatial_expert
from contrast_features import add_contrasts

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
NAME = 'v46_nested_structure_search'
CHECKPOINTS = (350, 700, 1400)
STRUCTURES = (
    dict(grow_policy='SymmetricTree', depth=6),
    dict(grow_policy='Depthwise', depth=6, min_data_in_leaf=30),
    dict(grow_policy='Lossguide', depth=8, max_leaves=31, min_data_in_leaf=30),
)
PATHS = tuple({**structure, 'l2_leaf_reg': l2}
              for structure in STRUCTURES for l2 in (5, 15))
CANDIDATES = tuple({**path, 'iterations': iterations}
                   for path in PATHS for iterations in CHECKPOINTS)
STAGE_BUDGET_SECONDS = 1200
SETTINGS = dict(inner_folds=3, inner_seed_offset=10000, learning_rate=.045,
                paths=list(PATHS), checkpoints=list(CHECKPOINTS), threshold=.5,
                spatial=dict(knots=5, C=.1, weight=.5),
                selection_order=['pooled_correct_count_desc', 'pooled_log_loss_asc', 'configuration_index_asc'],
                stage_budget_seconds=STAGE_BUDGET_SECONDS)
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    NAME: dict(context=False, model='nested_structure_search', **SETTINGS),
}
iterate.BLEND = []
RUN_OUT = None
DEADLINE = None


class StageBudgetExceeded(TimeoutError):
    pass


def check_budget():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise StageBudgetExceeded('The whole stage exceeded its 1200-second budget; incomplete.')


class BudgetCallback:
    def after_iteration(self, info):
        # Stop the C++ learner at an iteration boundary, then raise in Python.
        return DEADLINE is None or time.monotonic() < DEADLINE


def alarm_handler(signum, frame):
    raise StageBudgetExceeded('The whole stage exceeded its 1200-second budget; incomplete.')


def new_cat(cats, seed, config):
    return CatBoostClassifier(**{**iterate.CAT, **config}, cat_features=cats,
                              loss_function='Logloss', random_seed=seed,
                              thread_count=4, verbose=False, allow_writing_files=False)


def train_cat(x, y, cats, seed, config):
    check_budget()
    model = new_cat(cats, seed, config)
    with threadpool_limits(limits=4):
        model.fit(x, y, callbacks=[BudgetCallback()])
    check_budget()
    return model


def fit_spatial(x, y, seed):
    """Fit a separate spline inside precisely the supplied training rows."""
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


def spatial_values(spatial, x):
    mask = spatial.eligible(x).to_numpy()
    values = spatial.expert.predict_proba(spatial.design(x.loc[mask]))[:, 1] if mask.any() else np.empty(0)
    return mask, values


def combine_spatial(probabilities, mask, values):
    result = np.asarray(probabilities).copy()
    result[mask] = .5 * result[mask] + .5 * values
    return result


def select_configuration(x, y, cats, seed):
    """No outer validation data or labels enter this function."""
    y = np.asarray(y)
    oof = np.full((len(y), len(CANDIDATES)), np.nan)
    inner_audits = []
    splits = StratifiedKFold(3, shuffle=True, random_state=seed + 10000)
    for inner, (a, b) in enumerate(splits.split(x, y)):
        check_budget()
        assert not set(a) & set(b)
        spatial = fit_spatial(x.iloc[a], y[a], seed + inner)
        mask, values = spatial_values(spatial, x.iloc[b])
        for path_number, path in enumerate(PATHS):
            model = train_cat(x.iloc[a], y[a], cats, seed + inner,
                              {**path, 'iterations': max(CHECKPOINTS)})
            for checkpoint_number, checkpoint in enumerate(CHECKPOINTS):
                check_budget()
                p = model.predict_proba(x.iloc[b], ntree_end=checkpoint)[:, 1]
                oof[b, path_number * len(CHECKPOINTS) + checkpoint_number] = combine_spatial(p, mask, values)
            del model
        inner_audits.append(dict(inner_fold=inner, train_rows=len(a), validation_rows=len(b),
                                 train_index_sha256=hashlib.sha256(np.asarray(x.index[a], dtype=np.int64).tobytes()).hexdigest(),
                                 validation_index_sha256=hashlib.sha256(np.asarray(x.index[b], dtype=np.int64).tobytes()).hexdigest()))
        print(f'Nested selector seed={seed} inner_fold={inner + 1}/3 complete', flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError('Incomplete inner OOF; refusing configuration selection.')
    table = []
    for index, config in enumerate(CANDIDATES):
        correct = int(np.sum((oof[:, index] >= .5) == y))
        table.append(dict(configuration_index=index, config=config, correct_count=correct,
                          accuracy=correct / len(y), log_loss=float(log_loss(y, oof[:, index], labels=[0, 1]))))
    winner = min(table, key=lambda row: (-row['correct_count'], row['log_loss'], row['configuration_index']))
    check_budget()
    return dict(winner['config']), dict(inner_table=table, selected_configuration_index=winner['configuration_index'],
                                      selected_config=winner['config'], inner_folds=inner_audits)


class NestedStructurePredictor:
    def __init__(self, spatial_model, metadata):
        self.spatial_model = spatial_model
        self.metadata = metadata

    def predict_proba(self, raw):
        return self.spatial_model.predict_proba(add_contrasts(raw))


def fit(name, matrices, train_idx, y, predict_idx, seed):
    check_budget()
    if name == iterate.BASELINE:
        result = BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
        check_budget()
        return result
    if name != NAME:
        raise ValueError(f'Unknown candidate: {name}')
    started = time.monotonic()
    train_idx, predict_idx = np.asarray(train_idx), np.asarray(predict_idx)
    target = np.asarray(y)
    if len(target) != len(train_idx) or len(np.unique(train_idx)) != len(train_idx):
        raise ValueError('Targets must match unique outer training rows.')
    if np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Outer train and prediction rows must be disjoint.')
    raw, cats = matrices[False]
    x = add_contrasts(raw)
    training = x.iloc[train_idx]
    config, metadata = select_configuration(training, target, cats, seed)
    # Both learners are fresh fits on all outer training rows. The chosen tree
    # count is baked into the saved model, so export cannot use extra trees.
    spatial = fit_spatial(training, target, seed)
    spatial.base_model = train_cat(training, target, cats, seed, config)
    metadata.update(seed=int(seed), outer_train_rows=len(target), prediction_rows=len(predict_idx),
                    refit_tree_count=int(config['iterations']))
    predictor = NestedStructurePredictor(spatial, metadata)
    p = predictor.predict_proba(raw.iloc[predict_idx])[:, 1]
    check_budget()
    metadata['elapsed_seconds'] = time.monotonic() - started
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT / f'nested_training_seed{seed}_{key}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return p, predictor


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'nested_structure_policy' in protocol:
        return
    for filename in ['nested_structure_search.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT / filename
        shutil.copy2(source, out / filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['nested_structure_policy'] = (
        'Fixed outer3 folds unchanged. Within each outer training set, inner3-fold OOF selects '
        'one of 18 complete v34 spatial pipelines using pooled accuracy, then Logloss, then frozen '
        'configuration index. Six Cat paths train to1400; ntree_end350/700/1400 reuses checkpoints. '
        'Every inner split fits its own training-only spatial expert, shared only across that split\'s '
        'six paths. Fresh complete outer-training Cat and spline refits use chosen exact tree count. '
        'No outer-validation labels for selection. Threshold .5 throughout. The 1200s budget covers '
        'the entire invoked stage, including baseline, all outer folds and confirmations. Timeout '
        'marks incomplete and cannot create selection, gate, or export outputs for that stage. '
        'Existing iterate gate compares original v6; any >83% candidate still requires v44 stability '
        'comparison before online approval.')
    protocol['nested_search_settings'] = SETTINGS
    protocol['nested_configurations_in_tie_order'] = list(CANDIDATES)
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)  # Freeze dependencies before even the baseline fit.
    check_budget()
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def run_stage(stage, out):
    global RUN_OUT, DEADLINE
    RUN_OUT = out
    started = time.monotonic()
    DEADLINE = started + STAGE_BUDGET_SECONDS
    guarded_outputs = ['selection.json'] if stage == 'screen' else (
        ['gate.json'] if stage == 'confirm' else ['submission.csv', 'test_probabilities.csv'])
    existing = {name for name in guarded_outputs if (out / name).exists()}
    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, STAGE_BUDGET_SECONDS)
    status = 'failed'
    try:
        {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[stage](out)
        check_budget()
        status = 'completed'
    except BaseException as error:
        # CatBoost can wrap an exception raised by a Python callback/signal in
        # CatBoostError. The stage deadline still determines incompleteness.
        if isinstance(error, StageBudgetExceeded) or time.monotonic() >= DEADLINE:
            status = 'incomplete_timeout'
            for name in guarded_outputs:
                if name not in existing:
                    (out / name).unlink(missing_ok=True)
            if isinstance(error, StageBudgetExceeded):
                raise
            raise StageBudgetExceeded('Stage deadline reached inside the learner; incomplete.') from error
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        DEADLINE = None
        if out.exists():
            (out / f'{stage}_status.json').write_text(json.dumps(dict(
                stage=stage, status=status, elapsed_seconds=time.monotonic() - started,
                budget_seconds=STAGE_BUDGET_SECONDS, partial_results_eligible=False), indent=2) + '\n')


iterate.fit = fit
iterate.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_020')
    args = parser.parse_args()
    run_stage(args.stage, iterate.ROOT / 'artifacts' / args.run)


if __name__ == '__main__':
    sys.modules['nested_structure_search'] = sys.modules[__name__]
    NestedStructurePredictor.__module__ = 'nested_structure_search'
    main()
