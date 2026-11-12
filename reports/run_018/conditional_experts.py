"""Frozen v34 global model plus probability-shrunk CryoSleep experts."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import iterate
import spatial_expert
from contrast_features import add_contrasts

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
NAME = 'v45_cryo_experts'
WEIGHT = .5
GLOBAL_SPATIAL = dict(knots=5, C=.1, weight=.5)
EXPERT_POLICY = dict(max_iterations=700, early_stopping_rounds=50,
                     inner_validation_fraction=.2, metric='Logloss',
                     refit='fresh_full_state_selected_iteration_count')
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    NAME: dict(context=False, model='conditional_experts', weight=WEIGHT,
               global_spatial=GLOBAL_SPATIAL, expert_cat=iterate.CAT,
               expert_policy=EXPERT_POLICY),
}
iterate.BLEND = []
RUN_OUT = None


def state_mask(x, state):
    """Only explicit True/False values route; missing or unfamiliar values do not."""
    return x.CryoSleep.astype('string').eq(str(state)).fillna(False).to_numpy(dtype=bool)


def fit_global(matrices, train_idx, y, predict_idx, seed):
    """Reproduce v34 without depending on another wrapper's mutable CONFIGS."""
    raw, cats = matrices[False]
    x = add_contrasts(raw)
    changed = {False: (x, cats)}
    _, cat_model = BASE_FIT(iterate.BASELINE, changed, train_idx, y, predict_idx, seed)
    training = x.iloc[train_idx]
    mask = spatial_expert.SpatialPredictor.eligible(training)
    specialist = training.loc[mask]
    spline = make_pipeline(
        SimpleImputer(strategy='median'),
        SplineTransformer(n_knots=GLOBAL_SPATIAL['knots'], degree=3, include_bias=False))
    side = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    spline.fit(specialist[['CabinNumber']])
    side.fit(specialist[['Side']])
    expert = LogisticRegression(C=GLOBAL_SPATIAL['C'], max_iter=1500, random_state=seed)
    global_model = spatial_expert.SpatialPredictor(
        cat_model, spline, side, expert, GLOBAL_SPATIAL['weight'])
    expert.fit(global_model.design(specialist), np.asarray(y)[mask.to_numpy()])
    return global_model, x, cats


def new_cat(cats, seed, iterations):
    params = {**iterate.CAT, 'iterations': int(iterations)}
    return CatBoostClassifier(**params, cat_features=cats, loss_function='Logloss',
                              eval_metric='Logloss', random_seed=seed,
                              thread_count=4, verbose=False, allow_writing_files=False)


def fit_expert(x, y, cats, seed, state):
    mask = state_mask(x, state)
    state_x = x.loc[mask]
    state_y = np.asarray(y)[mask]
    values, counts = np.unique(state_y, return_counts=True)
    metadata = dict(state=str(state), state_train_rows=len(state_y),
                    class_counts={str(int(k)): int(v) for k, v in zip(values, counts)})
    # Actual development folds have hundreds of examples from both classes.
    # Tiny or one-class inputs cannot support stratified binary early stopping.
    if len(state_y) < 10 or len(values) < 2 or counts.min() < 2:
        metadata.update(status='global_only_insufficient_class_samples', selected_iterations=None)
        return None, metadata
    a, b = train_test_split(np.arange(len(state_y)),
                            test_size=EXPERT_POLICY['inner_validation_fraction'],
                            stratify=state_y, random_state=seed)
    selector = new_cat(cats, seed, EXPERT_POLICY['max_iterations'])
    selector.fit(state_x.iloc[a], state_y[a], eval_set=(state_x.iloc[b], state_y[b]),
                 early_stopping_rounds=EXPERT_POLICY['early_stopping_rounds'], use_best_model=True)
    selected = int(selector.get_best_iteration()) + 1
    if selected < 1:
        raise RuntimeError('Expert early stopping did not produce a valid best iteration.')
    metadata.update(status='fitted', selected_iterations=selected,
                    inner_train_rows=len(a), inner_validation_rows=len(b),
                    inner_best_log_loss=float(selector.get_best_score()['validation']['Logloss']))
    del selector
    # A new model is essential: every state training row participates in refit,
    # and the selection-validation labels are no longer an evaluation set.
    expert = new_cat(cats, seed, selected)
    expert.fit(state_x, state_y)
    return expert, metadata


class ConditionalPredictor:
    def __init__(self, global_model, experts, metadata):
        self.global_model = global_model
        self.experts = experts
        self.metadata = metadata
        self.weight = float(WEIGHT)

    def predict_proba(self, raw):
        x = add_contrasts(raw)
        result = self.global_model.predict_proba(x).copy()
        for state, expert in self.experts.items():
            mask = state_mask(x, state)
            if expert is not None and mask.any():
                result[mask] = ((1 - self.weight) * result[mask]
                                + self.weight * expert.predict_proba(x.loc[mask]))
        # Unrecognized/missing CryoSleep rows never enter either assignment.
        return result


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name != NAME:
        raise ValueError(f'Unknown conditional-expert candidate: {name}')
    started = time.monotonic()
    target = np.asarray(y)
    if len(target) != len(train_idx):
        raise ValueError('Targets must match outer training rows.')
    global_model, x, cats = fit_global(matrices, train_idx, target, predict_idx, seed)
    experts, audits = {}, []
    for state in [False, True]:
        experts[state], audit = fit_expert(x.iloc[train_idx], target, cats, seed, state)
        audits.append(audit)
        print('Conditional expert', json.dumps(audit), flush=True)
    metadata = dict(seed=int(seed), outer_train_rows=len(target), experts=audits,
                    weight=WEIGHT, unknown_policy='unchanged_v34',
                    note='On the Earth/sleep spline slice, final weights are .25 global Cat, .25 spline, .5 state Cat.')
    predictor = ConditionalPredictor(global_model, experts, metadata)
    probabilities = predictor.predict_proba(matrices[False][0].iloc[predict_idx])[:, 1]
    metadata['seconds'] = time.monotonic() - started
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT / f'conditional_training_seed{seed}_{key}.json').write_text(
            json.dumps(metadata, indent=2) + '\n')
    return probabilities, predictor


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'conditional_expert_policy' in protocol:
        return
    for filename in ['conditional_experts.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT / filename
        shutil.copy2(source, out / filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['conditional_expert_policy'] = (
        'Global model exactly v34: v6 Cat on observed-only spend contrasts plus the existing spatial spline. '
        'Known CryoSleep states receive fixed .5 global/.5 state expert probabilities; unknown stays global. '
        'Each expert uses identical contrast inputs and v6 Cat parameters. Within each outer training '
        'state subset, stratified20% validation selects iterations by Logloss with patience50/max700. '
        'A fresh model refits all state training rows for that count. No outer-validation label access. '
        'Global spline contribution is diluted from .5 to .25 on its eligible slice.')
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def evaluate(name, matrices, train, dev, seed, out):
    # Freeze dependent sources before the first baseline/candidate is trained.
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def probe():
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    p, predictor = fit(NAME, matrices, dev[a], y[a], dev[b], iterate.SEEDS[0])
    raw = matrices[False][0].iloc[dev[b]]
    global_p = predictor.global_model.predict_proba(add_contrasts(raw))[:, 1]
    unknown = ~(state_mask(raw, False) | state_mask(raw, True))
    assert np.array_equal(p[unknown], global_p[unknown])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    with tempfile.TemporaryDirectory(prefix='conditional-probe-') as folder:
        artifact = Path(folder) / 'model.joblib'
        joblib.dump(predictor, artifact)
        restored = joblib.load(artifact)
        np.testing.assert_array_equal(restored.predict_proba(raw)[:, 1], p)
    result = dict(predictor.metadata, prediction_rows=len(b),
                  unknown_rows=int(unknown.sum()), unknown_exactly_unchanged=True,
                  valid_probabilities=True, serialization_exactly_unchanged=True,
                  accuracy_computed=False,
                  probe_note='One development fold resource/behavior probe only; not screening.')
    previous = iterate.ROOT / 'artifacts/run_012/v34_contrasts_spatial_seed20261112.npz'
    if previous.exists():
        old = np.load(previous)['prob'][b]
        result['v34_global_max_absolute_difference'] = float(np.max(np.abs(old - global_p)))
        np.testing.assert_allclose(global_p, old, rtol=0, atol=1e-12)
    print(json.dumps(result), flush=True)
    return result


iterate.fit = fit
iterate.evaluate = evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_018')
    args = parser.parse_args()
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT / 'artifacts' / args.run
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    sys.modules['conditional_experts'] = sys.modules[__name__]
    ConditionalPredictor.__module__ = 'conditional_experts'
    main()
