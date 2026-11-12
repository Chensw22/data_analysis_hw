"""Consistent posterior-weighted HomePlanet expansion for originally missing rows."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

import iterate
import spatial_expert
from contrast_features import add_contrasts

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
CONTROL = 'v34_contrasts_spatial'
NAME = 'v58_homeplanet_marginal'
PLANETS = ['Earth', 'Europa', 'Mars']
Q_FEATURES = ['CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt',
              'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNumber', 'Side']
Q_PARAMS = dict(iterations=200, depth=4, learning_rate=.06, l2_leaf_reg=5)
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    CONTROL: dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    NAME: dict(context=False, model='planet_marginal', posterior=Q_PARAMS,
               posterior_features=Q_FEATURES, target_cat=iterate.CAT,
               branches=PLANETS, original_spatial=dict(knots=5, C=.1, weight=.5),
               routing='original_HomePlanet_missing_only'),
}
iterate.BLEND = []
RUN_OUT = None


def missing_planet(x):
    return (x.HomePlanet.isna() | x.HomePlanet.eq('__MISSING__')).to_numpy(dtype=bool)


def posterior_features(x):
    if 'Transported' in x:
        raise ValueError('Transported must never enter the attribute posterior pipeline.')
    return x[Q_FEATURES]


def fit_posterior(outer_x, seed):
    """Attribute labels only: this function does not accept Transported labels."""
    known = ~missing_planet(outer_x)
    qx = posterior_features(outer_x)
    labels = outer_x.HomePlanet.loc[known].astype(str)
    if set(labels.unique()) != set(PLANETS):
        raise RuntimeError('All three HomePlanet classes must be present in the outer training data.')
    cats = qx.select_dtypes(include=['object', 'str', 'bool']).columns.tolist()
    model = CatBoostClassifier(**Q_PARAMS, cat_features=cats, random_seed=seed,
        thread_count=4, verbose=False, allow_writing_files=False, loss_function='MultiClass')
    model.fit(qx.loc[known], labels)
    return model


def posterior_probabilities(model, x):
    if list(model.classes_) != PLANETS:
        raise RuntimeError('Unexpected attribute posterior class order.')
    p = model.predict_proba(posterior_features(x))
    if (p.shape != (len(x), 3) or not np.isfinite(p).all()
        or not ((p >= 0) & (p <= 1)).all()
        or not np.allclose(p.sum(axis=1), 1., atol=1e-12)):
        raise RuntimeError('Invalid HomePlanet posterior probabilities.')
    return p


def expand_training(outer_x, posterior):
    """Return contrast features, original row positions, and unit-mass row weights."""
    missing = missing_planet(outer_x)
    observed_positions = np.flatnonzero(~missing)
    missing_positions = np.flatnonzero(missing)
    features = add_contrasts(outer_x)
    frames = [features.iloc[observed_positions].copy()]
    positions = [observed_positions]
    weights = [np.ones(len(observed_positions))]
    if len(missing_positions):
        q = posterior_probabilities(posterior, outer_x.iloc[missing_positions])
        for column, planet in enumerate(PLANETS):
            branch = features.iloc[missing_positions].copy()
            branch['HomePlanet'] = planet
            frames.append(branch)
            positions.append(missing_positions)
            weights.append(q[:, column])
    row_positions = np.concatenate(positions)
    sample_weights = np.concatenate(weights)
    np.testing.assert_allclose(np.bincount(row_positions, weights=sample_weights,
                                          minlength=len(outer_x)), 1., rtol=0, atol=1e-12)
    return pd.concat(frames, ignore_index=True), row_positions, sample_weights


class PlanetMarginalPredictor:
    def __init__(self, original, marginal, posterior, metadata):
        self.original = original
        self.marginal = marginal
        self.posterior = posterior
        self.metadata = metadata
        self.planets = list(PLANETS)

    def predict_proba(self, raw):
        x = add_contrasts(raw)
        result = self.original.predict_proba(x).copy()
        missing = missing_planet(raw)
        if missing.any():
            q = posterior_probabilities(self.posterior, raw.loc[missing])
            mixture = np.zeros((int(missing.sum()), 2))
            for column, planet in enumerate(self.planets):
                branch = x.loc[missing].copy()
                branch['HomePlanet'] = planet
                mixture += q[:, column, None] * self.marginal.predict_proba(branch)
            result[missing] = mixture
        return result


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in [CONTROL, NAME]:
        raise ValueError(name)
    raw, cats = matrices[False]
    changed = {False: (add_contrasts(raw), cats)}
    original_p, original = spatial_expert.fit_spatial(
        CONTROL, changed, train_idx, y, predict_idx, seed)
    if name == CONTROL:
        # The same wrapper has no missing-row correction when marginal is absent.
        from contrast_features import ContrastPredictor
        return original_p, ContrastPredictor(original)
    started = time.monotonic()
    target = np.asarray(y, dtype=np.int64)
    if len(target) != len(train_idx):
        raise ValueError('Target labels must align with outer training rows.')
    outer = raw.iloc[train_idx]
    posterior = fit_posterior(outer, seed)
    augmented, positions, weights = expand_training(outer, posterior)
    cat = CatBoostClassifier(**iterate.CAT, cat_features=cats, random_seed=seed,
        thread_count=4, verbose=False, allow_writing_files=False, loss_function='Logloss')
    cat.fit(augmented, target[positions], sample_weight=weights)
    # Reuse the original known-Earth expert. No posterior branches fit its knots or coefficients.
    marginal = spatial_expert.SpatialPredictor(cat, original.spline, original.side_encoder,
                                              original.expert, original.weight)
    metadata = dict(seed=int(seed), outer_training_rows=len(outer),
        posterior_training_rows=int((~missing_planet(outer)).sum()),
        missing_planet_training_rows=int(missing_planet(outer).sum()),
        expanded_target_rows=len(positions), total_sample_weight=float(weights.sum()),
        objective='posterior-weighted complete-data logloss approximation; not exact observed-data likelihood',
        spatial_policy='original outertrain known-Earth expert, evaluated on each completed branch',
        observed_planet_policy='unchanged original v34 probabilities')
    predictor = PlanetMarginalPredictor(original, marginal, posterior, metadata)
    p = predictor.predict_proba(raw.iloc[predict_idx])[:, 1]
    observed = ~missing_planet(raw.iloc[predict_idx])
    np.testing.assert_array_equal(p[observed], original_p[observed])
    metadata['seconds_after_original_model'] = time.monotonic() - started
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        (RUN_OUT/f'planet_training_seed{seed}_{key}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Planet marginal', json.dumps(metadata), flush=True)
    return p, predictor


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'planet_marginal_policy' in protocol:
        return
    for filename in ['planet_marginal.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['planet_marginal_policy'] = (
        'HomePlanet posterior uses only outertrain rows with known HomePlanet, raw compact12 '
        'attributes excluding HomePlanet, Transported, IDs and missing-count proxies. Missing '
        'target-training rows expand into all three HomePlanet branches with posterior weights '
        'summing to one per original row. Observed rows have weight one. Original missingness '
        'information remains unchanged in both target training and prediction. Target CatBoost '
        'uses original v6 parameters. This is weighted complete-data logloss, not exact Bayesian '
        'observed-data marginal likelihood. Reuse the original v34 spatial spline trained only '
        'on original known Earth/sleep eligible outertrain rows; apply it to Earth prediction '
        'branches at original .5 weight. Only originally missing HomePlanet predictions change; '
        'observed HomePlanet rows return original v34 predictions bitwise unchanged.')
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def probe():
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    p, model = fit(NAME, matrices, dev[a], y[a], dev[b], iterate.SEEDS[0])
    raw = matrices[False][0].iloc[dev[b]]
    original_p = model.original.predict_proba(add_contrasts(raw))[:, 1]
    observed = ~missing_planet(raw)
    np.testing.assert_array_equal(p[observed], original_p[observed])
    previous = iterate.ROOT/'artifacts/run_012/v34_contrasts_spatial_seed20261112.npz'
    previous_difference = None
    if previous.exists():
        old = np.load(previous)['prob'][b]
        previous_difference = float(np.max(np.abs(original_p-old)))
        np.testing.assert_allclose(original_p, old, atol=1e-12, rtol=0)
    with tempfile.TemporaryDirectory(prefix='planet-probe-') as folder:
        path = Path(folder)/'model.joblib'
        joblib.dump(model, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.predict_proba(raw)[:, 1], p)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    result = dict(model.metadata, query_rows=len(b), missing_query_rows=int((~observed).sum()),
                  observed_rows_bitwise_unchanged=True, valid_probabilities=True,
                  original_v34_max_difference=previous_difference,
                  serialization_bitwise_unchanged=True, accuracy_computed=False)
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_029')
    args = parser.parse_args()
    if args.stage == 'probe':
        probe()
        return
    RUN_OUT = iterate.ROOT/'artifacts'/args.run
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)


if __name__ == '__main__':
    sys.modules['planet_marginal'] = sys.modules[__name__]
    PlanetMarginalPredictor.__module__ = 'planet_marginal'
    main()
