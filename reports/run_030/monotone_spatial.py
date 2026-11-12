"""Matched feature/CTR controls for one fixed spending-monotonicity prior."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from threadpoolctl import threadpool_limits

import iterate
import spatial_expert
import contrast_features

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
CONTROL = 'v59_monotone_control'
NAME = 'v60_monotone_spatial'
DROP_COLUMNS = ('TotalSpend', 'AllSpendingZero', 'PaidServices')
SIGNS = {'RoomService': -1, 'FoodCourt': 1, 'ShoppingMall': 1, 'Spa': -1, 'VRDeck': -1}
CONSTRAINTS = {**SIGNS, **{name + '_log1p': sign for name, sign in SIGNS.items()},
               'LuxurySpend': -1, 'DiningShoppingMinusLuxury': 1, 'LogDiningShoppingLuxuryRatio': 1}
SPATIAL = dict(knots=5, C=.1, weight=.5)
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    'v34_contrasts_spatial': dict(context=False, model='spatial', **SPATIAL),
    CONTROL: dict(context=False, model='monotone_control', drop_columns=list(DROP_COLUMNS),
                  max_ctr_complexity=1, constraints={}, spatial=SPATIAL, cat=iterate.CAT),
    NAME: dict(context=False, model='monotone_spatial', drop_columns=list(DROP_COLUMNS),
               max_ctr_complexity=1, constraints=CONSTRAINTS, spatial=SPATIAL, cat=iterate.CAT),
}
iterate.BLEND = []


def prepare_raw(raw):
    return iterate.base.features(raw.drop(columns='Transported', errors='ignore'), 'engineered_no_ids')


def candidate_features(prepared):
    x = contrast_features.add_contrasts(prepared).drop(columns=list(DROP_COLUMNS))
    if not set(CONSTRAINTS).issubset(x.columns):
        raise ValueError('The fixed 13 monotonic features must all be present.')
    return x


class MonotoneSpatialPredictor:
    def __init__(self, model, columns, constraints):
        self.model = model
        self.columns = list(columns)
        self.constraints = dict(constraints)
        self.drop_columns = list(DROP_COLUMNS)
        self.max_ctr_complexity = 1

    def predict_proba(self, prepared):
        x = candidate_features(prepared)
        return self.model.predict_proba(x[self.columns])

    def predict_proba_raw(self, raw):
        return self.predict_proba(prepare_raw(raw)[0])


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    raw, cats = matrices[False]
    if name == 'v34_contrasts_spatial':
        p, model = spatial_expert.fit_spatial(name,
            {False: (contrast_features.add_contrasts(raw), cats)}, train_idx, y, predict_idx, seed)
        return p, contrast_features.ContrastPredictor(model)
    if name not in [CONTROL, NAME]:
        raise ValueError(f'Unknown monotonicity candidate: {name}')
    if len(y) != len(train_idx) or np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Targets must match training rows; prediction rows must be disjoint.')
    x = candidate_features(raw)
    cats = [col for col in cats if col in x]
    constraints = dict(CONSTRAINTS) if name == NAME else {}
    params = dict(iterate.CAT, max_ctr_complexity=1)
    if constraints:
        params['monotone_constraints'] = constraints
    base_model = CatBoostClassifier(**params, cat_features=cats, loss_function='Logloss',
                                    random_seed=seed, thread_count=4, verbose=False,
                                    allow_writing_files=False)
    with threadpool_limits(limits=4):
        base_model.fit(x.iloc[train_idx], np.asarray(y))
    training = x.iloc[train_idx]
    mask = spatial_expert.SpatialPredictor.eligible(training)
    specialist = training.loc[mask]
    spline = make_pipeline(SimpleImputer(strategy='median'),
                           SplineTransformer(n_knots=5, degree=3, include_bias=False))
    side = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    spline.fit(specialist[['CabinNumber']])
    side.fit(specialist[['Side']])
    expert = LogisticRegression(C=.1, max_iter=1500, random_state=seed)
    model = spatial_expert.SpatialPredictor(base_model, spline, side, expert, .5)
    expert.fit(model.design(specialist), np.asarray(y)[mask.to_numpy()])
    predictor = MonotoneSpatialPredictor(model, x.columns, constraints)
    return predictor.predict_proba(raw.iloc[predict_idx])[:, 1], predictor


def audit_raw_monotonicity(predictor, raw):
    """Rebuild ALL row-local features; missingness itself is held fixed."""
    grid = np.array([0., 1., 10., 100., 1000., 10000.])
    results = []
    for col, direction in SIGNS.items():
        eligible = raw[col].notna()
        base = raw.loc[eligible].copy()
        probabilities = []
        for value in grid:
            changed = base.copy()
            changed[col] = value
            probabilities.append(predictor.predict_proba_raw(changed)[:, 1])
        steps = np.diff(np.column_stack(probabilities), axis=1) * direction
        violations = int(np.sum(steps < -1e-10))
        minimum = float(steps.min()) if steps.size else None
        results.append(dict(feature=col, direction=direction, rows=len(base),
                            adjacent_checks=int(steps.size), violations=violations,
                            minimum_signed_change=minimum))
        if violations:
            raise RuntimeError(f'Raw end-to-end monotonicity failed for {col}; stop the route without encoding repairs.')
    return dict(grid=grid.tolist(), tolerance=1e-10, features=results,
                note='Finite numerical checks of the full feature construction and spatial blend; not empirical proof of a causal spending mechanism.')


def probe():
    train, _, matrices, dev = iterate.dataset()
    rng = np.random.default_rng(iterate.SEEDS[0])
    chosen = rng.choice(dev, size=640, replace=False)
    a, b = chosen[:512], chosen[512:]
    target = train.Transported.iloc[a].astype(int).to_numpy()
    _, predictor = fit(NAME, matrices, a, target, b, iterate.SEEDS[0])
    # Cross all known planets and Cryo states while retaining observed missing
    # patterns/cabins; deliberately change observed spending through zero too.
    original = train.iloc[b].drop(columns='Transported').copy()
    scenarios = []
    for planet in ['Earth', 'Europa', 'Mars']:
        for cryo in [True, False, None]:
            part = original.copy()
            part['HomePlanet'] = planet
            part['CryoSleep'] = cryo
            scenarios.append(part)
    raw = pd.concat(scenarios, ignore_index=True)
    audit = audit_raw_monotonicity(predictor, raw)
    p = predictor.predict_proba_raw(raw)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    with tempfile.TemporaryDirectory(prefix='monotone-probe-') as folder:
        path = Path(folder) / 'model.joblib'
        joblib.dump(predictor, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.predict_proba_raw(raw), p)
        assert restored.columns == predictor.columns and restored.constraints == CONSTRAINTS
    fitted = predictor.model.base_model
    assert fitted.get_param('max_ctr_complexity') == 1
    assert fitted.get_param('monotone_constraints') == CONSTRAINTS
    prepared, _ = prepare_raw(raw)
    route = spatial_expert.SpatialPredictor.eligible(prepared)
    result = dict(training_rows=len(a), audit_rows=len(raw), spatial_route_rows=int(route.sum()),
                  nonspatial_route_rows=int((~route).sum()),
                  rows_with_input_missing=int(raw.isna().any(axis=1).sum()),
                  constraints=CONSTRAINTS, feature_columns=predictor.columns,
                  max_ctr_complexity=1, serialization_exact=True, monotonicity=audit,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  accuracy_computed=False)
    print(json.dumps(result), flush=True)
    return result


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'monotone_spatial_policy' in protocol:
        return
    for filename in ['monotone_spatial.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT / filename
        shutil.copy2(source, out / filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['monotone_spatial_policy'] = (
        'Original v6 and v34 controls unchanged. Both v59/v60 remove exactly TotalSpend, '
        'AllSpendingZero and PaidServices, retain every other full-v34 field, and explicitly '
        'set max_ctr_complexity=1 to exclude categorical/float combination paths. v59 has '
        'no monotonicity constraints. v60 constrains raw/log FoodCourt and ShoppingMall +1, '
        'raw/log RoomService/Spa/VRDeck -1, LuxurySpend -1, both spending contrasts +1. '
        'Same Cat depth6/700/.045/L2=5, original training-only spatial expert and eligibility, '
        'knots5/C=.1/.5 blend, decision threshold .5. Fixed prior, no sign/parameter search. '
        'The raw prediction API recomputes all derived fields and preserves feature columns '
        'and constraints in its serialized wrapper. Monotonicity applies to changes of observed '
        'nonnegative amounts with other raw inputs/missingness fixed, not to NaN-to-observed '
        'changes or a causal interpretation. Real raw-reconstruction monotonicity probe must '
        'pass before screening; failure ends this route without encoding repairs.')
    protocol['monotone_constraints'] = CONSTRAINTS
    protocol['monotone_drop_columns'] = list(DROP_COLUMNS)
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


iterate.fit = fit
iterate.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_030')
    args = parser.parse_args()
    if args.stage == 'probe':
        probe()
        return
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](
        iterate.ROOT / 'artifacts' / args.run)


if __name__ == '__main__':
    sys.modules['monotone_spatial'] = sys.modules[__name__]
    MonotoneSpatialPredictor.__module__ = 'monotone_spatial'
    main()
