"""Change only the existing Earth/sleep spatial expert's numerical coordinate."""
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import iterate
import spatial_expert
import contrast_features

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_DATASET = iterate.dataset
BASE_EVALUATE = iterate.evaluate
NAME = 'v52_order_coordinate'
CONTROL = 'v34_contrasts_spatial'
COORDINATE = 'SpatialGroupCoordinate'
SPATIAL = dict(knots=5, C=.1, weight=.5)
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    CONTROL: dict(context=False, model='spatial', **SPATIAL),
    NAME: dict(context=False, model='order_coordinate', coordinate=COORDINATE,
               eligible='original_Earth_CryoSleep_known_CabinNumber_and_Side', **SPATIAL),
}
iterate.BLEND = []


def group_coordinate(raw):
    """A row-local input attribute, never a label/group lookup or row-id mapping."""
    passenger = raw.PassengerId.astype('string')
    if passenger.isna().any() or not passenger.str.fullmatch(r'\d+_\d+').all():
        raise ValueError('Every PassengerId must have the numeric group_member format.')
    return pd.to_numeric(passenger.str.split('_').str[0], errors='raise').astype(float)


def prepare_raw(raw):
    """The serialized predictor's independent raw-data feature construction API."""
    roster = raw.drop(columns='Transported', errors='ignore')
    coordinate = group_coordinate(roster)
    x, cats = iterate.base.features(roster, 'engineered_no_ids')
    x[COORDINATE] = coordinate.to_numpy()
    return x, cats


def dataset():
    train, test, matrices, dev = BASE_DATASET()
    roster = pd.concat([train.drop(columns='Transported', errors='ignore'),
                        test.drop(columns='Transported', errors='ignore')], ignore_index=True)
    x, cats = matrices[False]
    if len(x) != len(roster) or not x.index.equals(pd.RangeIndex(len(roster))):
        raise ValueError('Prepared matrix must preserve full-roster positional row order.')
    enriched = x.copy()
    enriched[COORDINATE] = group_coordinate(roster).to_numpy()
    matrices = {**matrices, False: (enriched, list(cats))}
    return train, test, matrices, dev


def without_coordinate(matrices):
    x, cats = matrices[False]
    return {False: (x.drop(columns=COORDINATE, errors='ignore'), list(cats))}


class OrderCoordinatePredictor:
    def __init__(self, base_model, spline, side_encoder, expert, cat_columns):
        self.base_model = base_model
        self.spline = spline
        self.side_encoder = side_encoder
        self.expert = expert
        self.cat_columns = list(cat_columns)
        self.coordinate_column = COORDINATE
        self.weight = .5

    def design(self, x):
        smooth = self.spline.transform(x[[self.coordinate_column]])
        side = self.side_encoder.transform(x[['Side']])
        return np.concatenate([side, (smooth[:, :, None] * side[:, None, :]).reshape(len(x), -1)], axis=1)

    def predict_proba(self, prepared):
        x = contrast_features.add_contrasts(prepared)
        # The global CatBoost receives exactly v34's original column names/order.
        p = self.base_model.predict_proba(x[self.cat_columns])[:, 1]
        mask = spatial_expert.SpatialPredictor.eligible(x)
        if mask.any():
            if self.coordinate_column not in x or x.loc[mask, self.coordinate_column].isna().any():
                raise ValueError('Eligible rows require a PassengerId-derived spatial coordinate.')
            specialist = self.expert.predict_proba(self.design(x.loc[mask]))[:, 1]
            p[mask] = (1 - self.weight) * p[mask] + self.weight * specialist
        return np.column_stack([1 - p, p])

    def predict_proba_raw(self, raw):
        return self.predict_proba(prepare_raw(raw)[0])


def fit(name, matrices, train_idx, y, predict_idx, seed):
    ordinary = without_coordinate(matrices)
    if name == iterate.BASELINE:
        return BASE_FIT(name, ordinary, train_idx, y, predict_idx, seed)
    x, cats = ordinary[False]
    changed = {False: (contrast_features.add_contrasts(x), cats)}
    if name == CONTROL:
        p, model = spatial_expert.fit_spatial(CONTROL, changed, train_idx, y, predict_idx, seed)
        return p, contrast_features.ContrastPredictor(model)
    if name != NAME:
        raise ValueError(f'Unknown coordinate candidate: {name}')
    if len(y) != len(train_idx) or np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Targets must match training rows and prediction rows must be disjoint.')
    prepared, _ = matrices[False]
    if COORDINATE not in prepared or prepared[COORDINATE].isna().any():
        raise ValueError('Use dataset()/prepare_raw() to build the PassengerId-derived coordinate.')
    _, base_model = BASE_FIT(iterate.BASELINE, changed, train_idx, y, predict_idx, seed)
    training = prepared.iloc[train_idx]
    mask = spatial_expert.SpatialPredictor.eligible(training)
    specialist = training.loc[mask]
    # Only this input coordinate differs from the existing spatial expert.
    spline = make_pipeline(SimpleImputer(strategy='median'),
                           SplineTransformer(n_knots=5, degree=3, include_bias=False))
    side = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    spline.fit(specialist[[COORDINATE]])
    side.fit(specialist[['Side']])
    expert = LogisticRegression(C=.1, max_iter=1500, random_state=seed)
    predictor = OrderCoordinatePredictor(base_model, spline, side, expert, changed[False][0].columns)
    expert.fit(predictor.design(specialist), np.asarray(y)[mask.to_numpy()])
    return predictor.predict_proba(prepared.iloc[predict_idx])[:, 1], predictor


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'order_coordinate_policy' in protocol:
        return
    for filename in ['order_coordinate.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT / filename
        shutil.copy2(source, out / filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['order_coordinate_policy'] = (
        'Only the existing Earth/CryoSleep=True/known CabinNumber and P-or-S expert coordinate '
        'changes from CabinNumber to numeric PassengerId group prefix. Eligibility, 5 cubic knots, '
        'side interactions, LogisticRegression C=.1, and .5 probability weight stay unchanged. '
        'All spline fitting uses outer-training eligible rows only. Full-roster attributes explicitly '
        'drop Transported before adding a row-local coordinate; roster positional alignment is checked. '
        'The coordinate is removed for v6/v34 controls and never enters the candidate global CatBoost, '
        'whose exact original v34 column names/order are saved in its wrapper. predict_proba_raw '
        'constructs features from original attributes without any global roster/index lookup. '
        'This is an empirical ordering-coordinate experiment, not a claim that group ID is time '
        'or physical distance. No missing-cabin recovery, wider routing, target encoding, or '
        'simultaneous knot/regularization/weight search. Existing gate compares v6; comparison '
        'with v44 and stability verification remain necessary before any submission approval.')
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


iterate.dataset = dataset
iterate.fit = fit
iterate.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_024')
    args = parser.parse_args()
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](
        iterate.ROOT / 'artifacts' / args.run)


if __name__ == '__main__':
    sys.modules['order_coordinate'] = sys.modules[__name__]
    OrderCoordinatePredictor.__module__ = 'order_coordinate'
    main()
