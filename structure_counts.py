"""One matched v34 ablation adding fixed, unlabeled roster membership counts."""
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

import iterate
BASE_DATASET = iterate.dataset
BASE_EVALUATE = iterate.evaluate
import spatial_expert
from contrast_features import add_contrasts

BASE_FIT = spatial_expert.ORIGINAL_FIT
CONTROL = 'v34_contrasts_spatial'
NAME = 'v66_structure_counts'
COUNTS = ['GroupSize', 'CabinOccupants']
LOOKUP = None
EXPECTED_ROSTER_HASH = None
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    CONTROL: dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    NAME: dict(context=False, model='spatial_counts', knots=5, C=.1, weight=.5,
               counts=COUNTS, roster='all_public_unlabeled_rows', missing_cabin='NaN'),
}
iterate.BLEND = []


class RosterCounts:
    """Immutable-by-contract membership lookup; no target or peer attributes stored."""
    def __init__(self, roster):
        raw = roster.drop(columns='Transported', errors='ignore')
        ids = raw.PassengerId.astype(str)
        if ids.duplicated().any() or raw.PassengerId.isna().any():
            raise ValueError('Fixed roster requires unique observed PassengerId values.')
        self.cabins = pd.Series(raw.Cabin.to_numpy(), index=ids.to_numpy())
        groups = ids.str.split('_').str[0]
        self.group_counts = groups.value_counts().to_dict()
        self.cabin_counts = raw.Cabin.value_counts(dropna=True).to_dict()
        canonical = raw[['PassengerId', 'Cabin']].reset_index(drop=True)
        self.roster_sha256 = hashlib.sha256(
            pd.util.hash_pandas_object(canonical, index=True).to_numpy().tobytes()).hexdigest()

    def transform(self, raw):
        raw = raw.drop(columns='Transported', errors='ignore')
        ids = raw.PassengerId.astype(str)
        groups = ids.str.split('_').str[0]
        if not ids.isin(self.cabins.index).all() or not groups.isin(self.group_counts).all():
            raise ValueError('Passenger/group is outside the fixed public roster.')
        known_cabins = self.cabins.reindex(ids.to_numpy()).to_numpy()
        observed = raw.Cabin.to_numpy()
        same = (pd.isna(known_cabins) & pd.isna(observed)) | (
            pd.Series(known_cabins).eq(pd.Series(observed)).fillna(False).to_numpy())
        if not same.all() or not raw.Cabin.dropna().isin(self.cabin_counts).all():
            raise ValueError('Cabin differs from the fixed public roster.')
        return pd.DataFrame({
            'GroupSize': groups.map(self.group_counts).to_numpy(dtype=float),
            'CabinOccupants': raw.Cabin.map(self.cabin_counts).to_numpy(dtype=float),
        }, index=raw.index)


def dataset():
    global LOOKUP
    train, test, matrices, dev = BASE_DATASET()
    roster = pd.concat([train.drop(columns='Transported'), test.drop(
        columns='Transported', errors='ignore')], ignore_index=True)
    LOOKUP = RosterCounts(roster)
    if EXPECTED_ROSTER_HASH is not None and LOOKUP.roster_sha256 != EXPECTED_ROSTER_HASH:
        raise RuntimeError('Public roster membership changed since screening.')
    counts = LOOKUP.transform(roster)
    x, cats = matrices[False]
    if not x.index.equals(counts.index):
        raise ValueError('Roster and feature row order must match.')
    matrices[False] = (pd.concat([x, counts], axis=1), cats)
    return train, test, matrices, dev


class StructurePredictor:
    """predict_proba expects prepared features; predict_raw_proba uses frozen roster."""
    def __init__(self, model, lookup, columns, add_contrast, use_counts):
        self.model = model
        self.lookup = lookup
        self.columns = list(columns)
        self.add_contrast = bool(add_contrast)
        self.use_counts = bool(use_counts)

    def predict_proba(self, prepared):
        x = prepared.drop(columns='Transported', errors='ignore').copy()
        if not self.use_counts:
            x = x.drop(columns=COUNTS, errors='ignore')
        elif not set(COUNTS).issubset(x.columns):
            raise ValueError('Prepared prediction requires the frozen-roster count columns; use predict_raw_proba.')
        if self.add_contrast:
            x = add_contrasts(x)
        return self.model.predict_proba(x[self.columns])

    def predict_raw_proba(self, raw):
        raw = raw.drop(columns='Transported', errors='ignore').copy()
        # Validate membership even for controls: one explicit fixed-roster contract.
        counts = self.lookup.transform(raw)
        x, _ = iterate.base.features(raw, 'engineered_no_ids')
        if self.use_counts:
            x = pd.concat([x, counts], axis=1)
        return self.predict_proba(x)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if LOOKUP is None:
        raise RuntimeError('Initialize the fixed unlabeled roster through dataset().')
    x, cats = matrices[False]
    x = x.drop(columns=COUNTS, errors='ignore') if name != NAME else x.copy()
    if name == iterate.BASELINE:
        p, model = BASE_FIT(name, {False: (x, cats)}, train_idx, y, predict_idx, seed)
        return p, StructurePredictor(model, LOOKUP, x.columns, False, False)
    if name not in [CONTROL, NAME]:
        raise ValueError(name)
    x = add_contrasts(x)
    p, model = spatial_expert.fit_spatial(name, {False: (x, cats)}, train_idx, y,
                                        predict_idx, seed)
    return p, StructurePredictor(model, LOOKUP, x.columns, True, name == NAME)


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'structure_count_policy' in protocol:
        return
    for filename in ['structure_counts.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['structure_count_policy'] = (
        'Only two numeric columns added to v34: observed GroupSize and CabinOccupants from '
        'the fixed public train+test attribute roster, with Transported removed before counting. '
        'Missing Cabin remains NaN. No IDs, target statistics, peer attributes or imputation. '
        'v6/v34 controls explicitly strip both new columns. Original CatBoost, contrasts and '
        '5-knot/C=.1/.5 spatial expert unchanged. Counts never recomputed on a prediction subset. '
        'Serialized predictors retain membership maps and roster hash; raw prediction rejects '
        'unknown passengers/groups or changed cabins. Prepared prediction requires the same '
        'frozen feature-construction contract. No claim that roster counts equal actual ship totals.')
    protocol['structure_count_roster_sha256'] = LOOKUP.roster_sha256
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


iterate.dataset = dataset
iterate.fit = fit
iterate.evaluate = evaluate


def main():
    global EXPECTED_ROSTER_HASH
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_034')
    args = parser.parse_args()
    out = iterate.ROOT/'artifacts'/args.run
    if args.stage in ['confirm', 'export']:
        EXPECTED_ROSTER_HASH = json.loads((out/'protocol.json').read_text())[
            'structure_count_roster_sha256']
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](out)


if __name__ == '__main__':
    sys.modules['structure_counts'] = sys.modules[__name__]
    for cls in [RosterCounts, StructurePredictor]:
        cls.__module__ = 'structure_counts'
    main()
