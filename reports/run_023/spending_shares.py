"""Frozen v34 plus three zero-safe within-group spending allocation shares."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import iterate
import spatial_expert
from contrast_features import add_contrasts

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
CONTROL = 'v34_contrasts_spatial'
NAME = 'v51_hierarchical_spend'
# The first member is the numerator; every member is required to be observed.
SHARES = {
    'FoodWithinDiningShare': ['FoodCourt', 'ShoppingMall'],
    'RoomWithinLuxuryShare': ['RoomService', 'Spa', 'VRDeck'],
    'SpaWithinSpaVRShare': ['Spa', 'VRDeck'],
}
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    CONTROL: dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    NAME: dict(context=False, model='spatial_shares', knots=5, C=.1, weight=.5,
               shares=SHARES, missing='any_required_missing_or_zero_denominator_is_nan'),
}
iterate.BLEND = []


def add_shares(x, groups=SHARES):
    result = x.copy()
    for name, columns in groups.items():
        amounts = x[columns]
        # Full min_count propagates partial missingness; a real zero is retained.
        denominator = amounts.sum(axis=1, min_count=len(columns)).replace(0, np.nan)
        result[name] = amounts.iloc[:, 0] / denominator
    return result


class SharePredictor:
    def __init__(self, model, groups):
        self.model = model
        self.groups = {name: list(columns) for name, columns in groups.items()}

    def predict_proba(self, raw):
        return self.model.predict_proba(add_shares(add_contrasts(raw), self.groups))


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in [CONTROL, NAME]:
        raise ValueError(f'Unknown spending-share candidate: {name}')
    x, cats = matrices[False]
    groups = SHARES if name == NAME else {}
    changed = {False: (add_shares(add_contrasts(x), groups), cats)}
    # fit_spatial reads knots/C/weight from this candidate's own configuration.
    # This preserves the original .5 spline weight without a hidden v25 entry.
    p, model = spatial_expert.fit_spatial(name, changed, train_idx, y, predict_idx, seed)
    return p, SharePredictor(model, groups)


def record_protocol(out):
    path = out / 'protocol.json'
    protocol = json.loads(path.read_text())
    if 'spending_share_policy' in protocol:
        return
    for filename in ['spending_shares.py', 'contrast_features.py', 'spatial_expert.py']:
        source = iterate.ROOT / filename
        shutil.copy2(source, out / filename)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol['spending_share_policy'] = (
        'One predeclared candidate: v34 plus Food/(Food+Shopping), '
        'Room/(Room+Spa+VR), and Spa/(Spa+VR). Each ratio is NaN if any required '
        'amount is missing or its denominator is zero. No pseudocounts, clipping, '
        'imputation, learned weights, target use, or row aggregation. Original amounts, '
        'missing flags, totals and contrasts remain. Same v6 CatBoost and original '
        '5-knot/C=.1/.5-weight spatial expert; v6 and exact v34 are controls. '
        'These are allocation shares, not CLR/ILR coordinates or a fixed-budget model.')
    path.write_text(json.dumps(protocol, indent=2) + '\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


iterate.fit = fit
iterate.evaluate = evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_023')
    args = parser.parse_args()
    out = iterate.ROOT / 'artifacts' / args.run
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](out)


if __name__ == '__main__':
    sys.modules['spending_shares'] = sys.modules[__name__]
    SharePredictor.__module__ = 'spending_shares'
    main()
