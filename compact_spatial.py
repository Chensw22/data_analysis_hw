"""A frozen ablation of redundant engineered features in the strongest structure."""
import argparse
import hashlib
import json
import shutil
import sys
import iterate
import spatial_expert
from contrast_features import add_contrasts

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
CONTROL = 'v34_contrasts_spatial'
NAME = 'v53_compact_spatial'
COMPACT = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
           'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNumber', 'Side',
           'DiningShoppingMinusLuxury', 'LogDiningShoppingLuxuryRatio']
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    CONTROL: dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    NAME: dict(context=False, model='spatial', knots=5, C=.1, weight=.5, columns=COMPACT),
}
iterate.BLEND = []


class CompactPredictor:
    def __init__(self, model, columns):
        self.model, self.columns = model, list(columns)

    def predict_proba(self, x):
        return self.model.predict_proba(add_contrasts(x)[self.columns])


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    x, cats = matrices[False]
    x = add_contrasts(x)
    columns = COMPACT if name == NAME else list(x.columns)
    x = x[columns]
    changed = {False: (x, [c for c in cats if c in columns])}
    p, model = spatial_expert.fit_spatial(name, changed, train_idx, y, predict_idx, seed)
    return p, CompactPredictor(model, columns)


def evaluate(name, matrices, train, dev, seed, out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'compact_policy' not in protocol:
        for filename in ['compact_spatial.py', 'spatial_expert.py', 'contrast_features.py']:
            source = iterate.ROOT/filename
            shutil.copy2(source, out/filename)
            protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        protocol['compact_policy'] = ('One frozen ablation retaining13 original semantic fields plus2 fixed '
            'v34 spending contrasts. Remove duplicate amount transforms, aggregate amounts/counts and '
            'missing indicators, retaining native NaN/missing category. Same fixed CatBoost and spatial '
            'expert settings. No target-driven feature selection or successive subset search.')
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


iterate.fit = fit
iterate.evaluate = evaluate

if __name__ == '__main__':
    sys.modules['compact_spatial'] = sys.modules[__name__]
    CompactPredictor.__module__ = 'compact_spatial'
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_025')
    args = parser.parse_args()
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](
        iterate.ROOT/'artifacts'/args.run)
