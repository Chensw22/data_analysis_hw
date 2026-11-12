"""One fixed soft-target regularization experiment with an explicit inverse link."""
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
from catboost import CatBoostClassifier
import iterate
import spatial_expert
from contrast_features import add_contrasts, ContrastPredictor

BASE_FIT = spatial_expert.ORIGINAL_FIT
BASE_EVALUATE = iterate.evaluate
CONTROL = 'v63_crossentropy_control'
NAME = 'v64_label_smoothing'
EPSILON = .1
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    'v34_contrasts_spatial': dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    CONTROL: dict(context=False, model='crossentropy_spatial', epsilon=0.),
    NAME: dict(context=False, model='crossentropy_spatial', epsilon=EPSILON),
}
iterate.BLEND = []


def soft_targets(y, epsilon):
    y = np.asarray(y, dtype=float)
    if not 0 <= epsilon < 1 or not np.isin(y, [0., 1.]).all():
        raise ValueError('Binary targets and smoothing epsilon in [0,1) are required.')
    return (1-epsilon)*y+epsilon/2


def inverse_link(q, epsilon):
    # E[soft target|X] = (1-epsilon)*P(Y=1|X)+epsilon/2.
    # This algebra does not guarantee calibration of a finite fitted model.
    if epsilon == 0:
        return q
    return np.clip((q-epsilon/2)/(1-epsilon), 1e-6, 1-1e-6)


class InverseLinkPredictor:
    def __init__(self, model, epsilon):
        self.model, self.epsilon = model, float(epsilon)

    def predict_proba(self, x):
        q = self.model.predict_proba(x)[:, 1]
        p = inverse_link(q, self.epsilon)
        return np.column_stack([1-p, p])


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    x, cats = matrices[False]
    changed = {False: (add_contrasts(x), cats)}
    p, original = spatial_expert.fit_spatial('v34_contrasts_spatial', changed,
                                            train_idx, y, predict_idx, seed)
    if name == 'v34_contrasts_spatial':
        return p, ContrastPredictor(original)
    if name not in [CONTROL, NAME]:
        raise ValueError(name)
    epsilon = iterate.CONFIGS[name]['epsilon']
    model = CatBoostClassifier(**iterate.CAT, loss_function='CrossEntropy', cat_features=cats,
                               random_seed=seed, thread_count=4, verbose=False,
                               allow_writing_files=False)
    model.fit(changed[False][0].iloc[train_idx], soft_targets(y, epsilon))
    # Keep the original hard-label spatial expert, replacing only global CatBoost.
    original.base_model = InverseLinkPredictor(model, epsilon)
    predictor = ContrastPredictor(original)
    return predictor.predict_proba(x.iloc[predict_idx])[:, 1], predictor


def evaluate(name, matrices, train, dev, seed, out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'smoothing_policy' not in protocol:
        for filename in ['label_smoothing.py', 'contrast_features.py', 'spatial_expert.py']:
            source = iterate.ROOT/filename
            shutil.copy2(source, out/filename)
            protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        protocol['smoothing_policy'] = ('One fixed epsilon=.1, targets .05/.95; CrossEntropy global '
            'CatBoost with same v34 features/hyperparameters. Matched CrossEntropy hard-label control '
            'epsilon0 plus original v6/v34. Before original spatial probability blend, map q to '
            'clip((q-.05)/.9,1e-6,1-1e-6); theoretical inverse of soft conditional expectation, not '
            'a claim of empirically calibrated probabilities. Original spatial expert still trains '
            'on hard outertrain labels, no validation tuning, no smoothing-strength search.')
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


iterate.fit = fit
iterate.evaluate = evaluate

if __name__ == '__main__':
    sys.modules['label_smoothing'] = sys.modules[__name__]
    InverseLinkPredictor.__module__ = 'label_smoothing'
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_032')
    args = parser.parse_args()
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](
        iterate.ROOT/'artifacts'/args.run)
