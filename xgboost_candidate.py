"""One frozen XGBoost recipe, with train-only stopping and full-fold refit."""
import model_families
if __name__ == '__main__':
    model_families.ensure_openmp()

import argparse
import hashlib
import json
import shutil
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost
import iterate
import contrast_features

# model_families import replaces iterate.fit; restore the original engine.
contrast_features.BASE_FIT = model_families.ORIGINAL_FIT
contrast_features.spatial_expert.ORIGINAL_FIT = model_families.ORIGINAL_FIT

XGB = 'v49_xgboost'
BLEND = 'v50_xgboost_v34_blend'
PARAMS = dict(max_depth=4, learning_rate=.035, min_child_weight=5,
              reg_lambda=5., reg_alpha=.1, subsample=.85, colsample_bytree=.9,
              tree_method='hist', objective='binary:logistic', eval_metric='logloss', n_jobs=4)
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    'v25_spatial_blend': dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    'v34_contrasts_spatial': dict(context=False, model='spatial', knots=5, C=.1, weight=.5),
    XGB: dict(context=False, model='xgboost', params=PARAMS, cap=1800, patience=60,
              inner_fraction=.2),
    BLEND: dict(context=False, model='blend', weight=.5),
}
iterate.BLEND = []
RUN_OUT = None


class XGBPredictor:
    def __init__(self, prep, model, metadata):
        self.prep, self.model, self.metadata = prep, model, metadata

    def predict_proba(self, x):
        return self.model.predict_proba(self.prep.transform(contrast_features.add_contrasts(x)))


class BlendPredictor:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def predict_proba(self, x):
        return .5 * self.first.predict_proba(x) + .5 * self.second.predict_proba(x)


def make_preprocessor(x, cats):
    # Dense output: zero numeric/one-hot entries remain actual zeros, not sparse missing values.
    return ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
        ('numeric', 'passthrough', [c for c in x if c not in cats]),
    ], sparse_threshold=0)


def train_xgb(x, cats, y, seed):
    x = contrast_features.add_contrasts(x)
    a, b = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    prep = make_preprocessor(x, cats)
    xa = prep.fit_transform(x.iloc[a]); xb = prep.transform(x.iloc[b])
    selector = XGBClassifier(**PARAMS, n_estimators=1800, early_stopping_rounds=60, random_state=seed)
    selector.fit(xa, y[a], eval_set=[(xb, y[b])], verbose=False)
    count = int(selector.best_iteration) + 1
    final_prep = make_preprocessor(x, cats)
    model = XGBClassifier(**PARAMS, n_estimators=count, random_state=seed)
    model.fit(final_prep.fit_transform(x), y, verbose=False)
    return XGBPredictor(final_prep, model, dict(seed=int(seed), iterations=count,
                        inner_train_rows=len(a), inner_stop_rows=len(b), refit_rows=len(y)))


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name not in (XGB, BLEND):
        return contrast_features.fit(name, matrices, train_idx, y, predict_idx, seed)
    x, cats = matrices[False]
    model = train_xgb(x.iloc[train_idx], cats, np.asarray(y), seed)
    if RUN_OUT is not None:
        key = hashlib.sha256(np.asarray(train_idx).tobytes()).hexdigest()[:12]
        (RUN_OUT/f'{name}_seed{seed}_{key}.json').write_text(json.dumps(model.metadata, indent=2)+'\n')
    if name == BLEND:
        _, other = contrast_features.fit('v34_contrasts_spatial', matrices, train_idx, y, predict_idx, seed)
        model = BlendPredictor(model, other)
    return model.predict_proba(x.iloc[predict_idx])[:, 1], model


ORIGINAL_EVALUATE = iterate.evaluate


def evaluate(name, matrices, train, dev, seed, out):
    protocol = json.loads((out/'protocol.json').read_text())
    if 'xgboost_policy' not in protocol:
        for filename in ['xgboost_candidate.py', 'contrast_features.py', 'spatial_expert.py', 'model_families.py', 'requirements-xgboost.lock.txt']:
            path = iterate.ROOT/filename
            shutil.copy2(path, out/filename)
            protocol['sources'][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['xgboost_policy'] = 'Fixed depth4 recipe; inner20% logloss early stopping (cap1800, patience60), fresh full-outertrain refit. Dense onehot vocabulary learned only from training split. No outer validation for fitting or model selection. Fixed50/50 blend with v34, no fitted blend weights.'
        protocol['xgboost_version'] = xgboost.__version__
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return ORIGINAL_EVALUATE(name, matrices, train, dev, seed, out)


iterate.fit = fit
iterate.evaluate = evaluate

if __name__ == '__main__':
    import sys
    sys.modules['xgboost_candidate'] = sys.modules[__name__]
    XGBPredictor.__module__ = 'xgboost_candidate'
    BlendPredictor.__module__ = 'xgboost_candidate'
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_022')
    args = parser.parse_args()
    RUN_OUT = iterate.ROOT/'artifacts'/args.run
    if args.stage != 'screen':
        protocol = json.loads((RUN_OUT/'protocol.json').read_text())
        if protocol['xgboost_version'] != xgboost.__version__:
            raise RuntimeError('XGBoost version changed since frozen screen.')
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)
