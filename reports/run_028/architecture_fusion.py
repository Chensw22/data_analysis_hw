"""Predeclared equal fusion of the strongest seed ensemble and a different prior."""
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import iterate

RAW_EVALUATE = iterate.evaluate
import seed_ensemble
SEED_CONFIGS = dict(iterate.CONFIGS)
import tabpfn_architecture as architecture

NAME = 'v57_seed_architecture_fusion'
CAT_NAME = 'v44_seed_ensemble'
ARCH_NAME = architecture.NAME
ARCH_CONFIG = dict(iterate.CONFIGS[ARCH_NAME])
iterate.CONFIGS = {**SEED_CONFIGS, ARCH_NAME: ARCH_CONFIG,
                   NAME: dict(context=False, model='fixed_fusion', cat_weight=.5,
                              components=[CAT_NAME, ARCH_NAME])}
iterate.BLEND = []


class FusionPredictor:
    def __init__(self, cat, prior):
        self.cat, self.prior = cat, prior

    def predict_proba(self, x):
        return .5*self.cat.predict_proba(x)+.5*self.prior.predict_proba(x)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name not in [ARCH_NAME, NAME]:
        return seed_ensemble.fit(name, matrices, train_idx, y, predict_idx, seed)
    x, cats = matrices[False]
    prior = architecture.ArchitecturePredictor(x.iloc[train_idx], y, cats, seed, architecture.ENSEMBLE)
    p = prior.predict_proba(x.iloc[predict_idx])[:, 1]
    if name == ARCH_NAME:
        return p, prior
    q, cat = seed_ensemble.fit(CAT_NAME, matrices, train_idx, y, predict_idx, seed)
    return .5*p+.5*q, FusionPredictor(cat, prior)


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'fusion_policy' in protocol:
        return
    architecture.verify_all()
    for filename in ['architecture_fusion.py', 'seed_ensemble.py', 'contrast_features.py',
                     'spatial_expert.py', 'tabpfn_architecture.py', architecture.MANIFEST]:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = architecture.environment_lock()
    (out/'requirements-architecture.lock.txt').write_text(lock)
    protocol['architecture_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['fusion_policy'] = ('One predeclared50/50 probability average: v44 seed ensemble and '
        'v54 family-balanced TabPFN ensemble. No member/weight/threshold search. Development '
        'screen was motivated by lower v54 logloss but weights frozen before combined scores. '
        'All components refitted in this run; same-run OOF components reused for deterministic '
        'fusion only. Repeated/group/full-data fitting reconstructs the entire model graph.')
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    record_protocol(out)
    if name != NAME:
        return RAW_EVALUATE(name, matrices, train, dev, seed, out)
    components = [iterate.cross_validate(n, matrices, train, dev, seed, out)
                  for n in [CAT_NAME, ARCH_NAME]]
    p = np.mean([c[0] for c in components], axis=0)
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    folds = np.array([accuracy_score(y[b], p[b] >= .5) for _, b in
                     StratifiedKFold(3, shuffle=True, random_state=seed).split(dev, y)])
    seconds = sum(c[2] for c in components)
    np.savez(out/f'{name}_seed{seed}.npz', prob=p, fold_accuracy=folds, seconds=seconds)
    return p, folds, seconds


iterate.fit = fit
iterate.evaluate = evaluate

if __name__ == '__main__':
    sys.modules['architecture_fusion'] = sys.modules[__name__]
    FusionPredictor.__module__ = 'architecture_fusion'
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_028')
    args = parser.parse_args()
    architecture.torch.set_num_threads(4)
    out = iterate.ROOT/'artifacts'/args.run
    if args.stage != 'screen':
        architecture.check_environment(out)
    {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](out)
