"""Strict nested two-model fusion with one predeclared Cryo reliability modifier."""
import argparse
import gc
import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import iterate
BASE_EVALUATE = iterate.evaluate
import spatial_expert
from contrast_features import add_contrasts, ContrastPredictor
import tabpfn_architecture as architecture

BASE_FIT = spatial_expert.ORIGINAL_FIT
GLOBAL = 'v61_nested_global_fusion'
CONDITIONAL = 'v62_nested_cryo_fusion'
SEED_OFFSETS = [0, 100, 200]
META = dict(inner_folds=3, C=.1, logit_clip=1e-6, scaler='StandardScaler',
            threshold=.5, unknown='exact_global_fallback')
SOURCES = ['nested_conditional_fusion.py', 'contrast_features.py', 'spatial_expert.py',
           'tabpfn_architecture.py', architecture.MANIFEST]
iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False, model='catboost', **iterate.CAT),
    GLOBAL: dict(context=False, model='nested_fusion', conditional=False, meta=META,
                 seed_offsets=SEED_OFFSETS, foundation=architecture.ENSEMBLE,
                 checkpoints=architecture.CHECKPOINTS),
    CONDITIONAL: dict(context=False, model='nested_fusion', conditional=True, meta=META,
                      seed_offsets=SEED_OFFSETS, foundation=architecture.ENSEMBLE,
                      checkpoints=architecture.CHECKPOINTS),
}
iterate.BLEND = []
RUN_OUT = None
RUN_ID = None
DEADLINE = None
PAIR_CACHE = {}
PAYLOAD_CACHE = {}
# Frozen before the first full screen; confirmation runs two new random splits
# and one grouped split, so it needs a larger budget than one screen.
STAGE_SECONDS = dict(probe=900, screen=900, confirm=2700, export=1200)


class StageTimeout(RuntimeError):
    pass


def check_budget():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise StageTimeout('Frozen whole-stage budget exhausted; partial results are not eligible.')


def alarm_handler(signum, frame):
    raise StageTimeout('Frozen whole-stage budget exhausted; partial results are not eligible.')


def begin_stage(run_id, seconds=900):
    global RUN_ID, DEADLINE
    RUN_ID = str(run_id)
    DEADLINE = time.monotonic() + seconds
    PAIR_CACHE.clear()
    PAYLOAD_CACHE.clear()


def fingerprint(matrices, train_idx, y, predict_idx, seed):
    if RUN_ID is None:
        raise RuntimeError('Initialize a run-scoped cache before fitting.')
    train_idx = np.asarray(train_idx, dtype=np.int64)
    predict_idx = np.asarray(predict_idx, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if len(train_idx) != len(y) or len(np.unique(train_idx)) != len(train_idx):
        raise ValueError('Training indices and labels must align without repeated rows.')
    if np.intersect1d(train_idx, predict_idx).size:
        raise ValueError('Training and prediction rows overlap.')
    x, cats = matrices[False]
    h = hashlib.sha256()
    h.update(json.dumps(dict(run=RUN_ID, seed=int(seed), columns=list(x.columns), cats=cats,
        configs=iterate.CONFIGS, sources={p: hashlib.sha256((iterate.ROOT/p).read_bytes()).hexdigest()
                                        for p in SOURCES}), sort_keys=True).encode())
    for array in [train_idx, y, predict_idx]:
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    rows = x.iloc[np.concatenate([train_idx, predict_idx])]
    h.update(pd.util.hash_pandas_object(rows, index=True).to_numpy().tobytes())
    return h.hexdigest()


def cryo_sign(x):
    text = x.CryoSleep.astype('string')
    return np.where(text.eq('True').fillna(False), 1.,
                    np.where(text.eq('False').fillna(False), -1., 0.))


def meta_features(probabilities, raw, conditional):
    p = np.clip(np.asarray(probabilities), META['logit_clip'], 1-META['logit_clip'])
    z = np.log(p)-np.log1p(-p)
    if conditional:
        z = np.column_stack([z, cryo_sign(raw)*(z[:, 1]-z[:, 0])])
    return z


def fit_meta(probabilities, raw, y, conditional, seed):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=META['C'], max_iter=2000, random_state=seed))
    model.fit(meta_features(probabilities, raw, conditional), np.asarray(y))
    return model


def combine(global_meta, conditional_meta, probabilities, raw, conditional):
    global_p = global_meta.predict_proba(meta_features(probabilities, raw, False))
    if not conditional:
        return global_p
    result = conditional_meta.predict_proba(meta_features(probabilities, raw, True))
    unknown = cryo_sign(raw) == 0
    result[unknown] = global_p[unknown]
    return result


class SeedPredictor:
    def __init__(self, members):
        self.members = members

    def predict_proba(self, raw):
        return np.mean([m.predict_proba(raw) for m in self.members], axis=0)


def fit_v34(matrices, train_idx, y, predict_idx, seed):
    raw, cats = matrices[False]
    x = add_contrasts(raw)
    _, cat = BASE_FIT(iterate.BASELINE, {False: (x, cats)}, train_idx, y, predict_idx, seed)
    training = x.iloc[train_idx]
    mask = spatial_expert.SpatialPredictor.eligible(training)
    sub = training.loc[mask]
    spline = make_pipeline(SimpleImputer(strategy='median'),
        SplineTransformer(n_knots=5, degree=3, include_bias=False))
    side = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    spline.fit(sub[['CabinNumber']])
    side.fit(sub[['Side']])
    expert = LogisticRegression(C=.1, max_iter=1500, random_state=seed)
    model = spatial_expert.SpatialPredictor(cat, spline, side, expert, .5)
    expert.fit(model.design(sub), np.asarray(y)[mask.to_numpy()])
    return ContrastPredictor(model)


def pair_fit(matrices, train_idx, y, predict_idx, seed, keep=False):
    check_budget()
    key = fingerprint(matrices, train_idx, y, predict_idx, seed)
    if key in PAIR_CACHE:
        p, models = PAIR_CACHE[key]
        if keep and models is None:
            raise RuntimeError('Cannot reuse a probability-only fit as a retained model.')
        return p, models
    members = []
    for offset in SEED_OFFSETS:
        check_budget()
        members.append(fit_v34(matrices, train_idx, y, predict_idx, seed+offset))
    cat = SeedPredictor(members)
    x, cats = matrices[False]
    p_cat = cat.predict_proba(x.iloc[predict_idx])[:, 1]
    check_budget()
    foundation = architecture.ArchitecturePredictor(x.iloc[train_idx], y, cats, seed,
                                                   architecture.ENSEMBLE)
    p_pfn = foundation.predict_proba(x.iloc[predict_idx])[:, 1]
    check_budget()
    p = np.column_stack([p_cat, p_pfn])
    if not np.isfinite(p).all():
        raise RuntimeError('Invalid base-model predictions.')
    models = (cat, foundation) if keep else None
    PAIR_CACHE[key] = (p, models)
    if not keep:
        members = cat = foundation = None
        gc.collect()
    return p, models


def nested_payload(matrices, train_idx, y, predict_idx, seed):
    key = fingerprint(matrices, train_idx, y, predict_idx, seed)
    if key in PAYLOAD_CACHE:
        check_budget()
        return PAYLOAD_CACHE[key]
    train_idx = np.asarray(train_idx, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    oof = np.full((len(y), 2), np.nan)
    coverage = np.zeros(len(y), dtype=np.int64)
    audits = []
    for inner, (a, b) in enumerate(StratifiedKFold(3, shuffle=True,
                              random_state=seed+1000).split(train_idx, y)):
        if np.intersect1d(train_idx[a], predict_idx).size:
            raise ValueError('Outer validation rows leaked into inner training.')
        oof[b], _ = pair_fit(matrices, train_idx[a], y[a], train_idx[b], seed+inner)
        coverage[b] += 1
        audits.append(dict(inner_fold=inner, train_rows=len(a), oof_rows=len(b)))
        print('Nested pair finished', json.dumps(audits[-1]), flush=True)
    if not (coverage == 1).all() or not np.isfinite(oof).all():
        raise RuntimeError('Every outer-training row needs exactly one out-of-fold base prediction.')
    raw = matrices[False][0].iloc[train_idx]
    global_meta = fit_meta(oof, raw, y, False, seed)
    conditional_meta = fit_meta(oof, raw, y, True, seed)
    p, models = pair_fit(matrices, train_idx, y, predict_idx, seed, keep=True)
    payload = dict(oof=oof, outer_probabilities=p, models=models, global_meta=global_meta,
                   conditional_meta=conditional_meta, audits=audits)
    PAYLOAD_CACHE[key] = payload
    return payload


class FusionPredictor:
    def __init__(self, payload, conditional):
        self.cat, self.foundation = payload['models']
        self.global_meta = payload['global_meta']
        self.conditional_meta = payload['conditional_meta']
        self.conditional = conditional

    def predict_proba(self, raw):
        p = np.column_stack([self.cat.predict_proba(raw)[:, 1],
                             self.foundation.predict_proba(raw)[:, 1]])
        return combine(self.global_meta, self.conditional_meta, p, raw, self.conditional)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    check_budget()
    if name == iterate.BASELINE:
        return BASE_FIT(name, matrices, train_idx, y, predict_idx, seed)
    if name not in [GLOBAL, CONDITIONAL]:
        raise ValueError(name)
    payload = nested_payload(matrices, train_idx, y, predict_idx, seed)
    raw = matrices[False][0].iloc[predict_idx]
    p = combine(payload['global_meta'], payload['conditional_meta'],
                payload['outer_probabilities'], raw, name == CONDITIONAL)[:, 1]
    check_budget()
    if RUN_OUT is not None:
        key = fingerprint(matrices, train_idx, y, predict_idx, seed)[:12]
        meta = payload['conditional_meta' if name == CONDITIONAL else 'global_meta']
        (RUN_OUT/f'{name}_seed{seed}_{key}.json').write_text(json.dumps(dict(
            inner_folds=payload['audits'], outer_training_rows=len(train_idx),
            coefficients=meta[-1].coef_.tolist(), intercept=meta[-1].intercept_.tolist(),
            scaler_mean=meta[0].mean_.tolist(), scaler_scale=meta[0].scale_.tolist(),
            cached_pair_fits=len(PAIR_CACHE)), indent=2)+'\n')
    return p, FusionPredictor(payload, name == CONDITIONAL)


def record_protocol(out):
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    if 'nested_conditional_policy' in protocol:
        return
    architecture.verify_all()
    for filename in SOURCES:
        source = iterate.ROOT/filename
        shutil.copy2(source, out/source.name)
        protocol['sources'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = architecture.environment_lock()
    (out/'requirements-nested-fusion.lock.txt').write_text(lock)
    protocol['nested_fusion_environment_sha256'] = hashlib.sha256(lock.encode()).hexdigest()
    protocol['nested_conditional_policy'] = (
        'Outertrain inner3fold paired v44/v54 OOF only; same complete innerOOF rows and labels '
        'fit both scaler+LR C=.1 metas. Global uses two logits. Conditional adds only signedCryo '
        'times their difference: True+1/False-1/unknown0; unknown prediction exactly global. '
        'No Planet/other passthrough, state intercepts, weight search or threshold tuning. '
        'Nine inner and three outer paired refits shared by two metas, using run-scoped '
        'source/config/schema/index/feature-value/label/seed fingerprints. No historical development OOF. '
        'Frozen whole-stage budgets: screen900s, confirm2700s, export1200s. '
        'Alarm and budget checks reject partial stages. '
        'Built with PriorLabs-TabPFN.')
    path.write_text(json.dumps(protocol, indent=2)+'\n')


def evaluate(name, matrices, train, dev, seed, out):
    check_budget()
    record_protocol(out)
    return BASE_EVALUATE(name, matrices, train, dev, seed, out)


def invalidate_stage(out, stage, preexisting, error):
    outputs = {'screen': ['selection.json', 'screen_results.csv'],
               'export': ['submission.csv', 'test_probabilities.csv']}.get(stage, [])
    for filename in outputs:
        if filename not in preexisting:
            (out/filename).unlink(missing_ok=True)
    (out/'nested_fusion_stage_status.json').write_text(json.dumps(
        dict(stage=stage, status='failed', reason=str(error), partial_results_eligible=False), indent=2)+'\n')
    (out/'gate.json').write_text(json.dumps(dict(passed=False,
        reason='Nested fusion stage failed; incomplete results cannot authorize export.'))+'\n')


def probe():
    train, _, matrices, dev = iterate.dataset()
    y = train.Transported.iloc[dev].astype(int).to_numpy()
    a, b = next(StratifiedKFold(3, shuffle=True, random_state=iterate.SEEDS[0]).split(dev, y))
    started = time.monotonic()
    p_global, _ = fit(GLOBAL, matrices, dev[a], y[a], dev[b], iterate.SEEDS[0])
    count = len(PAIR_CACHE)
    p_conditional, _ = fit(CONDITIONAL, matrices, dev[a], y[a], dev[b], iterate.SEEDS[0])
    assert count == len(PAIR_CACHE) == 4
    unknown = cryo_sign(matrices[False][0].iloc[dev[b]]) == 0
    np.testing.assert_array_equal(p_global[unknown], p_conditional[unknown])
    assert np.isfinite(p_global).all() and np.isfinite(p_conditional).all()
    payload = next(iter(PAYLOAD_CACHE.values()))
    for column, filename in [(0, 'run_019/v44_seed_ensemble_seed20261112.npz'),
                             (1, 'run_026/v54_TabPFN_architecture_ensemble_seed20261112.npz')]:
        path = iterate.ROOT/'artifacts'/filename
        if path.exists():
            # Only checks unchanged base reproduction, never used to train either meta.
            old = np.load(path)['prob'][b]
            np.testing.assert_allclose(payload['outer_probabilities'][:, column], old, atol=1e-12, rtol=0)
    elapsed = time.monotonic()-started
    estimate = 3*elapsed+30
    result = dict(outer_training_rows=len(a), outer_query_rows=len(b), paired_base_fits=count,
        unknown_rows=int(unknown.sum()), unknown_bitwise_equal=True, valid_probabilities=True,
        first_outer_seconds=elapsed, projected_three_outer_seconds_with_30s_overhead=estimate,
        projected_within_screen_budget=estimate <= STAGE_SECONDS['screen'], accuracy_computed=False,
        note='One nested outer-fold resource/boundary probe; no score and no fold reduction.')
    print(json.dumps(result), flush=True)
    return result


iterate.fit, iterate.evaluate = fit, evaluate


def main():
    global RUN_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'screen', 'confirm', 'export'])
    parser.add_argument('--run', default='run_031')
    args = parser.parse_args()
    architecture.torch.set_num_threads(4)
    RUN_OUT = iterate.ROOT/'artifacts'/args.run if args.stage != 'probe' else None
    if args.stage == 'screen' and RUN_OUT.exists():
        raise RuntimeError('Run already exists; do not overwrite a completed or partial run.')
    if args.stage == 'confirm' and (RUN_OUT/'gate.json').exists():
        raise RuntimeError('Confirmation already recorded; do not overwrite it.')
    if args.stage == 'export' and (RUN_OUT/'submission.csv').exists():
        raise RuntimeError('Submission already exported; do not overwrite it.')
    preexisting = {p.name for p in RUN_OUT.iterdir()} if RUN_OUT and RUN_OUT.exists() else set()
    begin_stage(str(RUN_OUT) if RUN_OUT else f'probe-{time.monotonic_ns()}', STAGE_SECONDS[args.stage])
    previous = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, STAGE_SECONDS[args.stage])
    try:
        if args.stage == 'probe':
            probe()
        else:
            if args.stage in ['confirm', 'export']:
                architecture.verify_all()
                protocol = json.loads((RUN_OUT/'protocol.json').read_text())
                expected = protocol['nested_fusion_environment_sha256']
                if (hashlib.sha256(architecture.environment_lock().encode()).hexdigest() != expected
                    or hashlib.sha256((RUN_OUT/'requirements-nested-fusion.lock.txt').read_bytes()).hexdigest() != expected):
                    raise RuntimeError('Nested fusion environment changed since screening.')
            {'screen': iterate.screen, 'confirm': iterate.confirm, 'export': iterate.export}[args.stage](RUN_OUT)
            check_budget()
    except Exception as error:
        if RUN_OUT is not None and RUN_OUT.exists():
            invalidate_stage(RUN_OUT, args.stage, preexisting, error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    sys.modules['nested_conditional_fusion'] = sys.modules[__name__]
    for cls in [SeedPredictor, FusionPredictor]:
        cls.__module__ = 'nested_conditional_fusion'
    main()
