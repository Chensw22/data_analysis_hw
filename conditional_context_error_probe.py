"""One fixed fold, two attribute-conditioned zero-shot support pools; no tuning."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None, 'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from sklearn.metrics import log_loss

import audit_top5_references as audit
import error_rescue_feature_probe as previous
import exaone_validation_data as data
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts

ROOT = Path(__file__).resolve().parent
NAME = 'Mitra_Earth_Cryo_conditional_context_zeroshot'
PLAN = 'reports/error_targeted_20261117/context_protocol.json'
OUT = ROOT / 'artifacts/error_targeted_20261117/context_fold0'
REPORT = ROOT / 'reports/error_targeted_20261117/context_result.json'
ANCHOR = ROOT / 'artifacts/error_rescue_20261117/feature_probe_fold0/compact13_fixed15_fold0.npz'
SEED, BUDGET = 20261112, 1200
STATES = ('True', 'False')


def route(frame, state):
    return (frame.HomePlanet.eq('Earth') & frame.CryoSleep.eq(state)).to_numpy(dtype=bool)


def predict_conditional(features, train_ids, y_train, query_ids, raw):
    """Only training labels enter models; query labels and error masks are absent."""
    train_ids, query_ids = np.asarray(train_ids), np.asarray(query_ids)
    if (len(train_ids) != 4636 or len(query_ids) != 2318
            or len(np.unique(train_ids)) != 4636 or len(np.unique(query_ids)) != 2318
            or np.intersect1d(train_ids, query_ids).size):
        raise ValueError('Exact disjoint original fold0 IDs required.')
    y_train = base.binary_labels(y_train, len(train_ids))
    p = np.asarray(raw).copy()
    if p.shape != (2318,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Complete raw reference required.')
    training, query = features.loc[train_ids], features.loc[query_ids]
    routed = np.zeros(len(query_ids), dtype=bool)
    logs, arrays = {}, {}
    for state in STATES:
        counts.require_false()
        sm, qm = route(training, state), route(query, state)
        if not sm.any() or not qm.any() or len(np.unique(y_train[sm])) != 2:
            raise RuntimeError('Both prespecified routes require support, queries and both support classes; no fallback.')
        sid, qid = train_ids[sm], query_ids[qm]
        if np.intersect1d(sid, qid).size or np.any(routed & qm):
            raise RuntimeError('Route membership overlap.')
        predictor = None
        tick = time.monotonic()
        try:
            predictor = base.fit_mitra(training.loc[sid], y_train[sm], sid, SEED, 'mps',
                recipe='zero-shot', query_chunk=2318, log_path=OUT / f'state_{state}_training.json')
            if (len(predictor.support) != len(sid) or not np.array_equal(predictor.labels, y_train[sm])
                    or predictor.preprocessor.columns != base.COMPACT):
                raise RuntimeError('Original compact13 and complete matching support required.')
            result = predictor.predict_proba(query.loc[qid])[:, 1]
            if result.shape != (len(qid),) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
                raise RuntimeError('Complete finite route predictions required.')
            p[qm] = result
            routed |= qm
            logs[state] = dict(training_rows=len(sid), query_rows=len(qid),
                train_ids_sha256=base.ids_hash(sid), query_ids_sha256=base.ids_hash(qid),
                support_class_counts=np.bincount(y_train[sm], minlength=2).tolist(),
                actual_features=predictor.preprocessor.n_features, original_columns=list(predictor.preprocessor.columns),
                checkpoint_sha256=base.WEIGHT_SHA, selected_steps=0,
                seconds=time.monotonic()-tick)
            arrays[f'support_ids_{state}'] = sid
            arrays[f'query_ids_{state}'] = qid
        finally:
            if predictor is not None:
                predictor.release()
            predictor = None
            base.release_memory()
    if not np.array_equal(p[~routed], np.asarray(raw)[~routed]):
        raise RuntimeError('Unrouted query probabilities must remain bitwise equal to raw.')
    return p, routed, logs, arrays


def run():
    if OUT.exists() or REPORT.exists():
        raise RuntimeError('No overwrite or resume.')
    plan = json.loads((ROOT / PLAN).read_text())
    if (plan['name'] != NAME or plan['seed'] != SEED or plan['states'] != list(STATES)
            or plan['budget_seconds'] != BUDGET or plan['fold_index'] != 0):
        raise RuntimeError('Frozen scope differs.')
    original, lock = counts.verify_reference()
    names = sorted(set(previous.OWN + [Path(__file__).name, PLAN,
        'reports/error_rescue_20261117/feature_probe_result.json',
        'reports/error_rescue_20261117/feature_probe_root_audit.json']
        + [n for n in original['source_hashes'] if n.startswith('third_party/mitra/')]))
    commit = previous.committed(names)
    OUT.mkdir(parents=True)
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Complete conditional-context diagnostic exceeded1200s.')
    handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    base.DEADLINE = started + BUDGET
    try:
        inputs = data.load_data()  # Original6954 labels only; old1739 labels are not read.
        splits, refs, evidence = data.random_references(inputs)
        a, b = splits[0]
        train_ids, query_ids = inputs['dev'][a], inputs['dev'][b]
        record = json.loads((ROOT / 'reports/error_rescue_20261117/feature_probe_result.json').read_text())
        if audit.file_hash(ANCHOR) != record['evidence_files'][str(ANCHOR.relative_to(ROOT))]:
            raise RuntimeError('Error-mask anchor hash differs.')
        previous.verify_sources(json.loads((ANCHOR.parent / 'freeze.json').read_text()))
        with np.load(ANCHOR, allow_pickle=False) as z:
            for name, expected in [('train_ids', train_ids), ('query_ids', query_ids), ('query_y', inputs['y'][b])]:
                np.testing.assert_array_equal(z[name], expected)
            masks = dict(all=np.ones(len(b), dtype=bool), five_common_wrong=z['common_wrong_mask'].copy(),
                         five_common_correct=z['common_correct_mask'].copy())
        if masks['five_common_wrong'].sum() != 292 or masks['five_common_correct'].sum() != 1736:
            raise RuntimeError('Fixed fold0 error-mask populations differ.')
        frozen = dict(source_commit=commit, sources={n:audit.file_hash(ROOT/n) for n in names},
            environment_sha256=hashlib.sha256(lock.encode()).hexdigest(), protocol=plan)
        frozen['sources'].update(inputs['evidence']); frozen['sources'].update(evidence)
        frozen['sources'][str(ANCHOR.relative_to(ROOT))] = audit.file_hash(ANCHOR)
        audit.json_write(OUT / 'freeze.json', frozen)
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS required; no fallback.')
        torch.set_num_threads(4)
        torch.mps.set_per_process_memory_fraction(.65)
        raw = refs['v71_Mitra2_zeroshot'][b]
        p, routed, logs, arrays = predict_conditional(inputs['features'], train_ids,
            inputs['y'][a].copy(), query_ids, raw)
        y = inputs['y'][b]
        query = inputs['features'].loc[query_ids]
        for state in STATES:
            masks[f'Earth_Cryo_{state}'] = route(query, state)
        result = dict(status='complete_single_fold_adaptive_diagnostic', name=NAME, source_commit=commit,
            accuracy=float(np.mean((p >= .5) == y)), log_loss=float(log_loss(y, p, labels=[0, 1])),
            raw_accuracy=float(np.mean((raw >= .5) == y)), raw_log_loss=float(log_loss(y, raw, labels=[0, 1])),
            paired={n:previous.paired(p, raw, y, mask) for n,mask in masks.items()},
            slices={n:dict(rows=int(mask.sum()), actual_positive=int(y[mask].sum()),
                correct=int(((p[mask]>=.5)==y[mask]).sum()),
                false_positive=int(((p[mask]>=.5)&(y[mask]==0)).sum()),
                false_negative=int(((p[mask]<.5)&(y[mask]==1)).sum())) for n,mask in masks.items()},
            routes=logs, unrouted_exact=True, seconds=time.monotonic()-started,
            interpretation='One reused fold; two routes form one candidate. Conditional support and refitted preprocessing change together; no stability or upload claim.')
        np.savez(OUT / 'predictions.npz', prob=p, raw=raw, y=y, train_ids=train_ids, query_ids=query_ids,
            routed_mask=routed, **arrays, **{n+'_mask':m for n,m in masks.items()})
        result['prediction_sha256'] = audit.file_hash(OUT / 'predictions.npz')
        previous.verify_sources(frozen)
        audit.json_write(REPORT, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        audit.json_write(OUT / 'failure.json', dict(status='failed_incomplete', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        base.release_memory()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == '__main__':
    run()
