"""Fixed eight-cell residual calibration on strictly nested raw-Mitra OOF."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import audit_top5_references as audit
import error_overlap_diagnostic as aligned
import planet_mixture_experiment as nested_audit
import three_family_mean_screen as common

ROOT = Path(__file__).resolve().parent
REPORT = ROOT/'reports/spatial_offset_20261117'
OUT = ROOT/'artifacts/spatial_offset_20261117'
NESTED = ROOT/'artifacts/planet_mixture_20261117'


def cells(roster):
    if 'Transported' in roster: raise ValueError('Attributes only.')
    parts = roster.Cabin.str.split('/', expand=True).reindex(columns=[0, 1, 2])
    number = pd.to_numeric(parts[1], errors='coerce')
    rank = number.groupby([parts[0], parts[2]], dropna=False).rank(method='average', pct=True)
    bins = np.minimum(np.floor(rank.fillna(0).to_numpy()*4).astype(int), 3)
    eligible = (roster.HomePlanet.eq('Earth') & parts[0].eq('G') & roster.CryoSleep.eq(False)
                & parts[2].isin(['P', 'S']) & number.notna()).fillna(False).to_numpy(bool)
    result = np.full(len(roster), -1, dtype=np.int64)
    result[eligible] = bins[eligible] + 4*parts[2].eq('S').fillna(False).to_numpy(bool)[eligible]
    return result


def objective(beta, logits, y, index):
    z = logits+beta[index]
    loss = np.mean(np.logaddexp(0, z)-y*z)+.1*np.mean(beta**2)
    gradient = np.bincount(index, weights=expit(z)-y, minlength=len(beta))/len(y)+.2*beta/len(beta)
    return float(loss), gradient


def fit_offset(p, y, group, constant=False):
    p, y, group = np.asarray(p), np.asarray(y), np.asarray(group)
    if p.shape != y.shape or group.shape != y.shape or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Complete aligned training predictions and binary labels required.')
    if np.any((p < 0) | (p > 1)) or np.any((group < -1) | (group > 7)): raise ValueError('Invalid probability/cell.')
    mask = group >= 0
    if not mask.any(): raise ValueError('No eligible training rows.')
    index = np.zeros(mask.sum(), dtype=int) if constant else group[mask]
    n = 1 if constant else 8
    result = minimize(objective, np.zeros(n), args=(logit(np.clip(p[mask], 1e-7, 1-1e-7)), y[mask], index),
                      jac=True, method='L-BFGS-B', options={'maxiter':500, 'ftol':1e-12})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all():
        raise RuntimeError('Offset optimization did not converge: '+str(result.message))
    empty = np.bincount(index, minlength=n) == 0
    if not np.all(result.x[empty] == 0): raise RuntimeError('Empty cells must retain zero offset.')
    return dict(beta=result.x.tolist(), constant=constant, training_rows=int(mask.sum()),
                objective=float(result.fun), iterations=int(result.nit), cell_counts=np.bincount(index, minlength=n).tolist())


def predict(p, group, fitted):
    p, group = np.asarray(p), np.asarray(group)
    result = p.copy(); mask = group >= 0
    index = np.zeros(mask.sum(), dtype=int) if fitted['constant'] else group[mask]
    result[mask] = expit(logit(np.clip(p[mask], 1e-7, 1-1e-7))+np.asarray(fitted['beta'])[index])
    assert np.array_equal(result[~mask], p[~mask])
    return result


def main():
    if OUT.exists() or (REPORT/'result.json').exists(): raise FileExistsError('No overwrite/repeat.')
    started = time.monotonic()
    own = ['spatial_offset_candidate.py', 'test_spatial_offset_candidate.py', 'reports/spatial_offset_20261117/protocol.json',
           'error_overlap_diagnostic.py', 'audit_top5_references.py', 'exaone_validation_data.py',
           'three_family_mean_screen.py', 'prior_adapt_mean_screen.py', 'planet_mixture_experiment.py']
    common.require_committed(own)
    nested = json.loads((NESTED/'summary.json').read_text())
    if nested.get('status') != 'complete' or nested['completed_fits'] != 18 or nested['completed_outer_folds'] != 3:
        raise RuntimeError('Complete nested source required, regardless of its continuation outcome.')
    if audit.file_hash(NESTED/'summary.json') != audit.file_hash(ROOT/'reports/planet_mixture_20261117/summary.json'):
        raise RuntimeError('Nested evidence mismatch.')
    common.require_committed(['reports/planet_mixture_20261117/summary.json'])
    origin = json.loads((NESTED/'freeze.json').read_text())
    nested_audit.assert_frozen(origin)
    if audit.file_hash(NESTED/'freeze.json') != audit.file_hash(ROOT/'reports/planet_mixture_20261117/freeze.json'):
        raise RuntimeError('Nested freeze mismatch.')
    if json.loads((NESTED/'protocol.json').read_text())['recipe'] != nested_audit.RECIPE:
        raise RuntimeError('Nested recipe mismatch.')
    evidence_nested = {str((NESTED/name).relative_to(ROOT)):audit.file_hash(NESTED/name)
                       for name in ['summary.json', 'freeze.json', 'protocol.json']}
    for name, digest in nested['result_files_sha256'].items():
        if audit.file_hash(NESTED/name) != digest: raise RuntimeError('Nested artifact changed: '+name)
        evidence_nested[str((NESTED/name).relative_to(ROOT))] = digest
    inputs, splits, folds, probabilities, evidence = aligned.load_authenticated()
    dev, y = inputs['dev'], inputs['y']
    train, test, _ = audit.read_development_only(ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', dev)
    group = cells(pd.concat([train, test], ignore_index=True))[dev]
    base = probabilities['rawMitra']; preds = {'raw':base.copy(), 'constant':np.full(len(y), np.nan), 'spatial':np.full(len(y), np.nan)}
    with np.load(NESTED/'oof.npz') as z:
        assert np.array_equal(z['dev_ids'], dev) and np.array_equal(z['fold_id'], folds)
        assert np.array_equal(z['raw'], base)
    records = []; OUT.mkdir(parents=True)
    freeze = dict(sources={name:audit.file_hash(ROOT/name) for name in own},
                  nested_summary_sha256=audit.file_hash(NESTED/'summary.json'), source_commit=common.require_committed(own), evidence=evidence)
    audit.json_write(REPORT/'freeze.json', freeze)
    for outer, (a, b) in enumerate(splits):
        row = json.loads((NESTED/f'outer{outer}.json').read_text())
        path = NESTED/f'outer{outer}_inner_oof.npz'
        if audit.file_hash(path) != row['inner_oof_sha256']: raise RuntimeError('Inner cache changed.')
        assert row['status'] == 'complete' and row['outer_fold'] == outer
        assert row['outer_train_ids_sha256'] == audit.row_hash(dev[a])
        assert row['outer_validation_ids_sha256'] == audit.row_hash(dev[b])
        assert row['inner_split_seed'] == 20261117+outer and row['cached_outer_fit_seed'] == 20261112+outer
        evidence_nested[str(path.relative_to(ROOT))] = audit.file_hash(path)
        with np.load(path) as z:
            assert np.array_equal(z['outer_train_ids'], dev[a])
            inner = z['raw'].copy(); expected = np.full(len(a), -1)
            for k, (ia, ib) in enumerate(StratifiedKFold(3, shuffle=True, random_state=20261117+outer).split(a, y[a])):
                assert not np.intersect1d(dev[a][ia], dev[b]).size
                expected[ib] = k
                job_dir = NESTED/'jobs'/f'outer{outer}_inner{k}_raw'
                job = json.loads((job_dir/'job.json').read_text())
                worker = json.loads((job_dir/'worker_result.json').read_text())
                assert job['seed'] == 20261117+100*outer+k
                assert job['kind'] == 'raw' and job['outer_fold'] == outer and job['inner_fold'] == k
                assert audit.file_hash(job_dir/'rows.npz') == job['rows_sha256']
                with np.load(job_dir/'rows.npz') as packet:
                    assert set(packet.files) == {'train_ids','query_ids','outer_train_ids','outer_validation_ids','train_labels'}
                    for key, value in [('train_ids',dev[a][ia]),('query_ids',dev[a][ib]),('outer_train_ids',dev[a]),
                                       ('outer_validation_ids',dev[b]),('train_labels',y[a][ia])]:
                        assert np.array_equal(packet[key], value)
                    nested_audit.validate_rows(packet['train_ids'],packet['query_ids'],packet['outer_train_ids'],packet['outer_validation_ids'],packet['train_labels'])
                probability = nested_audit.verify_prediction(job_dir,worker,job,dev[a][ia],dev[a][ib],y[a][ia])
                assert np.array_equal(probability, inner[ib])
                metadata = json.loads((job_dir/'metadata.json').read_text())
                common.check_fold_log(metadata,'mitra',dev[a][ia],dev[a][ib],y[a][ia],job['seed'])
                for name in ['rows.npz','job.json','worker_result.json','metadata.json','prediction.npz']:
                    file = job_dir/name; evidence_nested[str(file.relative_to(ROOT))] = audit.file_hash(file)
            assert np.array_equal(z['inner_fold_id'], expected)
        fits = {}
        for key in ['constant', 'spatial']:
            fitted = fit_offset(inner, y[a], group[a], constant=key=='constant')
            preds[key][b] = predict(base[b], group[b], fitted)
            fits[key] = fitted
        records.append(dict(fold=outer, train_ids_sha256=audit.row_hash(dev[a]), validation_ids_sha256=audit.row_hash(dev[b]),
                            inner_oof_sha256=audit.file_hash(path), fits=fits))
        if time.monotonic()-started > 300: raise TimeoutError('Fixed CPU analysis budget exceeded.')
    scores = {key:dict(accuracy=float(accuracy_score(y, p>=.5)), log_loss=float(log_loss(y, p)),
        eligible_log_loss=float(log_loss(y[group>=0], p[group>=0])),
        fold_accuracy=[float(accuracy_score(y[b], p[b]>=.5)) for _, b in splits],
        corrected_vs_raw=int(((base>=.5)!=y)[(p>=.5)==y].sum()),
        regressed_vs_raw=int(((base>=.5)==y)[(p>=.5)!=y].sum())) for key, p in preds.items()}
    spatial, raw, constant = scores['spatial'], scores['raw'], scores['constant']
    checks = dict(gain_ge_point1pp=spatial['accuracy']-raw['accuracy']>=.001-1e-12,
        accuracy_ge_constant=spatial['accuracy']>=constant['accuracy'],
        logloss_le_raw=spatial['log_loss']<=raw['log_loss'], eligible_logloss_le_raw=spatial['eligible_log_loss']<=raw['eligible_log_loss'],
        logloss_le_constant=spatial['log_loss']<=constant['log_loss'],
        at_least_two_folds_ge_raw=sum(a>=b for a,b in zip(spatial['fold_accuracy'], raw['fold_accuracy']))>=2)
    result = dict(status='complete', metrics=scores, fold_fits=records, eligible_rows=int((group>=0).sum()),
        continuation=dict(checks=checks, passed=all(checks.values())),
        automatic_upload=False, adaptive_development=True, old_holdout_labels_read=False,
        seconds=time.monotonic()-started, source_sha256=freeze['sources'])
    freeze['nested_evidence_sha256'] = evidence_nested
    audit.json_write(REPORT/'freeze.json', freeze)
    nested_audit.assert_frozen(origin)
    nested_audit.verify_hashes({**freeze['sources'], **evidence, **evidence_nested})
    if time.monotonic()-started > 300: raise TimeoutError('Fixed CPU analysis budget exceeded.')
    np.savez(OUT/'oof.npz', dev_ids=dev, fold_id=folds, cells=group, **preds)
    result['oof_sha256'] = audit.file_hash(OUT/'oof.npz')
    for folder in [OUT, REPORT]: audit.json_write(folder/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
