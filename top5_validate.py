"""Frozen paired development checks for top-five candidates; not a new holdout.

Historical83% gates stay unchanged. This writes a separate A/S eligibility
decision under the user's later stability-qualified online-test authorization.
"""
import argparse
import hashlib
import importlib
import json
import signal
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import audit_top5_references as audit

ROOT = Path(__file__).resolve().parent
MODULES = ['mitra_mirror_ensemble', 'mitra_lastblock_candidate']
POLICY = ROOT/'reports/top5_validation_review.json'
AUTH = ROOT/'reports/top5_authorization.json'
EXPECTED_SPLITS = [('random', 20261113), ('random', 20261114), ('group', 20261112)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_policy():
    if audit.SPLITS != EXPECTED_SPLITS:
        raise RuntimeError('Actual validation splits differ from the frozen policy.')
    authorization = json.loads(AUTH.read_text())
    if digest(POLICY) != authorization['eligibility_policy_sha256']:
        raise RuntimeError('Authorized validation policy changed.')
    policy = json.loads(POLICY.read_text())
    if policy['frozen_validation_protocol']['repeated_random_validation']['split_seeds'] != [20261113, 20261114]:
        raise RuntimeError('Unexpected repeated split policy.')
    return policy


def cluster_interval(row_difference, groups, replicates=2000, seed=20261119):
    """Paired, passenger-weighted whole-group sensitivity, with fixed OOF fits."""
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    sizes = np.bincount(inverse)
    totals = np.bincount(inverse, weights=np.asarray(row_difference, dtype=float))
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 100):
        sampled = rng.integers(0, len(sizes), size=(min(100, replicates-start), len(sizes)))
        draws.extend((totals[sampled].sum(axis=1)/sizes[sampled].sum(axis=1)).tolist())
    lower, upper = np.quantile(draws, [.025, .975])
    return dict(point_gain=float(np.mean(row_difference)), lower95=float(lower), upper95=float(upper),
        replicates=replicates, seed=seed, groups=len(sizes), unit='whole PassengerId group',
        limitation='Fixed adaptively selected development OOF; not independent or multiplicity-corrected evidence.')


def reference_probabilities(reference, key, dev, groups, assignment, y):
    path = ROOT/'artifacts/top5_reference_audit'/reference
    summary = json.loads((path/'summary.json').read_text())
    if summary['status'] != 'complete' or not summary['historical_run_unchanged']:
        raise RuntimeError('Incomplete reference audit.')
    if summary['reference'] != reference or summary['version'] != audit.REFERENCES[reference][1]:
        raise RuntimeError('Reference model identity differs from the frozen comparison.')
    entry = next(row for row in summary['audits'] if f"{row['kind']}_{row['seed']}" == key)
    artifact = ROOT/entry['oof_path']
    if digest(artifact) != entry['oof_sha256']:
        raise RuntimeError('Reference OOF changed.')
    with np.load(artifact) as saved:
        if not (np.array_equal(saved['dev_ids'], dev)
                and np.array_equal(saved['group_ids'].astype(str), np.asarray(groups).astype(str))
                and np.array_equal(saved['fold_id'], assignment)):
            raise RuntimeError('Reference and candidate row/fold alignment differs.')
        probability = saved['prob'].copy()
    if abs(accuracy_score(y, probability>=.5)-entry['accuracy']) > 1e-12:
        raise RuntimeError('Current labels do not reproduce the reference score.')
    return probability, dict(path=str(artifact.relative_to(ROOT)), sha256=entry['oof_sha256'],
        summary_sha256=digest(path/'summary.json'), protocol_sha256=digest(path/'protocol.json'))


def make_decision(screen_accuracy, results, intervals):
    if [(r['kind'], r['seed']) for r in results] != EXPECTED_SPLITS:
        raise RuntimeError('Exactly the three prespecified validation splits are required.')
    random = [r for r in results if r['kind'] == 'random']
    group = next(r for r in results if r['kind'] == 'group')
    if len(random) != 2:
        raise RuntimeError('Both predeclared random splits are required.')
    a_checks = dict(
        screen_in_A_band=screen_accuracy >= .8198159332758125-.01-1e-12,
        both_random_gains_over_Mitra_positive=all(r['vs_mitra']['gain'] > 1e-12 for r in random),
        mean_random_gain_over_Mitra_at_least_point1pp=np.mean([r['vs_mitra']['gain'] for r in random]) >= .001-1e-12,
        group_gain_over_Mitra_nonnegative=group['vs_mitra']['gain'] >= -1e-12,
        within_one_pp_of_v44_on_every_check=all(r['vs_v44']['gain'] >= -.01-1e-12 for r in results))
    s_checks = dict(
        screen_above_83=screen_accuracy > .83,
        repeat_mean_above_83=np.mean([r['accuracy'] for r in random]) > .83,
        both_random_gains_over_v44_positive=all(r['vs_v44']['gain'] > 1e-12 for r in random),
        mean_random_gain_over_v44_above_one_pp=np.mean([r['vs_v44']['gain'] for r in random]) > .01,
        repeat_mean_gain_over_Mitra_positive=np.mean([r['vs_mitra']['gain'] for r in random]) > 0,
        group_gain_over_both_positive=all(group[f'vs_{ref}']['gain'] > 0 for ref in ['v44','mitra']),
        repeat_v44_cluster_lower_bound_positive=intervals['random_mean_vs_v44']['lower95'] > 0,
        group_v44_cluster_lower_bound_positive=intervals['group_vs_v44']['lower95'] > 0)
    return dict(A_eligible=bool(all(a_checks.values())), S_supported=bool(all(a_checks.values()) and all(s_checks.values())),
        A_checks={k:bool(v) for k,v in a_checks.items()}, S_checks={k:bool(v) for k,v in s_checks.items()},
        claim='A is qualified experimental testing within the practical band, not proven superiority. S remains limited to this reused development protocol.')


def validate(module_name, run):
    policy = check_policy()
    candidate = importlib.import_module(module_name)
    run_out = ROOT/'artifacts'/run
    candidate.configure_runtime(run_out/'stability', stage='stability')
    candidate.assert_frozen(run_out)
    source_manifest = json.loads((run_out/'protocol.json').read_text())
    if source_manifest['sources'].get('top5_validate.py') != digest(__file__):
        raise RuntimeError('Validation implementation was not frozen before screening.')
    output = run_out/'stability'
    output.mkdir(exist_ok=False)
    import iterate
    matrices, dev, y, groups, passengers = audit.load_audit_data(iterate)
    table = pd.read_csv(run_out/'screen_results.csv')
    screen_accuracy = float(table.loc[table.version.eq(candidate.NAME), 'accuracy'].iloc[0])
    with np.load(run_out/f'{candidate.NAME}_seed20261112.npz') as screen:
        screen_probability = screen['prob']
        if (screen_probability.shape != y.shape or not np.isfinite(screen_probability).all()
                or not ((screen_probability>=0)&(screen_probability<=1)).all()
                or abs(accuracy_score(y, screen_probability>=.5)-screen_accuracy)>1e-12):
            raise RuntimeError('Screen CSV does not reproduce saved candidate OOF accuracy.')
    audit.json_write(output/'protocol.json', dict(candidate=candidate.NAME,
        original_screen_protocol_sha256=digest(run_out/'protocol.json'),
        validation_policy_sha256=digest(POLICY), authorization_sha256=digest(AUTH),
        own_source_sha256=digest(__file__), split_recipe=audit.SPLITS,
        development_ids_sha256=audit.row_hash(dev), target_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
        disclosure=audit.DISCLOSURE))
    rows, paired = [], {}
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Frozen stability time budget exceeded; incomplete evidence.')
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, candidate.SETTINGS['stability_budget_seconds'])
    try:
        for kind, seed in audit.SPLITS:
            key = f'{kind}_{seed}'
            probability, assignment = np.empty(len(y)), np.empty(len(y), dtype=np.int8)
            fold_rows = []
            for fold, (a, b) in enumerate(audit.audit_splits(kind, seed, y, groups)):
                tick = time.monotonic()
                p, model = candidate.fit(candidate.NAME, matrices, dev[a], y[a], dev[b], seed+fold)
                p = np.asarray(p)
                if p.shape != (len(b),) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
                    raise RuntimeError('Incomplete/invalid candidate probabilities.')
                probability[b], assignment[b] = p, fold
                fold_rows.append(dict(fold=fold, accuracy=float(accuracy_score(y[b],p>=.5)),
                    seconds=time.monotonic()-tick, train_ids_sha256=audit.row_hash(dev[a]),
                    validation_ids_sha256=audit.row_hash(dev[b])))
                if hasattr(model, 'release'):
                    model.release()
                del model
                print(candidate.NAME,key,'fold',fold+1,'accuracy',fold_rows[-1]['accuracy'],flush=True)
            correct = (probability>=.5)==y
            row = dict(kind=kind, seed=seed, accuracy=float(correct.mean()),
                log_loss=float(log_loss(y,probability)), folds=fold_rows)
            for reference in ['mitra','v44']:
                p, evidence = reference_probabilities(reference,key,dev,groups,assignment,y)
                reference_correct = (p>=.5)==y
                difference = correct.astype(float)-reference_correct.astype(float)
                paired[(key,reference)] = difference
                row[f'vs_{reference}'] = dict(gain=float(difference.mean()),
                    reference_accuracy=float(reference_correct.mean()),
                    corrections=int((correct & ~reference_correct).sum()),
                    regressions=int((~correct & reference_correct).sum()), source=evidence)
            np.savez(output/f'{key}.npz',prob=probability,fold_id=assignment,dev_ids=dev,
                group_ids=np.asarray(groups).astype(str),passenger_ids=np.asarray(passengers).astype(str))
            rows.append(row)
            audit.json_write(output/f'{key}.json',row)
            print(json.dumps(row),flush=True)
        candidate.assert_frozen(run_out)
        intervals = {}
        for reference in ['mitra','v44']:
            averaged = np.mean([paired[(f'random_{seed}',reference)] for seed in [20261113,20261114]],axis=0)
            intervals[f'random_mean_vs_{reference}'] = cluster_interval(averaged,groups)
            intervals[f'group_vs_{reference}'] = cluster_interval(paired[('group_20261112',reference)],groups)
        decision = make_decision(screen_accuracy,rows,intervals)
        decision.update(status='complete',candidate=candidate.NAME,screen_accuracy=screen_accuracy,
            results=rows,paired_cluster_sensitivity=intervals,total_seconds=time.monotonic()-started,
            validation_policy_sha256=digest(POLICY),authorization_sha256=digest(AUTH),
            original_screen_protocol_sha256=digest(run_out/'protocol.json'),legacy_gate_unchanged=True,
            disclosure=audit.DISCLOSURE)
        audit.json_write(output/'eligibility.json',decision)
        print('ELIGIBILITY',json.dumps(decision),flush=True)
    except Exception as error:
        audit.json_write(output/'failure.json',dict(status='failed_inconclusive',error=repr(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',choices=MODULES,required=True)
    parser.add_argument('--run',required=True)
    arguments = parser.parse_args()
    validate(arguments.module,arguments.run)
