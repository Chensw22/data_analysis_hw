"""Measure the incremental value of three new model families on the same held bookings."""
import argparse
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs, summary
from full_roster_booking_validation_n23 import OUT, ROOT, data
from nori_booking_regression_n26 import SHA
from outside_panel_coverage_n26 import archives

ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/joint_new_architectures.json'


def load_models(raw, y, groups, fold, context):
    agents, _ = archives(raw, y, groups, fold, True)
    existing = {n: agents[n] for n in ['Mitra2_adapted', 'TabDPT', 'LimiX2', 'mask_net_joint',
                                       'conditional_bill_flow', 'latent_class_mixture']}
    ids = raw.PassengerId.to_numpy(str)
    for name in ['cat', 'budget', 'tabicl', 'v44_without_name']:
        with np.load(OUT / f'{name}.npz') as z:
            for key, expected in [('passenger', ids), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(z[key], expected)
            existing[name] = z['p']
    with np.load(ART / 'nov26_sparse_boolean_clauses/eight_fold_oof.npz') as z:
        for key, expected in [('row', np.arange(len(y))), ('passenger', ids), ('y', y), ('fold', fold)]:
            np.testing.assert_array_equal(z[key], expected)
        clauses = z['full']
    nori = np.full(len(y), np.nan)
    capsules = np.full(len(y), np.nan)
    for k in np.unique(fold):
        t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert not np.isin(groups[q], groups[t]).any()
        with np.load(ART / f'nov26_nori/context_{context}/fold{k}.npz') as z:
            for key, value in [('t', t), ('q', q), ('ids', ids[q]), ('labels', y[z['context']])]:
                np.testing.assert_array_equal(z[key], value)
            assert len(np.unique(z['context'])) == (context or len(t)) and np.isin(z['context'], t).all()
            assert str(z['checkpoint']) == SHA
            nori[q] = z['p']
        scores = []
        for seed in [4537, 8912]:
            with np.load(ART / f'nov26_booking_capsules/fold{k}_{seed}.npz') as z:
                for key, value in [('t', t), ('q', q), ('ids', ids[q]), ('labels', y[t])]:
                    np.testing.assert_array_equal(z[key], value)
                scores.append(z['dynamic'])
        capsules[q] = np.mean(scores, axis=0)
    return existing, {f'Nori_{context or "full"}': nori, 'learned_clauses': clauses, 'routed_capsules': capsules}


def calibrate(scores, y, groups, fold, ordered):
    assert np.isfinite(scores).all()
    x = logit(scores.clip(.003, .997))
    pred = np.empty(len(y), bool)
    covered = np.zeros(len(y), bool)
    for k in np.unique(fold):
        ix = np.flatnonzero(fold == k)
        if ordered:
            blocks = np.array_split(np.unique(groups[ix]), 4)
            splits = [(ix[~np.isin(groups[ix], block)], ix[np.isin(groups[ix], block)]) for block in blocks]
        else:
            folds = GroupKFold(4, shuffle=True, random_state=412 + int(k))
            splits = [(ix[t], ix[q]) for t, q in folds.split(ix, groups=groups[ix])]
        for t, q in splits:
            assert not np.isin(groups[t], groups[q]).any() and not covered[q].any()
            model = LogisticRegression(C=.1, max_iter=500).fit(x[t], y[t])
            pred[q] = model.predict(x[q])
            covered[q] = True
    assert covered.all()
    return pred


def run(context):
    raw, y, groups, fold = data()
    _, cur, _ = inputs()
    old, new = load_models(raw, y, groups, fold, context)
    a, b = np.column_stack(list(old.values())), np.column_stack(list(new.values()))
    scopes = dict(all=np.ones(len(y), bool), known=cur['known'], common=cur['common'],
                  fresh=cur['new'] & cur['known'])
    ref = cur['parent']
    votes = (b >= .5) != ref[:, None]
    report = dict(existing=list(old), added=list(new), common_coverage={}, comparisons={})
    old_unanimous = ((a >= .5) == ref[:, None]).all(axis=1)
    report['previously_unopposed'] = summary(votes.any(axis=1), ref != y, cur['common'] & old_unanimous, fold)
    for j, name in enumerate(new):
        only = votes[:, j] & (votes.sum(axis=1) == 1)
        report['common_coverage'][name] = summary(only, ref != y, cur['common'], fold)
    for count in range(4):
        mask = votes.sum(axis=1) == count
        report['common_coverage'][f'exactly_{count}_challenges'] = summary(mask, ref != y, cur['common'], fold)
    for protocol, ordered in [('shuffled', False), ('ordered', True)]:
        baseline = calibrate(a, y, groups, fold, ordered)
        added = calibrate(np.column_stack([a, b]), y, groups, fold, ordered)
        results = {}
        for scope, use in scopes.items():
            results[scope] = dict(
                baseline_correct=int((use & (baseline == y)).sum()),
                augmented_correct=int((use & (added == y)).sum()),
                against_baseline=summary(added != baseline, baseline != y, use, fold))
            if scope != 'all':
                fresh_challenge = (added != ref) & (baseline == ref)
                results[scope]['new_challenges_only'] = summary(fresh_challenge, ref != y, use, fold)
                results[scope]['against_release'] = summary(added != ref, ref != y, use, fold)
        report['comparisons'][protocol] = results
    output = REPORT if context else REPORT.with_stem(REPORT.stem + '_full_context')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print({name: {scope: [r['baseline_correct'], r['augmented_correct']] for scope, r in res.items()}
           for name, res in report['comparisons'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=int, choices=[1536, 0], default=1536)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.context)
