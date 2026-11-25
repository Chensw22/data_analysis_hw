"""Fit an exact joint distribution over every outcome assignment in a booking."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def archive(raw, y, groups, folds):
    scores = np.full(len(y), np.nan)
    for k in np.unique(folds):
        with np.load(ROOT / f'artifacts/nov26_joint_party_tabicl/full_{k}.npz') as data:
            t, q = data['t'], data['q']
            np.testing.assert_array_equal(q, np.flatnonzero(folds == k))
            np.testing.assert_array_equal(t, np.flatnonzero(folds != k))
            np.testing.assert_array_equal(data['train_y'], y[t])
            np.testing.assert_array_equal(data['passenger'], raw.PassengerId.iloc[q])
            assert not np.isin(groups[t], groups[q]).any()
            scores[q] = data['direct']
    return scores.clip(.002, .998)


def make_parties(raw, groups, scores):
    cabin = raw.Cabin.fillna('?').to_numpy()
    asleep = raw.CryoSleep.eq(True).to_numpy()
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    age = raw.Age.to_numpy()
    costs = raw[FEES].fillna(0).to_numpy()
    active = costs > 0
    result = []
    for ids in raw.groupby(groups).indices.values():
        size = len(ids)
        if size < 2:
            continue
        assert size <= 8
        p = scores[ids]
        states = ((np.arange(1 << size)[:, None] >> np.arange(size)) & 1).astype(float)
        residual = states - p
        i, j = np.triu_indices(size, 1)
        a, b = ids[i], ids[j]
        charge_overlap = (active[a] & active[b]).sum(1) / np.maximum(1, (active[a] | active[b]).sum(1))
        pair = np.column_stack([np.ones(len(i)), (cabin[a] != '?') & (cabin[a] == cabin[b]),
                                asleep[a] & asleep[b], paid[a] & paid[b],
                                (asleep[a] & paid[b]) | (paid[a] & asleep[b]),
                                np.abs(age[a] - age[b]) <= 5, charge_overlap])
        personal = np.column_stack([np.ones(size), logit(p), asleep[ids], paid[ids],
                                    (p.sum() - p) / (size-1) - .5])
        features = np.column_stack([residual @ personal, (residual[:, i] * residual[:, j]) @ pair / (size-1)])
        result.append((ids, states, states @ logit(p), features))
    return result


def fit(parties, y, width):
    def objective(w):
        loss, gradient = 3 * (w @ w), 6 * w
        for ids, states, base, xs in parties:
            x = xs[:, :width]
            index = int((1 << np.arange(len(ids))) @ y[ids])
            energy = base + x @ w
            loss += logsumexp(energy) - energy[index]
            gradient += softmax(energy) @ x - x[index]
        return loss, gradient
    opt = minimize(objective, np.zeros(width), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=100, ftol=1.e-10))
    assert opt.success, opt.message
    return opt.x


def run():
    raw, y, groups, folds, views, _ = projection(include_full=True)
    nine, known, _, hist = champions()
    p = archive(raw, y, groups, folds)
    parties = make_parties(raw, groups, p)
    party_fold = np.array([folds[ids[0]] for ids, *_ in parties])
    assert all(np.all(folds[ids] == f) for (ids, *_), f in zip(parties, party_fold))
    pred = np.full((len(y), 3), np.nan)
    joint_scores = []
    for k in np.unique(folds):
        candidates = np.flatnonzero(party_fold == k)
        # Both calibration and test groups were held out of the base probability model.
        for a, b in GroupKFold(3, shuffle=True, random_state=8201).split(candidates, groups=candidates):
            train = [parties[i] for i in candidates[a]]
            unary, coupled = [fit(train, y, width) for width in (5, 12)]
            for ix in candidates[b]:
                ids, states, base, x = parties[ix]
                independent = base + x[:, :5] @ unary
                field = base + x @ coupled
                pred[ids, 0] = softmax(independent) @ states
                pred[ids, 1] = softmax(field) @ states
                # Diagnostic only: clamp companions to their realized outcomes and toggle the focal person.
                pattern = int((1 << np.arange(len(ids))) @ y[ids])
                bits = 1 << np.arange(len(ids))
                pred[ids, 2] = expit(field[pattern | bits] - field[pattern & ~bits])
                joint_scores.append([logsumexp(e) - e[pattern] for e in (independent, field)])
    multi = raw.groupby(groups).PassengerId.transform('size').to_numpy() > 1
    assert np.isfinite(pred[multi]).all() and np.isnan(pred[~multi]).all()
    report = dict(multiple_passengers=int(multi.sum()), bookings=len(parties),
                  joint_log_loss=np.mean(joint_scores, axis=0).tolist(), views={})
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    for tag, (valid, parent) in views.items():
        assert known[valid].all()
        same = valid & (nine == parent)
        masks = dict(common=same, awake_paid_miss=same & ~parent & raw.CryoSleep.eq(False).to_numpy() & paid,
                     sleeper_false_alarm=same & parent & raw.CryoSleep.eq(True).to_numpy() & ~paid,
                     new_order=same & ~np.isin(groups, groups[hist]))
        probabilities = pred >= .5
        actions = dict(independent=probabilities[:, 0] != parent, coupled=probabilities[:, 1] != parent,
                       newly_flipped=(probabilities[:, 1] != parent) & (probabilities[:, 0] == parent),
                       oracle_known_companions=probabilities[:, 2] != parent)
        report['views'][tag] = {name: {cohort: ticker(mask & multi, y != parent, action)
                                       for cohort, mask in masks.items()} for name, action in actions.items()}
    print('full', report['views']['full'], 'loss', report['joint_log_loss'], flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/exact_peer_outcome_field.json'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
