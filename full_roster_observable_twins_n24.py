"""Measure observational ambiguity and score booking-held-out analogues at nested resolutions."""
import json

import numpy as np
import pandas as pd

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities


def nested_keys(raw):
    cabin = raw.Cabin.str.split('/', expand=True)
    frame = raw[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']].copy()
    frame['deck'], frame['side'] = cabin[0], cabin[2]
    frame = frame.astype('string').fillna('missing')
    levels = {}
    def freeze(label):
        levels[label] = frame.groupby(list(frame), sort=False, dropna=False).ngroup().to_numpy()
    freeze('registration')
    costs = raw[BILLS]
    for name in BILLS:
        frame[name+'_state'] = np.where(costs[name].isna(), -1, (costs[name] > 0).astype(int))
    freeze('registration_and_service_states')
    frame['age_band'] = np.floor(raw.Age.fillna(-10)/10)
    for name in BILLS:
        frame[name+'_amount_band'] = np.floor(np.log2(1+costs[name].fillna(0)))
    freeze('age_and_amount_bands')
    frame['age'] = raw.Age.fillna(-1)
    for name in BILLS:
        frame[name+'_amount'] = costs[name].fillna(-1)
    freeze('exact_age_and_bills')
    frame['cabin_number'] = pd.to_numeric(cabin[1], errors='coerce').fillna(-1)
    freeze('all_personal_fields_without_id_or_name')
    return levels


def witnesses(keys, y, group, mask):
    population = pd.DataFrame(dict(key=keys, group=group, target=y.astype(int)))
    total = population.groupby('key').target
    count, pos = total.transform('size').to_numpy(), total.transform('sum').to_numpy()
    own = population.groupby(['key', 'group']).target
    others = count-own.transform('size').to_numpy()
    other_pos = pos-own.transform('sum').to_numpy()
    same = np.where(y, other_pos, others-other_pos)
    opposite = others-same
    summary = total.agg(['size', 'sum'])
    minimum_errors = np.minimum(summary['sum'], summary['size']-summary['sum']).sum()
    def counts(use):
        return dict(rows=int(use.sum()), no_other_order=int((use & (others == 0)).sum()),
                    same_only=int((use & (same > 0) & (opposite == 0)).sum()),
                    opposite_only=int((use & (same == 0) & (opposite > 0)).sum()),
                    both=int((use & (same > 0) & (opposite > 0)).sum()))
    return dict(min_errors_for_this_key=int(minimum_errors), distinct_keys=len(summary),
                all_rows=counts(np.ones(len(y), bool)), common=counts(mask),
                common_true=counts(mask & y), common_false=counts(mask & ~y))


def lookup(keys, y, group, rows, folds):
    scores = np.empty((len(rows), 2))
    for fold in np.unique(folds):
        held = folds == fold
        q = rows[held]
        t = rows[~held & ~np.isin(group[rows], group[q])]
        assert not np.isin(group[t], group[q]).any()
        n = [np.full(len(q), len(t))]
        pos = [np.full(len(q), y[t].sum())]
        for key in keys.values():
            n.append(np.bincount(key[t], minlength=key.max()+1)[key[q]])
            pos.append(np.bincount(key[t], weights=y[t], minlength=key.max()+1)[key[q]])
        # Use disjoint rings of increasingly similar observations. Each donor enters once.
        n, pos = np.array(n), np.array(pos)
        rings = np.vstack([n[:-1]-n[1:], n[-1]])
        hits = np.vstack([pos[:-1]-pos[1:], pos[-1]])
        assert (hits >= 0).all() and (hits <= rings).all()
        np.testing.assert_array_equal(rings.sum(0), n[0])
        np.testing.assert_array_equal(hits.sum(0), pos[0])
        posterior = np.full(len(q), .5)
        nearest = np.full(len(q), .5)
        for count, count_pos, disjoint, positive in zip(n, pos, rings, hits, strict=True):
            posterior = (8*posterior+positive)/(8+disjoint)
            enough = count >= 10
            nearest[enough] = (count_pos[enough]+1)/(count[enough]+2)
        scores[held] = np.column_stack([nearest, posterior])
    assert ((scores >= 0) & (scores <= 1)).all()
    return scores


def run():
    _, old_y, old_group, rows, oldfold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, reference)
    keys = nested_keys(raw)
    common = ((full >= .5) != y[:, None]).all(1)
    hard = common[rows] & (champion != old_y)
    unseen = ~np.isin(group, old_group)
    report = dict(witnesses={k: witnesses(v, y, group, common) for k, v in keys.items()}, models={})
    a, b = lookup(keys, y, group, rows, oldfold), lookup(keys, y, group, np.arange(len(y)), fold)
    for j, mode in enumerate(('nearest_with_ten', 'disjoint_rings')):
        for weight in (1., .25):
            pa = weight*a[:, j] + (1-weight)*old[:, 0] >= .5
            if weight < 1:
                pa = np.where((old[:, 0] >= .5) != champion, champion, pa)
            pb = weight*b[:, j] + (1-weight)*gate >= .5
            report['models'][f'{mode}_{weight}'] = dict(historical=stats(pa, champion, old_y, old_group, oldfold),
                full=stats(pb, reference, y, group, fold), unseen=stats(pb[unseen], reference[unseen], y[unseen], group[unseen], fold[unseen]),
                common_correct=int((common & (pb == y)).sum()), persistent_both=int((hard & (pa == old_y) & (pb[rows] == old_y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/observable_twins.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    for k, d in report['witnesses'].items():
        print(k, d['common'], d['min_errors_for_this_key'])
    for k, d in report['models'].items():
        print(k, d['full']['corrected'], d['full']['harmed'], d['common_correct'], d['persistent_both'])


if __name__ == '__main__':
    run()
