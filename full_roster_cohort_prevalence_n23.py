"""Compare within-cohort prevalence estimates to individual scores on held-out bookings."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_unexpected_peers_n23 import reference_score


def patient_cells(raw, group):
    _, ids, sizes = np.unique(group, return_inverse=True, return_counts=True)
    return pd.DataFrame({
        'origin': raw.HomePlanet, 'sleep': raw.CryoSleep, 'destination': raw.Destination,
        'side': raw.Cabin.str.split('/').str[2], 'solo': sizes[ids] == 1,
    }).astype('string').fillna('?')


def population(frame, y, t, q):
    # Each cell borrows from its immediate parents (all ways to omit one field).
    # Counts are always estimated on bookings outside the entire validation fold.
    all_keys = list(frame)
    cache = {(): np.repeat(y[t].mean(), len(q))}
    for length in range(1, len(all_keys)+1):
        for keys in combinations(all_keys, length):
            counts = frame.iloc[t].assign(target=y[t]).groupby(list(keys)).target.agg(['size','sum'])
            values = frame.iloc[q].loc[:, list(keys)].merge(counts.reset_index(), on=list(keys),
                                                            how='left', sort=False)[['size','sum']].fillna(0)
            parent = np.mean([cache[keys[:i]+keys[i+1:]] for i in range(length)], axis=0)
            cache[keys] = (values['sum'].to_numpy()+20*parent) / (values['size'].to_numpy()+20)
    return cache[tuple(all_keys)]


def fraction(p, prior, y):
    def loss(alpha):
        prob = ((1-alpha)*p + alpha*prior).clip(1e-9, 1-1e-9)
        return -np.log(np.where(y, prob, 1-prob)).sum() + 2*alpha**2
    fit = minimize_scalar(loss, bounds=(0.,1.), method='bounded')
    assert fit.success
    return min([0., 1., fit.x], key=loss)


def run():
    raw, y, group, fold = data()
    fields = patient_cells(raw, group)
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    p = reference_score(full, ref, names.index('balanced_full'))
    earth = ((fields.origin == 'Earth') & (fields.sleep == 'True')).to_numpy(bool)
    solo = (fields.solo == 'True').to_numpy(bool)
    output = {k:np.full(len(y), np.nan) for k in ['prevalence','quarter','learned','learned_by_cohort']}
    weights = []
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        assert not np.isin(group[t],group[q]).any()
        prior = population(fields, y, t, q)
        output['prevalence'][q] = prior
        output['quarter'][q] = .75*p[q] + .25*prior
        splits = GroupKFold(4, shuffle=True, random_state=462)
        for a,b in splits.split(q, groups=group[q]):
            assert not np.isin(group[q[a]],group[q[b]]).any()
            w = fraction(p[q[a]], prior[a], y[q[a]])
            output['learned'][q[b]] = (1-w)*p[q[b]] + w*prior[b]
            details = {}
            for value in (False, True):
                donors = a[earth[q[a]] == value]
                query = b[earth[q[b]] == value]
                v = fraction(p[q[donors]], prior[donors], y[q[donors]])
                output['learned_by_cohort'][q[query]] = (1-v)*p[q[query]]+v*prior[query]
                details[str(value)] = float(v)
            weights.append(dict(fold=int(f), pooled=float(w), by_cohort=details))
    report = {}
    for key, prob in output.items():
        assert np.isfinite(prob).all()
        delta = ((prob >= .5) == y).astype(int) - (ref == y)
        report[key] = dict(accuracy=float(((prob >= .5) == y).mean()), fixed=int((delta == 1).sum()),
                           harmed=int((delta == -1).sum()), earth_fixed=int((delta[earth] == 1).sum()),
                           earth_harmed=int((delta[earth] == -1).sum()),
                           solo_earth_net=int(delta[earth & solo].sum()),
                           scope_accuracy=float(np.mean((prob[earth] >= .5) == y[earth])),
                           fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,group))
    (ROOT/'reports/nov23_matched_learning_curve/cohort_prevalence.json').write_text(
        json.dumps(dict(results=report, weights=weights), indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    run()
