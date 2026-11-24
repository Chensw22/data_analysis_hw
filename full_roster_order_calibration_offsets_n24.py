"""Estimate nested booking-held-out calibration offsets around the original Mitra probability."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_observable_twins_n24 import nested_keys


def fit_offset(key, fit, current, labels, precision):
    # Penalized log-likelihood is strictly concave for each independent category.
    size = int(key.max())+1
    ids = key[fit]
    fitted = np.zeros(size)
    for _ in range(30):
        prob = expit(current[fit]+fitted[ids])
        grad = np.bincount(ids, weights=labels[fit]-prob, minlength=size)-precision*fitted
        info = np.bincount(ids, weights=prob*(1-prob), minlength=size)+precision
        if np.max(np.abs(grad)) < 1e-7:
            break
        fitted += np.clip(grad/info, -1., 1.)
    else:
        raise ArithmeticError('Offset likelihood did not converge')
    return fitted[key]


def decisions(keys, p, labels, groups, fold, champion):
    base = p >= .5
    free = base == champion
    original = logit(p.clip(1e-6, 1-1e-6))
    out = {name: champion.copy() for name in ('direction', 'profiles', 'rescue_false_negatives',
                                              'shuffled_residual_control')}
    for f in np.unique(fold):
        ix = np.flatnonzero(fold == f)
        for a, b in GroupKFold(4, shuffle=True, random_state=348).split(ix, groups=groups[ix]):
            fit, q = ix[a], ix[b]
            assert not np.isin(groups[fit], groups[q]).any()
            start = original + fit_offset(base.astype(int), fit, original, labels, 4.)
            local = start.copy()
            control = start.copy()
            # Shuffle entire nested profiles among calibration passengers within
            # baseline direction and risk quintile. Query profiles stay unchanged.
            rng = np.random.default_rng(349)
            permutation = np.arange(len(p))
            band = np.searchsorted(np.quantile(p[fit], [.2, .4, .6, .8]), p)
            for s in np.unique(2*band[fit]+base[fit]):
                selected = fit[(2*band[fit]+base[fit]) == s]
                permutation[selected] = rng.permutation(selected)
            for profile in keys.values():
                keyed = 2*profile + base
                local += fit_offset(keyed, fit, local, labels, 4.)
                control += fit_offset(keyed[permutation], fit, control, labels, 4.)
            for name, z, eligible in [('direction', start, free), ('profiles', local, free),
                                       ('rescue_false_negatives', local, free & ~base),
                                       ('shuffled_residual_control', control, free)]:
                out[name][q] = np.where(eligible[q], z[q] >= 0, champion[q])
    return out


def run():
    _, old_y, old_group, rows, oldfold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    keys = nested_keys(raw)
    a = decisions({k: v[rows] for k, v in keys.items()}, old[:, 0], old_y, old_group, oldfold, champion)
    b = decisions(keys, full[:, names.index('mitra96')], y, group, fold, ref)
    common = ((full >= .5) != y[:, None]).all(1)
    hard = common[rows] & (champion != old_y)
    unseen = ~np.isin(group, old_group)
    report = {}
    for name, p in b.items():
        eligible = common & (p == y)
        report[name] = dict(historical=stats(a[name], champion, old_y, old_group, oldfold),
                            full=stats(p, ref, y, group, fold),
                            unseen=stats(p[unseen], ref[unseen], y[unseen], group[unseen], fold[unseen]),
                            common_false_negatives_recovered=int((eligible & y).sum()),
                            common_false_positives_recovered=int((eligible & ~y).sum()),
                            persistent_both=int((hard & (a[name] == old_y) & (p[rows] == old_y)).sum()))
        record = report[name]
        print(name, [record[k]['net'] for k in ('historical', 'full', 'unseen')],
              record['common_false_negatives_recovered'], record['common_false_positives_recovered'])
    path = ROOT/'reports/nov23_matched_learning_curve/order_calibration_offsets.json'
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    run()
