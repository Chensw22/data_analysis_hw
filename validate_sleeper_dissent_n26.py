"""Check whether other bookings support accepting low sleeper outcome probabilities."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from evaluate_nested_release_n26 import OUT, protocols, stages
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import DECK, panel_predictions, sources
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def select_cut(values, errors):
    # Candidates are ranked by probability of the independently estimated recorded-sleep outcome.
    # Choose a boundary on whole score ties, shrinking a new exception's precision toward the cohort's rate.
    if not len(values):
        return -np.inf
    cuts = np.unique(np.r_[.5, np.quantile(values, [.02,.05,.10,.20,.40])])
    best = 0.
    cut = -np.inf
    for score in cuts:
        mask = values <= score
        correct = int(errors[mask].sum())
        n = int(mask.sum())
        precision = (correct + 8 * errors.mean()) / (n + 8)
        gain = n * (2 * precision - 1)
        if gain > best:
            cut, best = score, gain
    return cut


def run():
    raw, y, groups, folds, known, versions = paired()
    with np.load(ROOT/'artifacts/nov26_manifest_sleep_contrast/probabilities.npz') as z:
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId)
        np.testing.assert_array_equal(z['folds'], folds)
        score = z['scores']
        np.testing.assert_array_equal(z['eligible'], raw.CryoSleep.notna() & ~raw[FEES].gt(0).any(axis=1))
    sleeper = raw.CryoSleep.eq(True).to_numpy() & ~raw[FEES].gt(0).any(axis=1).to_numpy()
    assert np.isfinite(score[sleeper]).all()
    estimates = dict(direct=score[:,0], constant=score[:,1], variable=score[:,2],
                     median=np.median(score[:,:3],axis=1))
    # Require two entirely disjoint halves to each confirm positive precision above the local cutoff.
    panels = sources()[-1]
    bank = panel_predictions(panels['full'], 'full')
    results = {}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    for key, src, _, t, q in protocols(panels, groups, include_full=True):
        if not key.startswith('full_'):
            continue
        with np.load(OUT / f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'], t)
            np.testing.assert_array_equal(z['outer'], q)
            anchor, *last = stages(src, bank, t, enabled[t], z['risk'])
        np.testing.assert_array_equal(anchor, versions[t,6])
        same_t = sleeper[t] & versions[t,:7].all(1) & np.column_stack(last).all(1)
        fit_ids = t[same_t]
        evaluation = q[sleeper[q] & versions[q].all(1)]
        assert not np.isin(groups[fit_ids], groups[evaluation]).any()
        for n, p in estimates.items():
            cut = select_cut(p[fit_ids], ~y[fit_ids])
            pooled = p[evaluation] <= cut
            verified = []
            for a, b in GroupKFold(2, shuffle=True, random_state=9151).split(fit_ids, groups=groups[fit_ids]):
                threshold = select_cut(p[fit_ids[a]], ~y[fit_ids[a]])
                val = fit_ids[b][p[fit_ids[b]] <= threshold]
                ok = (~y[val]).sum() > y[val].sum()
                verified.append((p[evaluation] <= threshold) & ok)
            for mode, active in [('pooled', pooled), ('replicated', verified[0] & verified[1])]:
                results.setdefault(n+'_'+mode, np.zeros(len(y), bool))[evaluation] = active
    population = known & sleeper & versions.all(1)
    fresh = ~np.isin(groups,groups[panels['historical']['ids']])
    assert (population & ~y).sum() == 388
    report = {}
    for n, action in results.items():
        report[n] = dict(full=ticker(population, ~y, action), new_orders=ticker(population & fresh, ~y, action),
                         fold_net=[int(np.sum(action * np.where(y,-1,1) * (folds==k))) for k in np.unique(folds)])
        print(n,report[n],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/validate_sleeper_dissent.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
