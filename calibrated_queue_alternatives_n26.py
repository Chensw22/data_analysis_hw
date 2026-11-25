"""Learn when to override nine frozen candidates, using other bookings from the same held base fold."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from evaluate_nested_release_n26 import OUT, protocols, stages
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import DECK, panel_predictions, sources
from outside_panel_coverage_n26 import archives
from target_mix_expert_routing_n26 import ROOT, ticker


def design(w, p, fields):
    reference = w[:,-1]
    changes = w != reference[:,None]
    signed = logit(p.clip(.005,.995)) * np.where(reference, -1., 1.)[:,None]
    votes = (signed > 0).mean(1)
    return np.column_stack([changes, reference, signed, np.median(signed, axis=1), votes, fields])


def prob(x, wrong, z):
    if len(np.unique(wrong)) < 2:
        return np.full(len(z), (wrong.sum() + 1) / (len(wrong) + 2))
    return LogisticRegression(C=.08, max_iter=500).fit(x, wrong).predict_proba(z)[:,1]


def run():
    raw, y, g, folds, known, versions = paired()
    # EXAONE's archived partial coverage came from different base folds; omit it when fitting a router.
    ext, _ = archives(raw, y, g, folds, True)
    peers = np.column_stack([vals for name, vals in ext.items() if name != 'EXAONE_grouped'])
    members = ['Mitra2_adapted','TabDPT','LimiX2']
    assert set(members).issubset(ext)
    p = np.column_stack([ext[n] for n in members])
    fields = np.column_stack([raw.CryoSleep.eq(True), raw.HomePlanet.eq('Earth'),
                              raw.groupby(g).PassengerId.transform('size') == 1, raw.Cabin.str.startswith('G').fillna(False)])
    v = sources()[-1]
    hist = v['historical']['ids']
    full = v['full']
    bank = panel_predictions(full,'full')
    with np.load(DECK / 'deck_estimates.npz') as z:
        enabled = z['eligible']
    folds_out = {}
    seen = np.zeros(len(y), bool)
    for key, _, _, t, q in protocols(v, g, include_full=True):
        if not key.startswith('full_'):
            continue
        assert not np.isin(g[t], g[q]).any() and len(np.unique(folds[np.r_[t,q]])) == 1
        with np.load(OUT / f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'], t)
            np.testing.assert_array_equal(z['outer'], q)
            a, *rest = stages(full, bank, t, enabled[t], z['risk'])
        w = versions[t].copy()
        np.testing.assert_array_equal(a, w[:,6])
        w[:,7:] = np.column_stack(rest)
        wrong = w[:,-1] != y[t]
        wt = (w != w[:, -1:]).any(1)
        query = versions[q]
        wq = (query != query[:, -1:]).any(1)
        assert not seen[q].any()
        seen[q] = True
        signals = {}
        for label, forecasts in [('three', p), ('all', peers)]:
            x, xx = design(w, forecasts[t], fields[t]), design(query, forecasts[q], fields[q])
            risk = prob(x, wrong, xx)
            for cohort in (False, True):
                s, h = wt == cohort, wq == cohort
                cut = prob(x[s], wrong[s], xx[h]) if h.any() else np.array([])
                decision = np.zeros(len(q), bool)
                decision[h] = (risk[h] > .5) & (cut > .5)
                signals[label + ('_disputed' if cohort else '_unanimous')] = decision
        signals['both_unanimous'] = signals['three_unanimous'] & signals['all_unanimous']
        signals['at_least_one_unanimous'] = signals['three_unanimous'] | signals['all_unanimous']
        signals['both_disputed'] = signals['three_disputed'] & signals['all_disputed']
        for name, bits in signals.items():
            folds_out.setdefault(name, np.zeros(len(y), bool))[q] = bits
    np.testing.assert_array_equal(seen, known)
    parent = versions[:,-1]
    same = known & (versions == parent[:,None]).all(1)
    fresh = known & ~np.isin(g, g[hist])
    report = {}
    for name, action in folds_out.items():
        cohorts = dict(all=known, unanimous=same, new_orders=fresh,
                       sleeper=same & raw.CryoSleep.eq(True).to_numpy() & parent)
        entry = {n: ticker(mask, y != parent, action) for n, mask in cohorts.items()}
        entry['fold_net'] = [int((np.where(y != parent, 1, -1) * action * (folds == k)).sum()) for k in np.unique(folds)]
        report[name] = entry
        print(name, entry['all'], entry['new_orders'], flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/calibrated_queue_alternatives.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
