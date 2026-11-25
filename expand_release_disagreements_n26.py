"""Test whether a positive candidate's changes generalize to a substantially wider set of passengers."""
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluate_nested_release_n26 import OUT, protocols, stages
from frozen_queue_shared_errors_n26 import paired
from held_expert_portfolios_n26 import DECK, outcome_scores, panel_predictions, sources
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def probability(x, labels, z):
    if len(np.unique(labels)) < 2:
        return np.full(len(z), (np.sum(labels)+1)/(len(labels)+2))
    return LogisticRegression(C=.08, max_iter=500).fit(x, labels).predict_proba(z)[:,1]


def run():
    raw, y, group, fold, known, versions = paired()
    panels = sources()[-1]
    bank = panel_predictions(panels['full'], 'full')
    evidence = outcome_scores(panels['full'], bank)
    parent = versions[:,0]
    signed = logit(evidence.clip(.005,.995)) * np.where(parent, -1., 1.)[:,None]
    fields = np.column_stack([parent, raw.HomePlanet.eq('Earth'), raw.CryoSleep.eq(True), raw[FEES].gt(0).any(axis=1),
                              raw.Cabin.str.startswith('G').fillna(False), raw.groupby(group).PassengerId.transform('size'),
                              np.quantile(signed, [.1,.25,.5,.75,.9], axis=1).T, signed[:,:18]])
    out = {}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    for key, src, _, t, q in protocols(panels, group, include_full=True):
        if not key.startswith('full_'):
            continue
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],t)
            np.testing.assert_array_equal(z['outer'],q)
            anchor, review, fare, final = stages(src, bank, t, enabled[t], z['risk'])
        np.testing.assert_array_equal(anchor,versions[t,6])
        assert not np.isin(group[t], group[q]).any() and np.unique(fold[np.r_[t,q]]).size == 1
        edited = final != parent[t]
        all_agree = (versions[q] == parent[q,None]).all(1)
        active = versions[q,-1] != parent[q]
        transform = StandardScaler().fit(fields[t])
        x, z = [transform.transform(fields[idx]).clip(-5,5) for idx in (t,q)]
        # One ranking imitates actual changes using model inputs alone. The other also
        # estimates whether such a change helped in separate historical bookings.
        propensity = probability(x, edited, z)
        chance = probability(x[edited], y[t][edited] != parent[t][edited], z)
        correct_prob = probability(x, y[t] != parent[t], z)
        rankings = dict(trace_similarity=propensity, expected_repair=propensity*(2*chance-1),
                        direct_error_probability=correct_prob)
        for name, rank in rankings.items():
            for factor in (1, 2):
                rule = np.zeros(len(q),bool)
                # Extrapolate separately in the two change directions, retaining previous candidates' decisions.
                for label in (False,True):
                    count = factor * int((active & (parent[q] == label)).sum())
                    eligible = np.flatnonzero(all_agree & (parent[q] == label))
                    if count:
                        high = eligible[np.argsort(-rank[eligible],kind='stable')[:count]]
                        rule[high] = True
                out.setdefault(name+f'_x{factor}',np.zeros(len(y),bool))[q] = rule
    final = versions[:,-1]
    same = known & (versions == final[:,None]).all(1)
    fresh = known & ~np.isin(group, group[panels['historical']['ids']])
    report = {}
    for n, r in out.items():
        assert not (r & ~same).any()
        candidate = final ^ r
        improved = (candidate == y).astype(int) - (final == y)
        report[n] = dict(all=ticker(known, final != y, r), untouched=ticker(same, final != y, r),
                         new_orders=ticker(fresh, final != y, r),
                         fold_gain=[int(improved[known & (fold==k)].sum()) for k in np.unique(fold)],
                         all_vs_champion=ticker(known, parent != y, candidate != parent))
        print(n,report[n]['all'],report[n]['new_orders'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/expand_release_disagreements.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
