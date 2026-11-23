"""Decide when independent models can correct passengers left untouched by five pending submissions."""
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_service_pairs_n23 import REPORT
from service_pair_portfolio_n23 import replay

EXPERTS = [('service_distribution_distance', 'budget'), ('reference_passenger_outcomes', 'conditional'),
           ('passenger_bag_boost', 'joint'), ('subspace_discriminants', 'qda_average')]


def crossfit(odds, dissent, groups, folds, y, ref):
    # Learn on other bookings in the same outer fold. Every expert used for these
    # passengers was fitted without the corresponding outer-fold outcomes.
    outputs = {}
    target = y != ref
    support = dissent.any(1)
    for fold in np.unique(folds):
        idx = np.flatnonzero((folds == fold) & support)
        for a, b in GroupKFold(3, shuffle=True, random_state=955).split(idx, groups=groups[idx]):
            t, q = idx[a], idx[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            for name, x in [('margin', odds[:, :1]), ('all_scores', odds)]:
                scaler = StandardScaler().fit(x[t])
                lr = LogisticRegression(C=.25, max_iter=250).fit(scaler.transform(x[t]), target[t])
                outputs.setdefault(name, np.zeros(len(y), bool))[q] = lr.predict(scaler.transform(x[q]))
    return outputs


def main():
    weight = json.loads((REPORT / 'test_budget.json').read_text())['test']['change_150']['weight']
    row, y, frame, g, _ = load()
    wrong, right, hard = cohorts(row, y)
    history, plans = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        ref, old = replay(frame, g, row, y, seed, weight)
        with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as src:
            fold, base = src['fold'], src['base']
        available = ~old['union'] & (ref == (base >= .5))
        p = []
        for name, key in EXPERTS:
            with np.load(ART / f'nov23_{name}/{seed}.npz') as z:
                for k, v in [('row', row), ('y', y), ('ref', ref), ('fold', fold)]:
                    np.testing.assert_array_equal(z[k], v)
                p.append(z[key])
        p = np.column_stack(p)
        dissent = (p >= .5) != ref[:, None]
        sign = np.where(ref, -1., 1.)
        odds = sign[:, None] * logit(np.column_stack([base, p]).clip(.002, .998))
        vote = dissent.sum(1)
        rules = dict(any=vote>=1, majority=vote>=3, unanimous=vote==4)
        rules.update(crossfit(odds, dissent, g, fold, y, ref))
        baseline = ref == y
        previous = baseline ^ old['pair_plus_corroborated']
        evidence = {}
        for name, pick in rules.items():
            selected = pick & available
            h = previous ^ selected
            m = paired(h, previous)
            evidence[name] = dict(extra=m, total=paired(h, baseline),
                                  hard_fixed=int((selected & ~baseline & hard).sum()),
                                  hard_harmed=int((selected & baseline & hard).sum()))
            if seed != 'v131':
                history.setdefault(name, []).append(h)
        if seed != 'v131':
            history.setdefault('previous', []).append(previous)
        plans[seed] = dict(eligible=int(available.sum()), eligible_errors=int((~baseline & available).sum()),
                           rules=evidence)
        print(seed, {k: (v['extra']['fixed'], v['extra']['harmed']) for k,v in evidence.items()}, flush=True)
    before = np.stack(history.pop('previous'))
    stable = {}
    for k, h in history.items():
        h = np.stack(h)
        stable[k] = dict(hard_stable=int((hard & h.all(0)).sum()), hard_ever=int((hard & h.any(0)).sum()),
                         common_stable=int((wrong & h.all(0)).sum()), stable_harms=int((right & ~h.any(0)).sum()),
                         gain95_over_previous_pp=interval((h.astype(float)-before).mean(0), g))
    write(REPORT / 'untouched.json', dict(plans=plans, stable=stable, experts=EXPERTS))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
