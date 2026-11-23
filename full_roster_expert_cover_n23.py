"""Compare greedy error coverage on other orders with blind majority decisions."""
import json

import numpy as np

from full_roster_booking_validation_n23 import ROOT, data
from full_roster_conditional_pool_n23 import banks
from full_roster_sleep_bill_factorial_n23 import VIEWS

REPORT = ROOT / 'reports/nov23_matched_learning_curve/expert_cover.json'


def scopes(votes, names, y):
    base = votes[:, names.index('mitra96')]
    compare = lambda keys: (votes[:, [names.index(k) for k in keys]] == base[:, None]).all(1)
    seven = (votes[:, :5] == base[:, None]).all(1) & compare(['without_near', 'only_near'])
    eleven = seven & compare([*VIEWS, 'balanced_full'])
    dissent = (votes != base[:, None]).sum(1)
    return dict(all=np.ones(len(y), bool), frequent_dissent=dissent >= 6,
                seven_agree=seven, eleven_agree=eleven)


def allocate(votes, y, base, pool):
    # Starting from Mitra, add the expert covering most as-yet-uncovered
    # training errors; use fewer total false changes only to resolve ties.
    errors = base != y
    uncovered = errors.copy()
    shifts = votes != base[:, None]
    harms = np.sum(shifts & ~errors[:, None], axis=0)
    ranking = []
    available = list(pool)
    while available:
        gains = (shifts & uncovered[:, None]).sum(0)
        best = max(available, key=lambda j: (gains[j], -harms[j], -j))
        ranking.append(best)
        available.remove(best)
        uncovered &= ~shifts[:, best]
    return ranking


def metrics(pred, oracle, base, truth, mask):
    hit = pred == truth
    before = base == truth
    repair = int((mask & hit & ~before).sum())
    harm = int((mask & ~hit & before).sum())
    possible = int((mask & oracle & ~before).sum())
    return dict(rows=int(mask.sum()), reference_errors=int((mask & ~before).sum()),
                fixed=repair, harmed=harm, net=repair-harm, oracle_fixes=possible,
                oracle_correct=int((mask & before).sum())+possible, vote_correct=int((mask & hit).sum()))


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    assert len(names) == len(set(names)) == 19
    small, large = half >= .5, full >= .5
    mi = names.index('mitra96')
    hb, base = small[:, mi], large[:, mi]
    group_large, group_small = scopes(large, names, y), scopes(small, names, y)
    # A separate descriptive scope permits a lone expert to dissent, as long as
    # that expert differs from native Mitra at both training sizes.
    group_large['recurrent_dissent'] = ((large != base[:, None]) & (small != hb[:, None])).any(1)
    assert sum(group_large['seven_agree'] & (base != y)) == 819
    assert sum(group_large['eleven_agree'] & (base != y)) == 329
    pool = [j for j in range(len(names)) if j != mi]
    unique = np.sum(large != base[:, None], axis=1) == 1
    singles = {}
    for j in pool:
        candidate = large[:, j]
        singles[names[j]] = {}
        for key, mask in group_large.items():
            info = metrics(candidate, candidate == y, base, y, mask)
            mine = unique & (candidate != base)
            info.update(unique_fixes=int((mine & mask & (candidate == y)).sum()),
                        unique_harms=int((mine & mask & (candidate != y)).sum()))
            singles[names[j]][key] = info

    comparisons = {}
    for policy in ('all', 'frequent_dissent'):
        ranks = {}
        sizes = [1, 3, 5, 9, len(pool)]
        predictions = {k: base.copy() for k in sizes}
        oracle = {k: np.zeros(len(y), bool) for k in sizes}
        for f in np.unique(fold):
            t = np.flatnonzero((fold % 2 == f % 2) & (fold != f) & group_small[policy])
            q = np.flatnonzero((fold == f) & group_large[policy])
            assert not np.isin(group[t], group[q]).any()
            order = allocate(small[t], y[t], hb[t], pool)
            ranks[int(f)] = [names[j] for j in order]
            for k in sizes:
                votes = large[q][:, order[:k]]
                positive = votes.sum(1)
                predictions[k][q] = np.where(2*positive == k, base[q], 2*positive > k)
                oracle[k][q] = (votes == y[q, None]).any(1)
        comparisons[policy] = dict(order_by_fold=ranks, sizes={})
        for k in sizes:
            by_scope = {scope: metrics(predictions[k], oracle[k], base, y, mask)
                        for scope, mask in group_large.items()}
            comparisons[policy]['sizes'][str(k)] = by_scope
    return dict(reference='mitra96', expert_count=len(pool), frequent_dissent_minimum=6,
                selection='Train on half-roster scores from other three folds of same parity; use '
                          'full-roster expert votes in query fold. Mitra is fallback and oracle option. '
                          'Majority uses selected experts only; ties preserve Mitra.',
                unique_definition='Only one of all 18 candidates changes the Mitra decision.',
                experts=singles, greedy=comparisons)


if __name__ == '__main__':
    result = run()
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, arm in result['greedy'].items():
        for size in ('1', '3', '5', '9', '18'):
            print(name, size, arm['sizes'][size]['all'])
