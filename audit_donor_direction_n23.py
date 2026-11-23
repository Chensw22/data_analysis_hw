"""Audit the incremental value of matched bills at equal flip counts within each direction."""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT / 'reports/nov23_donor_direction'
METHODS = ('always_negative', 'random', 'matched')


def counts(flips, errors):
    fixed = int((flips & errors).sum())
    harmed = int((flips & ~errors).sum())
    return dict(fixed=fixed, harmed=harmed, net=fixed - harmed)


def ranked_flips(ids, scores, passengers, budget, n):
    chosen = ids[np.lexsort((passengers[ids], -scores[ids]))[:budget]]
    result = np.zeros(n, dtype=bool)
    result[chosen] = True
    assert result.sum() == budget
    return result


def ranking(z, eligible, row, y, hard_zero):
    ref, base = z['ref'], z['base']
    errors = ref != y
    # A high signed score favors the opposite label regardless of reference direction.
    sign = np.where(ref, -1., 1.)
    random, matched = z['random'], z['matched']
    scores = {
        'baseline': sign * (base - .5),
        'random': sign * (random - .5),
        'matched': sign * (matched - .5),
        'increment': sign * (matched - random),
        'quarter_random': sign * (.75 * base + .25 * random - .5),
        'quarter_matched': sign * (.75 * base + .25 * matched - .5),
        'base_plus_increment': sign * (base + .25 * (matched - random) - .5),
    }
    result = {}
    selected = {}
    for direction, value in [('to_false', True), ('to_true', False)]:
        ids = np.flatnonzero(eligible & (ref == value))
        runs = {}
        for fraction in (.05, .10, .25, .50, 1.):
            budget = max(1, int(np.floor(len(ids) * fraction)))
            flags = {name: ranked_flips(ids, s, row, budget, len(y)) for name, s in scores.items()}
            if direction == 'to_true':
                selected[str(fraction)] = {name:mask for name,mask in flags.items()
                                           if name in ('quarter_random', 'quarter_matched', 'baseline')}
            runs[str(fraction)] = {
                'budget': budget,
                'hard_fixed_quarter_matched': int((hard_zero & errors & flags['quarter_matched']).sum()),
                'methods': {name: counts(v, errors) for name, v in flags.items()},
                'quarter_matched_only': counts(flags['quarter_matched'] & ~flags['quarter_random'], errors),
                'quarter_random_only': counts(flags['quarter_random'] & ~flags['quarter_matched'], errors),
            }
        result[direction] = dict(eligible=len(ids), reference_errors=int(errors[ids].sum()), budgets=runs,
                                 auc={k:float(roc_auc_score(errors[ids], s[ids])) for k,s in scores.items()})
    return result, selected


def main():
    row, y, frame, groups, _ = load()
    _, _, hard = cohorts(row, y)
    passengers = pd.read_csv(ROOT/'data/raw/train.csv', usecols=['PassengerId']).PassengerId.to_numpy()[row]
    zero = ~frame[BILLS].gt(0).any(axis=1).to_numpy()
    hard_zero = hard & zero
    assert zero.sum() == 2948 and hard_zero.sum() == 159
    correct, per_replay, selection_gains = {}, {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_zero_spend_analogues/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'], row)
            np.testing.assert_array_equal(z['y'], y)
            ref = z['ref']
            errors = ref != y
            eligible = zero & (ref == (z['base'] >= .5))
            for key in ('matched', 'random'):
                np.testing.assert_array_equal(np.isfinite(z[key]), eligible)
            predictions = {k:z['vote_' + k] for k in METHODS}
            for v in predictions.values():
                np.testing.assert_array_equal(v[~eligible], ref[~eligible])
            methods = {k:counts(v != ref, errors) for k,v in predictions.items()}
            direction, selected = ranking(z, eligible, row, y, hard_zero)
            per_replay[seed] = dict(eligible=int(eligible.sum()), direct=methods, direction=direction)
            if seed != 'v131':
                gain_if_changed = np.where(errors, 1., -1.)
                for budget, flags in selected.items():
                    weights = flags['quarter_matched'].astype(float)
                    for control in ('reference', 'quarter_random', 'baseline'):
                        diff = weights if control == 'reference' else weights - flags[control]
                        selection_gains.setdefault((budget, control), []).append(diff * gain_if_changed)
            correct[seed] = np.column_stack([v == y for v in predictions.values()])
    stable = np.logical_and.reduce([correct[str(s)] for s in SEEDS])
    signatures = {}
    for bits in product((False, True), repeat=len(METHODS)):
        group = hard_zero & (stable == bits).all(1)
        signatures[''.join(str(int(b)) for b in bits)] = dict(
            count=int(group.sum()), transported=int(y[group].sum()),
            uncommon_ids=sorted(passengers[group].tolist()) if group.sum() <= 10 else None)
    assert sum(s['count'] for s in signatures.values()) == hard_zero.sum()
    paired_intervals = {}
    for (budget, control), runs in selection_gains.items():
        gain = np.mean(runs, axis=0)
        paired_intervals.setdefault(budget, {})[control] = dict(net_mean=float(gain.sum()),
            accuracy_difference_pp=float(gain.mean()*100), interval95_pp=interval(gain, groups))
    report = dict(paired_to_true=paired_intervals, zero=int(zero.sum()), hard_zero=int(hard_zero.sum()), methods=METHODS,
                  stable_signatures=signatures, replays=per_replay)
    write(REPORT/'audit.json', report)
    print(signatures)
    for seed,r in per_replay.items():
        print(seed, r['direct'])
        for d,v in r['direction'].items():
            print(d,{k:round(x,3) for k,x in v['auc'].items()},
                  {fraction:[budget['methods'][name]['net'] for name in ('baseline','quarter_random','quarter_matched','base_plus_increment')]
                   for fraction, budget in v['budgets'].items()})


if __name__ == '__main__':
    main()
