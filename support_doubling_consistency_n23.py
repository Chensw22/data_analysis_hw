"""Check individual models and disjoint training-block growth on the same held-out passengers."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import family_scores, panel
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_support_trajectories_n23 import LEVEL, OUT, SUBSETS, fit_bank
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import compact_features
from matched_booking_learning_curve_n23 import portions

REPORT = ROOT/'reports/nov23_support_doubling_consistency'


def growth(bank):
    scores = dict(zip(SUBSETS, logit(bank.clip(1e-7, 1-1e-7))))
    early, late = [], []
    full = scores[(0, 1, 2, 3)]
    for i,j in combinations(range(4), 2):
        pair = scores[i,j]
        early.append(pair - .5*(scores[i,]+scores[j,]))
        if i == 0:
            complement = tuple(sorted(set(range(4))-{i,j}))
            late.append(full - .5*(pair+scores[complement]))
    early, late = np.stack(early), np.stack(late)
    steps = np.vstack([early, late])
    # A common signed shift exists only if every lower- and upper-level split agrees.
    shared = np.maximum(steps.min(0), 0) + np.minimum(steps.max(0), 0)
    return early, late, shared


def get_bank(seed, row, y, groups, fold):
    path = OUT/f'{seed}.npz'
    if seed == 'v131' and not path.exists():
        raw = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c: c not in ('Name', 'Transported'))
        bank, _ = fit_bank(compact_features(raw.iloc[row]), groups, y, fold, 20261123)
        np.savez_compressed(path, row=row, y=y, fold=fold, bank=bank, subsets=LEVEL)
    with np.load(path) as data:
        for k,v in [('row', row), ('y', y), ('fold', fold), ('subsets', LEVEL)]:
            np.testing.assert_array_equal(data[k], v)
        return data['bank']


def larger_training_sets(row, y, groups, scopes):
    splits = list(portions(groups))
    results = {}
    for name in ['tabicl', 'v44']:
        with np.load(ART/f'nov23_matched_learning_curve/{name}.npz') as bank:
            np.testing.assert_array_equal(bank['row'], row)
            np.testing.assert_array_equal(bank['y'], y)
            for f, small, large, query in splits:
                np.testing.assert_array_equal(bank['sizes'][f], [len(small), len(large), len(query)])
                np.testing.assert_array_equal(query, np.flatnonzero(bank['fold']==f))
                assert not np.intersect1d(groups[large], groups[query]).size
            before, after = bank['small'], bank['large']
        shift = (after-before)*(2*y.astype(int)-1)
        correct_small, correct_large = (before>=.5)==y, (after>=.5)==y
        results[name] = {k: dict(**paired(correct_large[m], correct_small[m]),
                                 correct_before=int(correct_small[m].sum()), correct_after=int(correct_large[m].sum()),
                                 moved_toward=int((shift[m]>0).sum()), median_true_probability_change=float(np.median(shift[m])))
                         for k,m in scopes.items()}
    return results


def main():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    paid_solo = (frame.BookingSize.to_numpy()==1) & frame[BILLS].gt(0).any(axis=1).to_numpy()
    replay, unanimous = {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, base, ref, families = family_scores(seed, row, y, frame)
        if seed != 'v131':
            unanimous.append(~ref & (np.column_stack(list(families.values())) < .5).all(1))
        replay[seed] = base, ref, get_bank(seed, row, y, groups, fold)
    target = hard & y & paid_solo & np.logical_and.reduce(unanimous)
    assert (hard.sum(), target.sum()) == (372, 86)
    scopes = dict(all=np.ones(len(y), bool), hard=hard, paid86=target, correct=right)
    plans, history = {}, {}
    for seed, (base, ref, bank) in replay.items():
        early, late, shared = growth(bank)
        h = (bank >= .5) == y
        margin = (2*y.astype(int)-1)[None, :]
        bands = {}
        for name, m in scopes.items():
            bands[name] = dict(passengers=int(m.sum()), sizes={})
            for size in range(1, 5):
                ok = h[LEVEL == size][:, m]
                # Average of individual accuracies, not accuracy after combining their probabilities.
                bands[name]['sizes'][size] = dict(mean_individual_accuracy=float(ok.mean()),
                                                   at_least_one_correct=int(ok.any(0).sum()), all_correct=int(ok.all(0).sum()))
            bands[name].update(all_steps_toward_truth=int(((shared*margin[0]>0)&m).sum()),
                               all_steps_away=int(((shared*margin[0]<0)&m).sum()),
                               early_late_direction_agreement=float((np.sign(early.mean(0)[m])==np.sign(late.mean(0)[m])).mean()))
        metrics = {}
        for key, delta in [('median_late', np.median(late, axis=0)), ('all_steps', shared)]:
            p = expit(logit(base.clip(1e-7, 1-1e-7)) + delta*np.log(1.5)/np.log(2))
            vote = np.where(ref == (base >= .5), p >= .5, ref)
            hit = vote == y
            metrics[key] = dict(**paired(hit, ref==y), hard_hits=int((hard & hit).sum()), target_hits=int((target & hit).sum()))
            if seed != 'v131':
                history.setdefault(key, []).append(hit)
        # Investigate the correct dissenting votes observed in the small models.
        # All four models train on disjoint quarters; exclude the focal booking from each.
        count = ((bank[LEVEL == 1] >= .5) != ref[None, :]).sum(0)
        for key, minimum in [('one_quarter', 1), ('majority_quarters', 3), ('unanimous_quarters', 4)]:
            flip = (count >= minimum) & (ref == (base >= .5))
            hit = (ref ^ flip) == y
            metrics[key] = dict(**paired(hit, ref == y), hard_hits=int((hard & hit).sum()),
                                target_hits=int((target & hit).sum()))
            if seed != 'v131':
                history.setdefault(key, []).append(hit)
        plans[seed] = dict(bands=bands, metrics=metrics)
        print(seed, metrics, flush=True)
    _, _, _, _, previous, _, _ = panel(row, y, frame)
    earlier = np.logical_or.reduce(list(previous.values()))
    assert (hard & earlier).sum() == 87
    stable = {k: dict(hard_ever=int((hard & np.any(h, axis=0)).sum()), hard_stable=int((hard & np.all(h, axis=0)).sum()),
                      stable_outside_prior_87=int((hard & ~earlier & np.all(h, axis=0)).sum()),
                      stable_damage=int((right & ~np.any(h, axis=0)).sum())) for k,h in history.items()}
    larger = larger_training_sets(row, y, groups, scopes)
    write(REPORT/'results.json', dict(plans=plans, stable=stable, larger_training_sets=larger))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
