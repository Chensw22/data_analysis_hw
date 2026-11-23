"""Evaluate extra labeled training orders on unchanged development passengers."""
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL, compact_features
from support_doubling_consistency_n23 import LEVEL, get_bank

LABEL = 'expanded_training_roster'


def fitting(x, labels, t, q):
    model = CatBoostClassifier(iterations=350, depth=6, learning_rate=.065, verbose=False,
                               random_seed=1923, thread_count=3, allow_writing_files=False)
    model.fit(x.iloc[t], labels.astype(int), cat_features=CATEGORICAL)
    return model.predict_proba(x.iloc[q])[:, 1]


def eight_fold(row, y):
    arrays = []
    for name in ['v44', 'tabicl']:
        with np.load(ART / f'nov23_matched_learning_curve/{name}.npz') as z:
            np.testing.assert_array_equal(row, z['row'])
            np.testing.assert_array_equal(y, z['y'])
            if arrays:
                np.testing.assert_array_equal(fold, z['fold'])
            fold = z['fold']
            arrays.append(z['large'])
    base = np.mean(arrays, axis=0)
    return fold, base, base >= .5


def compare_eight():
    row, y, _, groups, _ = load()
    _, _, hard, _ = hard_cohorts(row, y)
    fold, base, _ = eight_fold(row, y)
    files = {'cat': 'nov23_expanded_training_roster/eight',
             'tabicl': 'nov23_expanded_training_roster/eight_tabicl',
             'v44': 'nov23_matched_learning_curve/v44'}
    pairs = {}
    for name, path in files.items():
        with np.load(ART / (path + '.npz')) as z:
            for k, val in [('row', row), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(z[k], val)
            if name == 'v44':
                v44 = z['large']
            else:
                pairs[name] = np.stack([z['initial'], z['expanded']])
    pairs['cat_tabicl'] = (pairs['cat'] + pairs['tabicl']) / 2
    pairs['v44_tabicl'] = (v44[None, :] + pairs['tabicl']) / 2
    np.testing.assert_allclose(base, pairs['v44_tabicl'][0], rtol=0, atol=0)
    result = {}
    for name, (p, q) in pairs.items():
        before, after = (p >= .5) == y, (q >= .5) == y
        gain = after.astype(int)-before
        result[name] = dict(**paired(after, before), before=float(before.mean()),
                            hard_before=int(before[hard].sum()), hard_after=int(after[hard].sum()),
                            hard_fixed=int((hard & after & ~before).sum()), hard_harmed=int((hard & ~after & before).sum()),
                            net_by_fold=[int(gain[fold == f].sum()) for f in np.unique(fold)],
                            change_interval_pp=interval(gain, groups))
    old_cat, new_cat = pairs['cat'] >= .5
    old_tab, new_tab = pairs['tabicl'] >= .5
    flip_cat, flip_tab = old_cat != new_cat, old_tab != new_tab
    together = (old_cat == old_tab) & flip_cat & flip_tab
    result['overlap'] = dict(both_changed=int((flip_cat & flip_tab).sum()),
                             opposite_initial_decisions=int((flip_cat & flip_tab & (old_cat != old_tab)).sum()),
                             same_change_fixed=int((together & (new_cat == y)).sum()),
                             same_change_harmed=int((together & (new_cat != y)).sum()),
                             fixed_by_either=int(((flip_cat & (new_cat == y)) | (flip_tab & (new_tab == y))).sum()))
    write(ROOT / 'reports/nov23_matched_learning_curve/expanded_eight.json', result)
    print(result)


def run():
    row, y, _, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    raw = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda k: k != 'Name')
    target = raw.Transported.to_numpy(bool)
    group = raw.PassengerId.str.split('_').str[0].astype(int).to_numpy()
    np.testing.assert_array_equal(target[row], y)
    np.testing.assert_array_equal(group[row], groups)
    x = compact_features(raw)
    extra = np.setdiff1d(np.arange(len(x)), row)
    assert len(extra) == 1739
    output = ART / ('nov23_' + LABEL)
    output.mkdir(exist_ok=True)
    results, history = {}, {}
    for seed in [*map(str, SEEDS), 'v131', 'eight']:
        if seed == 'eight':
            fold, base, ref = eight_fold(row, y)
            initial = np.full(len(row), np.nan)
        else:
            with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as z:
                for k, v in [('row', row), ('y', y)]:
                    np.testing.assert_array_equal(z[k], v)
                fold, base, ref = z['fold'], z['base'], z['ref']
            initial, = get_bank(seed, row, y, groups, fold)[LEVEL == 4]
        full, shuffled = np.full(len(row), np.nan), np.full(len(row), np.nan)
        sizes = []
        for f in np.unique(fold):
            ix = np.flatnonzero(fold == f)
            q = row[ix]
            independent = ~np.isin(group, group[q])
            t = row[(fold != f) & independent[row]]
            add = extra[independent[extra]]
            enlarged = np.r_[t, add]
            assert len(np.unique(enlarged)) == len(enlarged)
            np.testing.assert_array_equal(np.sort(enlarged), np.flatnonzero(independent))
            if seed == 'eight':
                initial[ix] = fitting(x, target[t], t, q)
            elif f == 0:
                np.testing.assert_allclose(initial[ix], fitting(x, target[t], t, q), atol=1e-12, rtol=0)
            full[ix] = fitting(x, target[enlarged], enlarged, q)
            placebo = np.random.default_rng(609 + int(f)).permutation(target[add])
            shuffled[ix] = fitting(x, np.r_[target[t], placebo], enlarged, q)
            sizes.append([len(t), len(add), len(enlarged)])
        assert np.isfinite(np.column_stack([initial, full, shuffled])).all()
        kept = ref == (base >= .5)
        candidates = {'initial': initial >= .5, 'expanded': full >= .5, 'shuffled': shuffled >= .5,
                      'quarter_initial': np.where(kept, .75*base + .25*initial >= .5, ref),
                      'quarter_expanded': np.where(kept, .75*base + .25*full >= .5, ref)}
        scores = {k: dict(**paired(v == y, ref == y), hard_hits=int((hard & (v == y)).sum()))
                  for k, v in candidates.items()}
        results[seed] = dict(sizes=sizes, scores=scores,
                             expanded_vs_initial=paired((full >= .5) == y, (initial >= .5) == y))
        if seed in map(str, SEEDS):
            for k, vote in candidates.items():
                history.setdefault(k, []).append(vote == y)
        np.savez_compressed(output / f'{seed}.npz', row=row, y=y, fold=fold, ref=ref, initial=initial,
                            expanded=full, shuffled=shuffled)
        print(seed, sizes, {k:(v['fixed']-v['harmed'], v['hard_hits']) for k,v in scores.items()}, flush=True)
    stable = {k: dict(hard_ever=int((hard & np.any(hits, axis=0)).sum()),
                      hard_always=int((hard & np.all(hits, axis=0)).sum()),
                      right_always_harmed=int((right & ~np.any(hits, axis=0)).sum())) for k,hits in history.items()}
    write(ROOT / 'reports/nov23_matched_learning_curve/expanded_roster.json', dict(plans=results, stable=stable))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-eight', action='store_true')
    if parser.parse_args().compare_eight:
        compare_eight()
    else:
        run()
