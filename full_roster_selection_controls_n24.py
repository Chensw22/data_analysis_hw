"""Compare cached permutation selections against random feature subspaces in CatBoost."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_permutation_selection_n24 import OUT, cat, feature_bank


MODEL_SEED = 497
# The first permutation also appears in the originating experiment.
RANDOM_SEEDS = (1216, 1217, 1218, 1219)


def ordered_subspaces(cached, field_count, core):
    assert cached.shape == (4, 52)
    for cols in cached:
        assert len(np.unique(cols)) == 52 and cols.min() >= 0 and cols.max() < field_count
    np.testing.assert_array_equal(cached[2, :core], np.arange(core))
    shared_engineered = cached[0, cached[0] >= core]
    count = min(len(shared_engineered), 52 - core)
    np.testing.assert_array_equal(cached[2, core:core + count], shared_engineered[:count])
    subspaces = {'all': np.arange(field_count), 'selected': cached[0],
                 'protected_selected': cached[2]}
    for seed in RANDOM_SEEDS:
        order = np.random.default_rng(seed).permutation(field_count)
        random_cols = order[:52]
        protected_cols = np.r_[np.arange(core), order[order >= core][:52 - core]]
        if seed == RANDOM_SEEDS[0]:
            np.testing.assert_array_equal(cached[1], random_cols)
            np.testing.assert_array_equal(cached[3], protected_cols)
        subspaces[f'random_{seed}'] = random_cols
        subspaces[f'protected_random_{seed}'] = protected_cols
    return {key: np.sort(cols) for key, cols in subspaces.items()}


def predict(x, fields, core, y, groups, ix, folds, tag):
    probabilities = {}
    audit = []
    for fold in np.unique(folds):
        where = folds == fold
        test = ix[where]
        train = ix[~where & ~np.isin(groups[ix], groups[test])]
        assert not np.isin(groups[train], groups[test]).any()
        path = OUT / f'{tag}_{fold}.npz'
        with np.load(path, allow_pickle=False) as cache:
            for key, expected in [('train', train), ('query', test),
                                  ('features', fields), ('labels', y[train])]:
                np.testing.assert_array_equal(cache[key], expected)
            subspaces = ordered_subspaces(cache['cols'], len(fields), core)
        for mode, cols in subspaces.items():
            assert len(cols) in (len(fields), 52) and (np.diff(cols) > 0).all()
            x_train = x[np.ix_(train, cols)]
            x_test = x[np.ix_(test, cols)]
            model = cat(MODEL_SEED).fit(x_train, y[train])
            prob = model.predict_proba(x_test, thread_count=3)[:, 1]
            probabilities.setdefault(mode, np.full(len(ix), np.nan))[where] = prob
        audit.append(dict(fold=int(fold), train=len(train), query=len(test),
                          columns={k: c.tolist() for k, c in subspaces.items() if k != 'all'}))
        print(tag, int(fold), len(train), len(test), flush=True)
    for prefix in ['random', 'protected_random']:
        probabilities[prefix + '_mean'] = np.mean(
            [probabilities[f'{prefix}_{s}'] for s in RANDOM_SEEDS], axis=0)
    assert np.isfinite(np.stack(list(probabilities.values()))).all()
    return probabilities, audit


def measure(choice, baseline, y, groups, fold, mask):
    right = choice[mask] == y[mask]
    previous = baseline[mask] == y[mask]
    delta = right.astype(int) - previous.astype(int)
    return dict(n=int(mask.sum()), accuracy=float(right.mean()), fixed=int((delta == 1).sum()),
                harmed=int((delta == -1).sum()), net=int(delta.sum()),
                fold_net=[int(delta[fold[mask] == f].sum()) for f in np.unique(fold)],
                order_bootstrap_ci_pp=interval(delta, groups[mask]))


def run():
    _, old_y, old_groups, rows, old_fold, _, champion, _, _, _, _, old, _ = aligned()
    raw, y, groups, folds = data()
    names, half, full = banks(raw, y, folds)
    reference = latest_reference(y, groups, folds, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, groups, folds)
    np.testing.assert_array_equal(gate >= .5, reference)
    x, fields, core = feature_bank(raw)
    a, old_audit = predict(x, fields, core, y, groups, rows, old_fold, 'historical')
    b, new_audit = predict(x, fields, core, y, groups, np.arange(len(y)), folds, 'full')
    historical_mask = np.isin(np.arange(len(y)), rows)
    unseen = ~np.isin(groups, old_groups)
    assert unseen.sum() == 1044 and not np.any(unseen & historical_mask)
    shared = ((full >= .5) != y[:, None]).all(axis=1)
    hard = shared[rows] & (champion != old_y)
    report = dict(model_seed=MODEL_SEED, field_names=fields, core_count=core,
                  random_seeds=RANDOM_SEEDS, trained_in_original_column_order=True,
                  audit=dict(historical=old_audit, full=new_audit),
                  shared19=int(shared.sum()), shared19_and_v131_wrong=int(hard.sum()), results={})
    for mode in a:
        routed_old = np.where(champion == (old[:, 0] >= .5),
                              .75 * old[:, 0] + .25 * a[mode] >= .5, champion)
        routed_new = .75 * gate + .25 * b[mode] >= .5
        parts = {'full': np.ones(len(y), bool), 'historical_rows_in_full': historical_mask,
                 'extra_seen_group': ~historical_mask & ~unseen, 'extra_unseen_group': unseen}
        result = {name: measure(routed_new, reference, y, groups, folds, mask)
                  for name, mask in parts.items()}
        result['historical'] = measure(routed_old, champion, old_y, old_groups, old_fold,
                                       np.ones(len(rows), bool))
        result['shared19_correct'] = int((shared & (routed_new == y)).sum())
        result['shared19_correct_at_baseline'] = int((shared & (reference == y)).sum())
        result['stable_fixes'] = int((hard & (routed_old == old_y) & (routed_new[rows] == old_y)).sum())
        report['results'][mode] = result
        print(mode, [result[k]['net'] for k in ('historical', 'full', 'extra_unseen_group')], flush=True)
    target = ROOT / 'reports/nov23_matched_learning_curve/selection_controls.json'
    target.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(3):
        run()
