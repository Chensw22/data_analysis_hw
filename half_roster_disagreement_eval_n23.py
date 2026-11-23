"""Compare small and pooled calibration samples with identical queries and features."""
import json

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data
from full_roster_metric_adjudication_n23 import load_scores
from half_roster_disagreement_n23 import DEST

NEIGHBOR = ['fixed', 'learned', 'focus']


def inputs(scores, raw, groups):
    mitra, pick = scores[:, 0] >= .5, scores[:, -1] >= .5
    oriented = logit(scores.clip(.001, .999))*(2*pick.astype(int)-1)[:, None]
    x = np.column_stack([oriented, pick])
    fields = raw[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
    _, ids, size = np.unique(groups, return_inverse=True, return_counts=True)
    context = np.column_stack([x, np.log1p(fields.fillna(0)), fields.isna(), raw.Age.fillna(-1),
                               raw.CryoSleep.eq(True), raw.CryoSleep.isna(), size[ids]])
    return dict(linear=x, tree=context), mitra, pick


def run():
    raw, y, groups, fold = data()
    panel = fold % 2
    names = ['cat', 'budget', *NEIGHBOR]
    with np.load(DEST / 'others.npz') as z:
        for key, val in [('passenger', raw.PassengerId), ('panel', panel), ('y', y)]:
            np.testing.assert_array_equal(z[key], val)
        half_scores = np.column_stack([np.zeros(len(y)), *(z[k] for k in names)])
    for h in (0, 1):
        t, q = np.flatnonzero(panel != h), np.flatnonzero(panel == h)
        with np.load(DEST / f'mitra_{h}.npz') as z:
            for key, val in [('train', t), ('labels', y[t]), ('query', q), ('steps', 15)]:
                np.testing.assert_array_equal(z[key], val)
            half_scores[q, 0] = z['p']
    original = {k: load_scores(raw, y, fold, OUT / f'{k}.npz', ['p'])[:, 0]
                for k in [*FAMILIES, 'mitra96']}
    metrics = load_scores(raw, y, fold, ROOT / 'artifacts/nov23_full_roster_metric_neighbors/oof.npz', NEIGHBOR)
    full_scores = np.column_stack([original[k] for k in ['mitra96', 'cat', 'budget']] + [metrics])
    assert np.isfinite(half_scores).all() and ((0 <= half_scores) & (half_scores <= 1)).all()
    cohorts = np.column_stack(list(original.values()))
    shared = ((cohorts >= .5) != y[:, None]).all(1)
    unanimous_correct = ((cohorts >= .5) == y[:, None]).all(1)
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    sources = {'half': inputs(half_scores, raw, groups), 'seven_folds': inputs(full_scores, raw, groups)}
    hx, base, pick = sources['half']
    policies = {s: {} for s in sources}
    sizes = []
    for f in np.unique(fold):
        calibration = np.flatnonzero((panel == f % 2) & (fold != f))
        query = np.flatnonzero(fold == f)
        assert not np.isin(groups[calibration], groups[query]).any()
        splits = GroupKFold(5, shuffle=True, random_state=SEED+int(f)).split(calibration, groups=groups[calibration])
        samples = {f'block_{i}': calibration[j] for i, (_, j) in enumerate(splits)}
        np.testing.assert_array_equal(np.sort(np.concatenate(list(samples.values()))), calibration)
        samples['pooled'] = calibration
        counts = {}
        for block, rows in samples.items():
            ids = rows[base[rows] != pick[rows]]
            counts[block] = len(ids)
            for method in ['linear', 'tree']:
                model = (make_pipeline(StandardScaler(), LogisticRegression(C=.10, max_iter=500))
                         if method == 'linear' else CatBoostClassifier(iterations=120, depth=3, learning_rate=.04,
                             verbose=False, allow_writing_files=False, thread_count=2, random_seed=SEED))
                model.fit(hx[method][ids], (pick[ids] == y[ids]).astype(int))
                for s, (x, ref, choose) in sources.items():
                    ask = query[ref[query] != choose[query]]
                    swap = ask[model.predict_proba(x[method][ask])[:, 1] >= .5]
                    policy = policies[s].setdefault(f'{method}_{block}', ref.copy())
                    policy[swap] = choose[swap]
        sizes.append(counts)
    result = dict(shared=int(shared.sum()), calibration=sizes, queries={})
    for s, (_, ref, choice) in sources.items():
        entry = dict(baseline_accuracy=float((ref == y).mean()), disagreements=int((ref != choice).sum()))
        for name, policy in policies[s].items():
            fixed, harm = (policy == y) & (ref != y), (policy != y) & (ref == y)
            delta = fixed.astype(int)-harm
            entry[name] = dict(fixed=int(fixed.sum()), harmed=int(harm.sum()), net=int(delta.sum()))
            if name.endswith('pooled'):
                method = name.split('_')[0]
                block_accuracy = np.mean([policies[s][f'{method}_block_{i}'] == y for i in range(5)], axis=0)
                vs_small = (policy == y)-block_accuracy
                entry[name].update(accuracy=float((policy == y).mean()), shared_fixed=int((fixed & shared).sum()),
                    shared_harmed=int((harm & shared).sum()), unanimous_correct_harmed=int((harm & unanimous_correct).sum()),
                    historic_fixed=int((fixed & hard).sum()),
                    historic_harmed=int((harm & hard).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
                    fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
                    net_ci_pp=interval(delta, groups), net_over_mean_small=float(vs_small.sum()),
                    vs_small_ci_pp=interval(vs_small, groups))
        result['queries'][s] = entry
    full_base = full_scores[:, 0] >= .5
    low_votes = half_scores >= .5
    unified = (low_votes == low_votes[:, :1]).all(1)
    result['training_size_votes'] = {}
    for name, policy in [('half_mitra', low_votes[:, 0]),
                         ('mean_mitra', half_scores[:, 0]+full_scores[:, 0] >= 1),
                         ('unanimous_half', np.where(unified, low_votes[:, 0], full_base))]:
        fixed = (policy == y) & (full_base != y)
        harm = (policy != y) & (full_base == y)
        delta = fixed.astype(int)-harm
        result['training_size_votes'][name] = dict(fixed=int(fixed.sum()), harmed=int(harm.sum()),
            shared_fixed=int((fixed & shared).sum()), historical_fixed=int((fixed & hard).sum()),
            accuracy=float((policy == y).mean()), net_ci_pp=interval(delta, groups),
            fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)])
    path = ROOT / 'reports/nov23_matched_learning_curve/disagreement_training_size.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    for name, entry in result['queries'].items():
        print(name, entry['linear_pooled'], entry['tree_pooled'])


if __name__ == '__main__':
    run()
