"""Learn whether two disjoint solo training populations make the same reproducible mistakes."""
import json
from itertools import combinations

import numpy as np
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from scale_teacher_exception_training_n26 import manifest_features
from full_roster_booking_validation_n23 import ROOT, data
from tabnet_attention_transfer_n26 import RESULT, collect
from target_mix_expert_routing_n26 import ticker

REPORT = RESULT.parent / 'replicated_solo_training_errors.json'
OUT = ROOT / 'artifacts/nov26_replicated_solo_training_errors'
PAIRS = list(combinations(range(3), 2))


def teachers(x, y, train, query, seed):
    models = [ExtraTreesClassifier(n_estimators=80, min_samples_leaf=4, max_features=.8,
                                   n_jobs=1, random_state=seed),
              HistGradientBoostingClassifier(max_iter=110, max_leaf_nodes=12, learning_rate=.065,
                                             l2_regularization=3, early_stopping=False, random_state=seed)]
    return np.column_stack([model.fit(x[train], y[train]).predict_proba(x[query])[:, 1] for model in models])


def design_pair(x, pair, seed):
    logits = logit(pair.clip(.003, .997))
    mean = logits.mean(1)
    spread = np.abs(logits[:, 0] - logits[:, 1])
    # Swap the complete spread vector only among passengers with similar mean predictions.
    strata = np.floor(pair.mean((1, 2)) * 10).astype(int).clip(0, 9)
    perm = np.arange(len(x))
    rng = np.random.default_rng(seed)
    for band in np.unique(strata):
        rows = np.flatnonzero(strata == band)
        perm[rows] = rng.permutation(rows)
    base = np.column_stack([x, mean])
    return dict(mean=base, real=np.column_stack([base, spread]), sham=np.column_stack([base, spread[perm]]))


def experiment(x, y, g, t, q, k):
    # Each training passenger is queried on both of the other thirds.
    assert not np.isin(g[t], g[q]).any() and len(np.unique(g[t])) == len(t)
    train_predictions = np.full((len(t), 3, 2), np.nan)
    held_predictions = []
    for j, (other, one) in enumerate(KFold(3, shuffle=True, random_state=469+k).split(t)):
        values = teachers(x, y, t[one], np.r_[t[other], q], 220+k)
        train_predictions[other, j] = values[:len(other)]
        held_predictions.append(values[len(other):])
    assert (np.isfinite(train_predictions).all(axis=2).sum(axis=1) == 2).all()
    pair = train_predictions[np.isfinite(train_predictions).all(axis=2)].reshape(len(t), 2, 2)
    held = np.stack(held_predictions, axis=1)
    tr = design_pair(x[t], pair, 5201+k)
    query = [design_pair(x[q], held[:, [i, j]], 5320+k*3+m) for m, (i, j) in enumerate(PAIRS)]
    out = {}
    for key in tr:
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=8, learning_rate=.055,
                                               l2_regularization=8, early_stopping=False)
        model.fit(tr[key], y[t])
        out[key] = np.stack([model.predict_proba(v[key])[:, 1] for v in query], axis=1)
    out['base'] = np.column_stack([held[:, [i, j]].mean((1, 2)) for i, j in PAIRS])
    return out


def run():
    raw, y, g, fold = data()
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    solo = size == 1
    x = manifest_features(raw, size)
    scores = {}
    for k in range(8):
        t, q = np.flatnonzero(solo & (fold != k)), np.flatnonzero(solo & (fold == k))
        for key, values in experiment(x, y, g, t, q, k).items():
            scores.setdefault(key, np.full((len(y), 3), np.nan))[q] = values
    for values in scores.values():
        assert np.isfinite(values[solo]).all() and np.isnan(values[~solo]).all()
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'scores.npz', passenger=raw.PassengerId.to_numpy(str), fold=fold, **scores)
    report = {'solo_rows': int(solo.sum()), 'solo_accuracy': {
        key: float(np.mean((v[solo].mean(axis=1) >= .5) == y[solo])) for key, v in scores.items()}}
    for tag, batches in collect()[0].items():
        ids, common, parent = (np.concatenate([b[j] for b in batches]) for j in range(3))
        scope = solo[ids] & common & ~np.concatenate([b[3]['actual_and_scores'] for b in batches])
        wrong = parent != y[ids]
        repeated = {key: ((v[ids] >= .5) != parent[:, None]).all(1) for key, v in scores.items()}
        acts = dict(repeated)
        acts['new_vs_mean'] = repeated['real'] & ~repeated['mean']
        acts['new_vs_both_controls'] = acts['new_vs_mean'] & ~repeated['sham']
        acts['sham_only'] = repeated['sham'] & ~repeated['mean'] & ~repeated['real']
        acts['real_overrides_all_raw_pairs'] = repeated['real'] & ((scores['base'][ids] >= .5) == parent[:, None]).all(1)
        report[tag] = {}
        for key, change in acts.items():
            result = ticker(scope, wrong, change)
            delta = scope * change * np.where(wrong, 1, -1)
            result['net_by_fold'] = [int(delta[fold[ids] == k].sum()) for k in range(8)]
            report[tag][key] = result
        print(tag, {k: (v['fixed'], v['harmed']) for k, v in report[tag].items()}, flush=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
