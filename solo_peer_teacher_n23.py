"""Distill training-only, outcome-free booking companion attributes into solo predictions."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL

LABEL = 'solo_peer_teacher'


def inputs(row):
    frame, groups, size, _ = load_roster()
    bills = frame[BILLS]
    own = pd.get_dummies(frame, columns=CATEGORICAL, dtype=float)
    own[BILLS] = np.log1p(bills)
    own['known_total'] = np.log1p(bills.sum(axis=1, min_count=1))
    own['active_services'] = bills.gt(0).sum(axis=1)
    own['missing_bills'] = bills.isna().sum(axis=1)
    # Group sums and observed counts exclude the focal person's value column by column.
    totals = own.fillna(0.).groupby(groups).transform('sum')-own.fillna(0.)
    counts = own.notna().astype(int).groupby(groups).transform('sum')-own.notna().astype(int)
    peers = totals.div(counts.mask(counts == 0))
    assert peers.iloc[size == 1].isna().all().all()
    own_array = own.to_numpy(dtype=np.float32)[row]
    rich = np.column_stack([own_array, peers.iloc[row].to_numpy(dtype=np.float32), size[row]])
    return own_array, rich, size[row], groups[row]


def predict(x, ids, target, query):
    train = xgb.DMatrix(x[ids], label=target, nthread=1)
    fitted = xgb.train(dict(objective='reg:logistic', tree_method='hist', nthread=1,
                            eta=.055, max_depth=4, min_child_weight=5, reg_lambda=5.,
                            seed=727, base_score=.5), train, num_boost_round=180)
    return fitted.predict(xgb.DMatrix(x[query], nthread=1))


def fit_students(own, rich, groups, solo, y, t, q):
    mt = t[~solo[t]]
    st = t[solo[t]]
    result = dict(pooled=predict(own, t, y[t], q), solo=predict(own, st, y[st], q))
    size_only = np.column_stack([own, rich[:, -1]])
    for name, teacher_x in [('personal_teacher', own), ('size_teacher', size_only),
                            ('companion_teacher', rich)]:
        soft = np.full(len(mt), np.nan)
        for a, b in GroupKFold(3, shuffle=True, random_state=353).split(mt, groups=groups[mt]):
            assert not np.intersect1d(groups[mt[a]], groups[mt[b]]).size
            soft[b] = predict(teacher_x, mt[a], y[mt[a]], mt[b])
        assert np.isfinite(soft).all()
        target = y[t].astype(float)
        target[~solo[t]] = .5*target[~solo[t]] + .5*soft
        result[name] = predict(own, t, target, q)
    return result


def run():
    row, y, _, groups, _ = load()
    own, rich, size, known_groups = inputs(row)
    np.testing.assert_array_equal(groups, known_groups)
    solo = size == 1
    _, _, hard, right = hard_cohorts(row, y)
    report = ROOT/'reports'/('nov23_'+LABEL)
    directory = ART/('nov23_'+LABEL)
    directory.mkdir(exist_ok=True)
    plans, history = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            np.testing.assert_array_equal(row, cache['row'])
            np.testing.assert_array_equal(y, cache['y'])
            fold, ref, base = cache['fold'], cache['ref'], cache['base']
        predictions = {}
        for f in np.unique(fold):
            q = np.flatnonzero((fold == f) & solo)
            t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[fold == f]))
            assert not np.intersect1d(groups[t], groups[fold == f]).size
            for name, scores in fit_students(own, rich, groups, solo, y, t, q).items():
                predictions.setdefault(name, np.full(len(y), np.nan))[q] = scores
        hits = {}
        for name, p in predictions.items():
            np.testing.assert_array_equal(np.isfinite(p), solo)
            hits[name] = np.where(solo, p >= .5, ref) == y
            hits['quarter_'+name] = np.where(solo & (ref == (base >= .5)), .75*base+.25*p >= .5, ref) == y
        plans[seed] = {}
        for name, hit in hits.items():
            prefix = 'quarter_' if name.startswith('quarter_') else ''
            plans[seed][name] = dict(vs_reference=paired(hit, ref == y), solo_accuracy=float(hit[solo].mean()),
                                     vs_personal_teacher=paired(hit, hits[prefix+'personal_teacher']),
                                     vs_size_teacher=paired(hit, hits[prefix+'size_teacher']))
        if seed != 'v131':
            for name, hit in hits.items():
                history.setdefault(name, []).append(hit)
        np.savez_compressed(directory/f'{seed}.npz', row=row, y=y, solo_mask=solo, fold=fold, ref=ref, base=base, **predictions)
        print(seed, {name: pair['vs_reference']['fixed']-pair['vs_reference']['harmed'] for name, pair in plans[seed].items()}, flush=True)
    stable = {}
    for name, parts in history.items():
        hit = np.stack(parts)
        stable[name] = dict(hard_ever=int((hard & solo & hit.any(0)).sum()),
                            hard_stable=int((hard & solo & hit.all(0)).sum()),
                            stable_harms=int((right & solo & ~hit.any(0)).sum()))
    write(report/'results.json', dict(development=len(y), solo=int(solo.sum()),
                                      solo_hard=int((hard & solo).sum()), total_hard=int(hard.sum()),
                                      plans=plans, stable=stable))


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
