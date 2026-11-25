"""Learn whether a passenger shares the outcome of a different, known booking."""
import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import ROOT, data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as RELEASE

CACHE = ROOT/'artifacts/nov25_known_passenger_comparisons'
DEST = ROOT/'reports/nov23_matched_learning_curve/known_passenger_comparisons.json'
FAMILIES = ('query_only', 'known_outcomes', 'shuffled_outcomes', 'blind_agreement')


def design(raw):
    frame = compact_features(raw)
    group = raw.PassengerId.str.split('_').str[0]
    frame['PartySize'] = group.map(group.value_counts())
    for name in ('RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNumber', 'Age'):
        frame[name+'_missing'] = frame[name].isna().astype(float)
        v = frame[name].fillna(0)
        if name == 'Age':
            frame[name] = v / 30
        elif name == 'CabinNumber':
            frame[name] = v / 1000
        else:
            frame[name+'_paid'] = (v > 0).astype(float)
            frame[name] = np.log1p(v) / 5
    frame['PartySize'] = np.log(frame.PartySize)
    return pd.get_dummies(frame, columns=CATEGORICAL, dtype=float).to_numpy(dtype=np.float32)


def donors(x, group, train, target, rng):
    # Every query sees six near neighbours and six random passengers, label-blind.
    distances, indices = NearestNeighbors(n_neighbors=20).fit(x[train]).kneighbors(x[target])
    candidates = train[indices]
    valid = group[candidates] != group[target, None]
    assert (valid.sum(1) >= 6).all()
    nearest = np.take_along_axis(candidates, np.argsort(np.where(valid, distances, np.inf), axis=1, kind='stable')[:, :6], axis=1)
    distant = rng.choice(train, size=(len(target), 6))
    conflict = group[distant] == group[target, None]
    while conflict.any():
        distant[conflict] = rng.choice(train, size=conflict.sum())
        conflict = group[distant] == group[target, None]
    return np.column_stack([nearest, distant])


def fit_model(x, y):
    return CatBoostClassifier(iterations=240, depth=5, learning_rate=.08, thread_count=2, random_seed=357,
                              verbose=False, allow_writing_files=False).fit(x, y)


def agreement_features(x, q, d):
    query = np.repeat(x[q], d.shape[1], axis=0)
    anchor = x[d.ravel()]
    return np.column_stack([query, anchor, np.abs(query-anchor)])


def vote(agreement, anchor_labels):
    return np.where(anchor_labels, agreement, 1-agreement).mean(axis=1)


def fit(raw, y, group, folds):
    x = design(raw)
    CACHE.mkdir(exist_ok=True)
    output = np.empty((len(y), len(FAMILIES)))
    for f in np.unique(folds):
        t, q = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        assert not np.isin(group[t], group[q]).any()
        path = CACHE/f'fold{f}.npz'
        if not path.exists():
            rng = np.random.default_rng(578 + f)
            dt, dq = donors(x, group, t, t, rng), donors(x, group, t, q, rng)
            train_pairs, query_pairs = agreement_features(x, t, dt), agreement_features(x, q, dq)
            same = (y[t, None] == y[dt]).ravel()
            blind = fit_model(train_pairs, same)
            knows = fit_model(np.column_stack([train_pairs, y[dt.ravel()]]), same)
            query_only = fit_model(x[t], y[t]).predict_proba(x[q])[:, 1]
            def condition(labels):
                p = knows.predict_proba(np.column_stack([query_pairs, labels.ravel()]))[:, 1].reshape(dq.shape)
                return vote(p, labels)
            blind_p = blind.predict_proba(query_pairs)[:, 1].reshape(dq.shape)
            # Preserve each donor's placebo label across all of its appearances.
            shuffled = y.copy()
            shuffled[t] = rng.permutation(y[t])
            p = np.column_stack([query_only, condition(y[dq]), condition(shuffled[dq]), vote(blind_p, y[dq])])
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, q=q, t=t, training_y=y[t], passenger=raw.PassengerId.iloc[q].to_numpy(str), p=p)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for key, val in [('t', t), ('q', q), ('training_y', y[t]), ('passenger', raw.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[key], val)
            output[q] = z['p']
        print('completed fold', f, flush=True)
    assert np.isfinite(output).all() and ((0 <= output) & (output <= 1)).all()
    return output


def counts(ref, pred, y, mask):
    good = ref == y
    return dict(rows=int(mask.sum()), original_errors=int((mask & ~good).sum()),
                fixed=int((mask & ~good & (pred == y)).sum()), harmed=int((mask & good & (pred != y)).sum()))


def evaluate(reference_folds):
    raw, y, group, fold = data()
    prob = fit(raw, y, group, fold)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, hf = z['rows'], z['fold']
    _, panels = prediction_sets(raw, y, group, fold, rows, hf)
    votes = (np.column_stack(panels[0]) >= .5).sum(axis=1)
    majority = votes > 34
    unanimous = (votes == 0) | (votes == 68)
    near = (votes <= 3) | (votes >= 65)
    with np.load(ROOT/'artifacts/nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['fold'], fold)
        native = z['v96']
        best = z['certified_nine'].copy().astype(int)
        best[~z['certified_v131_equals_v96']] = -1
    for f in reference_folds:
        with np.load(RELEASE/f'release{f}.npz') as z:
            np.testing.assert_array_equal(z['q'], np.flatnonzero(fold == f))
            best[z['q']] = z['nine']
    known = best >= 0
    all_rows = np.ones(len(y), dtype=bool)
    result = {}
    for name, p in zip(FAMILIES, prob.T, strict=True):
        pred = p >= .5
        mixed = (.75*native + .25*p) >= .5
        # Only transfer a mixed decision on rows where the current release kept the native sign.
        challenger = np.where(best == (native >= .5), mixed, best)
        result[name] = dict(accuracy=float((pred == y).mean()), vs_majority=counts(majority,pred,y,all_rows),
                            unanimous=counts(majority,pred,y,unanimous), within_three_votes=counts(majority,pred,y,near),
                            known_nine_direct=counts(best,pred,y,known), known_nine_quarter=counts(best,challenger,y,known),
                            nine_quarter_folds=[counts(best,challenger,y,known & (fold == f)) for f in np.unique(fold)])
        print(name,result[name]['unanimous'],result[name]['known_nine_quarter'],flush=True)
    result['reference'] = dict(reconstructed_folds=list(reference_folds), full_consensus_count=68, known_nine=int(known.sum()))
    path = DEST if tuple(reference_folds) == (0,1,2) else DEST.with_stem(DEST.stem + '_all_folds')
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-reference-folds', action='store_true')
    folds = range(8) if parser.parse_args().all_reference_folds else range(3)
    with threadpool_limits(2):
        evaluate(folds)
