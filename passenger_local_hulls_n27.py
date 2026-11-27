"""Compare local convex reconstructions of manifest attributes under each transport outcome."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from exaone_validation_data import compact_features, CATEGORICAL
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART / 'nov27_local_hulls'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/passenger_local_hulls.json'


def attributes(frame):
    compact = compact_features(frame)
    cat = pd.get_dummies(compact[CATEGORICAL], dtype=float)
    money = compact[FEES].to_numpy()
    pieces = [cat.to_numpy() / np.sqrt(2), np.isnan(money), np.nan_to_num(money) > 0,
              np.log1p(np.nan_to_num(money)) / 5]
    for col, scale in [('Age', 25), ('CabinNumber', 500)]:
        v = compact[col].to_numpy(float)
        pieces += [np.isnan(v)[:, None], np.nan_to_num(v[:, None], nan=np.nanmedian(v)) / scale]
    booking = frame.PassengerId.str.split('_').str[0]
    size = booking.map(booking.value_counts()).to_numpy()
    pieces.append(np.log(size[:, None]))
    return np.column_stack(pieces)


def distances(x, t, q):
    """Compare nearest point, average local distance and nearest point in a regularized hull."""
    model = NearestNeighbors(n_neighbors=20, algorithm='brute').fit(x[t])
    result = []
    for start in range(0, len(q), 256):
        query = x[q[start:start + 256]]
        d, idx = model.kneighbors(query)
        offset = x[t[idx]] - query[:, None, :]
        gram = offset @ offset.transpose(0, 2, 1)
        # Frank-Wolfe on squared reconstruction error plus L2 weights regularization.
        # Initialize with uniform weights; the exact line search also works for duplicate records.
        ridge = .1
        gram[:, np.arange(20), np.arange(20)] += ridge
        w = np.full(d.shape, 1 / 20.)
        for _ in range(36):
            grad = (gram @ w[..., None]).squeeze(-1)
            col = np.argmin(grad, axis=1)
            step = -w.copy()
            step[np.arange(len(col)), col] += 1.
            curve = (step * (gram @ step[..., None]).squeeze(-1)).sum(1)
            rate = np.clip(-(grad * step).sum(1) / np.maximum(curve, 1e-12), 0, 1)
            w += rate[:, None] * step
        assert np.all(w >= -1e-12) and np.allclose(w.sum(1), 1)
        sq = np.square((offset * w[:, :, None]).sum(1)).sum(1) + ridge * np.square(w).sum(1)
        result.append(np.column_stack([d[:, 0] ** 2, (d * d).mean(1), sq]))
    return np.vstack(result)


def run():
    raw, y, groups, fold = data()
    _, blind, known, fresh = cohort(raw, y, fold)
    n = len(y)
    frame = pd.concat([raw, pd.read_csv(ROOT / 'data/raw/test.csv', usecols=raw.columns)], ignore_index=True)
    ids = frame.PassengerId.to_numpy(str)
    all_groups = frame.PassengerId.str.split('_').str[0].to_numpy()
    x = attributes(frame)
    dist = np.empty((len(frame), 2, 3))
    for k in range(9):
        tr = np.flatnonzero(fold != k)
        q = np.flatnonzero(fold == k) if k < 8 else np.arange(n, len(frame))
        assert not np.isin(all_groups[tr], all_groups[q]).any()
        for label in (0, 1):
            dist[q, label] = distances(x, tr[y[tr] == label], q)
    odds = np.log((dist[:, 0] + .005) / (dist[:, 1] + .005))
    with np.load(ART / 'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'], ids)
        np.testing.assert_array_equal(z['y'], y)
        first = z['first']
        anchor = {s: first ^ z[s + '_median'] for s in ('random', 'contiguous')}
    report = {'individual_accuracy': ((odds[:n] > 0) == y[:, None]).mean(0).tolist(), 'rules': {}}
    output = {}
    for scheme, base in anchor.items():
        part = splits(groups, fold, scheme)
        risk = np.where(first[:, None], -odds, odds)
        measures = {'point': risk[:, 0], 'neighbors': risk[:, 1], 'hull': risk[:, 2],
                    'hull_increment': np.where(risk[:, 2] > 0, risk[:, 2] - risk[:, 1], -np.inf)}
        for name, signal in measures.items():
            prob = expit(signal)
            action = np.zeros(len(ids), bool)
            cuts = []
            for k in range(8):
                for j in (0, 1, 2, 3, None):
                    tr = np.flatnonzero(known & (fold == k) & ((part != j) if j is not None else True))
                    threshold = calibrate(prob, first, y, tr, 1.)
                    if j is None:
                        cuts.append(threshold)
                    else:
                        q = np.flatnonzero((fold == k) & (part == j))
                        action[q] = prob[q] >= threshold[first[q].astype(int)]
            action[n:] = (prob[n:, None] >= np.array(cuts)[:, first[n:].astype(int)].T).sum(1) > 4
            output[scheme + '_' + name] = action & (base == first)
        for name in [*measures, 'hull_only', 'hull_and_neighbors']:
            if name == 'hull_only':
                action = output[scheme + '_hull'] & ~output[scheme + '_point'] & ~output[scheme + '_neighbors']
            elif name == 'hull_and_neighbors':
                action = output[scheme + '_hull'] & output[scheme + '_neighbors']
            else:
                action = output[scheme + '_' + name]
            # All thresholds were fitted against the same first-stage parent; retain its scored upgrades.
            err = base[:n] != y
            delta = action[:n] * np.where(err, 1, -1)
            report['rules'].setdefault(name, {})[scheme] = dict(
                all=count(action[:n], known, err), common=count(action[:n], blind, err),
                fresh=count(action[:n], known & fresh, err), test_changes=int(action[n:].sum()),
                folds=[int(delta[known & (fold == k)].sum()) for k in range(8)])
        print(scheme, [(a, v[scheme]['all'], v[scheme]['test_changes']) for a,v in report['rules'].items()])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'aligned.npz', passenger=ids, y=y, fold=fold, dist=dist, **output)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
