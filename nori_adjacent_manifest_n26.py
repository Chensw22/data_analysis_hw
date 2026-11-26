"""Test Nori with visible neighboring orders on the same cabin deck and side.

Run both 'real' and 'permuted' with .venv-tabpfn/bin/python, then run 'report'
with DYLD_LIBRARY_PATH=.venv-tabpfn/lib/python3.12/site-packages/torch/lib
and .venv/bin/python on Apple Silicon. The Nori installation is described in nori_booking_regression_n26.py.
Neighbor features are transductive summaries of the official unlabeled manifest.
"""
import argparse
from time import monotonic

import numpy as np
import pandas as pd

from nori_booking_regression_n26 import CACHE, FEES, ROOT, SHA, data, design, packet_dir
from nori_fare_party_n26 import predictor

ART = ROOT / 'artifacts/nov26_nori_adjacent_manifest'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/nori_adjacent_manifest.json'
WIDTH = 6
PERMUTATION_SEED = 903151


def neighbor_features(frame, permuted):
    groups = frame.PassengerId.str.split('_').str[0].astype(int)
    cabin = frame.Cabin.str.split('/', expand=True)
    fees = frame[FEES]
    observations = pd.DataFrame({
        'residents': np.ones(len(frame)),
        'sleep': frame.CryoSleep.astype(float),
        'any_spend': fees.gt(0).any(axis=1).where(fees.notna().all(axis=1)).astype(float),
        **{f'{name}_used': fees[name].gt(0).where(fees[name].notna()).astype(float) for name in FEES},
    })
    observations['any_spend'] = observations['any_spend'].mask(fees.gt(0).any(axis=1), 1.)
    data_matrix = observations.to_numpy()
    features = np.full((len(frame), data_matrix.shape[1] + 2), np.nan, dtype=np.float32)
    names = ['orders', 'orders_per_100_ids', 'passengers_per_order', *observations.columns[1:]]
    rng = np.random.default_rng(PERMUTATION_SEED)
    for _, rows in frame.groupby([cabin[0], cabin[2]], sort=True).indices.items():
        order_ids, index = np.unique(groups.iloc[rows], return_inverse=True)
        n = len(order_ids)
        if n < 2:
            continue
        assignments = rng.permutation(n) if permuted else np.arange(n)
        # Each complete order occupies one slot on this lane, regardless of party size.
        # The shuffle keeps density of the original lane slots and co-located passengers.
        slots = assignments[index]
        sums = np.zeros((n, data_matrix.shape[1]))
        counts = np.zeros_like(sums)
        np.add.at(sums, slots, np.nan_to_num(data_matrix[rows]))
        np.add.at(counts, slots, np.isfinite(data_matrix[rows]))
        positions = np.arange(n)
        start = np.maximum(0, positions - WIDTH)
        end = np.minimum(n, positions + WIDTH + 1)
        neighboring_orders = end - start - 1

        def exclude_self(arr):
            cumulative = np.vstack([np.zeros((1, arr.shape[1])), np.cumsum(arr, axis=0)])
            return cumulative[end] - cumulative[start] - arr

        nearby_sum, nearby_count = exclude_self(sums), exclude_self(counts)
        assert np.all(nearby_count[:, 0] >= neighboring_orders)
        rates = np.full_like(nearby_sum[:, 1:], np.nan)
        np.divide(nearby_sum[:, 1:], nearby_count[:, 1:], where=nearby_count[:, 1:] > 0, out=rates)
        density = 100 * neighboring_orders / (order_ids[end-1] - order_ids[start])
        values = np.column_stack([neighboring_orders, density, nearby_sum[:, 0]/neighboring_orders, rates])
        features[rows] = values[slots]
    return features, names


def roster(raw):
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns)
    return pd.concat([raw, test], ignore_index=True)


def run(mode):
    ART.mkdir(exist_ok=True)
    raw, y, groups, fold = data()
    x_extra, _ = neighbor_features(roster(raw), mode == 'permuted')
    torch, regressor = predictor()
    for k in range(8):
        path = ART/f'{mode}{k}.npz'
        start = monotonic()
        t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if path.exists():
            with np.load(path) as z:
                np.testing.assert_array_equal(z['q'], q)
                np.testing.assert_array_equal(z['t'], t)
                np.testing.assert_array_equal(z['labels'], y[t])
                np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[q].to_numpy(str))
                assert str(z['sha']) == SHA and np.isfinite(z['p']).all()
            continue
        with np.load(packet_dir(0)/f'fold{k}.npz') as z:
            for name, value in [('t', t), ('context', t), ('q', q), ('labels', y[t]), ('ids', raw.PassengerId.iloc[q].to_numpy(str))]:
                np.testing.assert_array_equal(z[name], value)
            assert str(z['checkpoint']) == SHA and str(z['recipe']) == (CACHE/'one_member.json').read_text()
        assert not np.isin(groups[t], groups[q]).any()
        x = np.column_stack([design(raw, groups, t), x_extra[:len(y)]])
        np.random.seed(6117+k)
        torch.manual_seed(6117+k)
        model = regressor(model_path=str(CACHE/'nori.pt'), model='nori-6m', device='cpu',
                          inference_config=str(CACHE/'one_member.json'), categorical_columns=[], augmentations=())
        model.fit(x[t], y[t].astype(float))
        p = np.asarray(model.predict(x[q]), dtype=float).ravel()
        assert len(p) == len(q) and np.isfinite(p).all()
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, t=t, q=q, p=p, labels=y[t], passenger=raw.PassengerId.iloc[q].to_numpy(str),
                                sha=SHA, seconds=monotonic()-start)
        temporary.replace(path)
        print(mode, k, int(((p >= .5) == y[q]).sum()), len(q), round(monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['real', 'permuted', 'report'])
    mode = parser.parse_args().mode
    if mode == 'report':
        from assess_nori_adjacent_manifest_n26 import report
        report()
    else:
        run(mode)
