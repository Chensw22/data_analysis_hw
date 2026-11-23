"""Expand disagreement supervision with two shared, label-disjoint calibration panels."""
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

import numpy as np

from full_roster_booking_validation_n23 import ROOT, data

DEST = ROOT / 'artifacts/nov23_half_roster_disagreement'


def fit_mitra(raw, y, groups, panel):
    import torch
    import mitra_counts_fixed15_online_first as counts
    from full_roster_mitra_validation_n23 import fit

    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    counts.verify_reference()
    roster = counts.load_roster()
    np.testing.assert_array_equal(roster.passenger_ids.iloc[:len(y)], raw.PassengerId)
    for half in (0, 1):
        t, q = np.flatnonzero(panel != half), np.flatnonzero(panel == half)
        assert not np.isin(groups[t], groups[q]).any()
        path = DEST / f'mitra_{half}.npz'
        if not path.exists():
            print('fitting Mitra', half, len(t), 'predicting', len(q), flush=True)
            p = fit(roster, t, q, y[t], counts)
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, train=t, labels=y[t], query=q, p=p,
                                    steps=counts.STEPS, checkpoint=counts.base.WEIGHT_SHA)
            tmp.replace(path)
        with np.load(path) as z:
            for key, value in [('train', t), ('labels', y[t]), ('query', q), ('steps', 15),
                               ('checkpoint', counts.base.WEIGHT_SHA)]:
                np.testing.assert_array_equal(z[key], value)
            print('finished Mitra', half, 'correct', int(((z['p'] >= .5) == y[q]).sum()), flush=True)


def fit_others(raw, y, groups, panel):
    import torch
    from full_roster_booking_validation_n23 import predictor
    from full_roster_metric_neighbors_n23 import train

    torch.set_num_threads(2)
    p = train(y, groups, panel)
    for family in ['cat', 'budget']:
        predict = predictor(family, raw, y)
        p[family] = np.empty(len(y))
        for half in (0, 1):
            t, q = np.flatnonzero(panel != half), np.flatnonzero(panel == half)
            assert not np.isin(groups[t], groups[q]).any()
            p[family][q] = predict(t, q, half)
    np.savez_compressed(DEST / 'others.npz', passenger=raw.PassengerId.to_numpy(str), panel=panel, y=y, **p)


def fit_family(family, raw, y, groups, panel):
    from full_roster_booking_validation_n23 import predictor

    predict = predictor(family, raw, y)
    p = np.empty(len(y))
    for h in (0, 1):
        train, query = np.flatnonzero(panel != h), np.flatnonzero(panel == h)
        assert not np.isin(groups[train], groups[query]).any()
        p[query] = predict(train, query, h)
        print(family, h, len(query), flush=True)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.savez_compressed(DEST / f'{family}.npz', passenger=raw.PassengerId.to_numpy(str), panel=panel, y=y, p=p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['mitra', 'others', 'tabicl', 'v44_without_name'])
    action = parser.parse_args().action
    raw, y, groups, fold = data()
    DEST.mkdir(exist_ok=True)
    if action in ('mitra', 'others'):
        (fit_mitra if action == 'mitra' else fit_others)(raw, y, groups, fold % 2)
    else:
        fit_family(action, raw, y, groups, fold % 2)
