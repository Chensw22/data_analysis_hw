"""Repeat the original eight-fold TabICL evaluation with all eligible labeled orders."""
import hashlib

import numpy as np
import pandas as pd
import torch

from matched_booking_learning_curve_n23 import ART, ROOT, SEED, portions
from tabicl_routed_context_n23 import CAT, DIGEST, WEIGHT, infer


def run():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    torch.set_num_threads(2)
    roster = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=lambda c: c != 'Name')
    y = roster.pop('Transported').to_numpy(bool)
    groups = roster.PassengerId.str.split('_').str[0].to_numpy()
    x = roster.drop(columns=['Cabin', 'PassengerId']).copy()
    x[['Deck', 'CabinNumber', 'Side']] = roster.Cabin.str.split('/', expand=True)
    x.CabinNumber = pd.to_numeric(x.CabinNumber)
    x[CAT] = x[CAT].fillna('__MISSING__').astype(str)
    with np.load(ART / 'nov23_matched_learning_curve/tabicl.npz') as cache:
        row, fold, old = cache['row'], cache['fold'], cache['large']
        np.testing.assert_array_equal(cache['y'], y[row])
    extra = np.setdiff1d(np.arange(len(y)), row)
    output = ART / 'nov23_expanded_training_roster'
    output.mkdir(exist_ok=True)
    expanded = np.full(len(row), np.nan)
    sizes = []
    for f, _, t, q in portions(groups[row]):
        np.testing.assert_array_equal(np.flatnonzero(fold == f), q)
        add = extra[~np.isin(groups[extra], groups[row[q]])]
        train = np.r_[row[t], add]
        np.testing.assert_array_equal(np.sort(train), np.flatnonzero(~np.isin(groups, groups[row[q]])))
        if f == 0:
            np.testing.assert_allclose(infer(x, y, row[t], row[q], SEED+f), old[q], rtol=0, atol=1e-6)
        expanded[q] = infer(x, y, train, row[q], SEED+f)
        sizes.append([len(t), len(add), len(train)])
        print('fold', f, 'context', len(train), flush=True)
    assert np.isfinite(expanded).all()
    np.savez_compressed(output / 'eight_tabicl.npz', row=row, y=y[row], fold=fold, initial=old,
                        expanded=expanded, sizes=sizes)


if __name__ == '__main__':
    run()
