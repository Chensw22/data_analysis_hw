"""Remove each reported service from every TabICL example and each complete query."""
import argparse
import hashlib
import os

import numpy as np

from exaone_validation_data import compact_features, CATEGORICAL
from masked_reference_receipts_n27 import bank_for, FEES, ART, ROOT
from matched_state_context_n26 import inputs

STORE = ART / 'nov27_complete_receipt_counterfactual'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/complete_receipt_counterfactual.json'


def splits(frame, y, groups, fold):
    complete = frame[FEES].notna().all(axis=1).to_numpy()
    for k in range(9):
        train = np.flatnonzero(fold != k) if k < 8 else np.arange(len(y))
        query = np.flatnonzero(fold == k) if k < 8 else np.arange(len(y), len(frame))
        query = query[complete[query]]
        assert not np.isin(groups[train], groups[query]).any()
        yield k, train, query


def source_folds(groups, original, protocol):
    if protocol == 'random':
        return original
    g = groups[:len(original)]
    out = np.full(len(g), -1)
    for k, chunk in enumerate(np.array_split(np.unique(g), 8)):
        out[np.isin(g, chunk)] = k
    assert np.all(out >= 0)
    return out


def data_dir(replica, protocol, fold):
    return STORE / (f'bank{replica}' if protocol == 'random' or fold == 8 else f'ordered_bank{replica}')


def train(replica, protocol, folds=None):
    import torch
    from tabicl_routed_context_n23 import WEIGHT, DIGEST, infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    torch.set_num_threads(2)
    frame, y, g, f, _ = inputs()
    x = compact_features(frame)
    assert set(x.columns) == set([*FEES, *CATEGORICAL, 'Age', 'CabinNumber'])
    for k, t, q in splits(frame, y, g, source_folds(g, f, protocol)):
        if folds is not None and k not in folds:
            continue
        if k == 8 and protocol == 'ordered':
            continue  # The full-training prediction is shared between validation protocols.
        dest = data_dir(replica, protocol, k)
        dest.mkdir(parents=True, exist_ok=True)
        bank = bank_for(t, g, k, replica)
        for name in ['original', *FEES]:
            path = dest / f'{k}_{name}.npz'
            if path.exists():
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['q'], q)
                    np.testing.assert_array_equal(z['bank'], bank)
                    np.testing.assert_array_equal(z['ids'], frame.PassengerId.iloc[q])
                continue
            view = x if name == 'original' else x.drop(columns=name)
            p = infer(view, y, bank, q, 7233 + k)
            assert np.isfinite(p).all()
            tmp = dest / f'{k}_{name}.{os.getpid()}.tmp.npz'
            np.savez_compressed(tmp, bank=bank, q=q, ids=frame.PassengerId.iloc[q].to_numpy(str),
                                labels=y[bank], checkpoint=DIGEST, probability=p)
            tmp.replace(path)
        print('bank', replica, 'fold', k, 'complete', len(q), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('replica', type=int)
    ap.add_argument('--protocol', choices=['random','ordered'], default='random')
    ap.add_argument('--folds', type=int, nargs='+', choices=range(9))
    args = ap.parse_args()
    train(args.replica, args.protocol, args.folds)
