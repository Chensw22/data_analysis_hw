"""Recreate deployed raw out-of-fold teacher scores inside three group-disjoint outer splits."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from full_roster_booking_validation_n23 import data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/nov25_grouped_peer_teacher'
OUTER = (0, 1, 2)


def run():
    import torch
    import mitra_candidate as base
    from prepare_residual_ready import load_frames, probability, SEED

    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    raw, y, group, folds = data()
    train, _, labels, features = load_frames()
    np.testing.assert_array_equal(raw.PassengerId, train.PassengerId)
    np.testing.assert_array_equal(y, labels)
    OUT.mkdir(exist_ok=True)
    for outer in OUTER:
        t, q = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        assert not np.intersect1d(group[t], group[q]).size
        scores = np.full(len(t), np.nan, dtype=np.float32)
        inner = np.full(len(t), -1)
        splitter = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for f, (a, b) in enumerate(splitter.split(t, y[t])):
            context, query = t[a], t[b]
            path = OUT / f'outer{outer}_inner{f}.npz'
            if not path.exists():
                model = base.fit_mitra(features.iloc[context], labels[context], context, SEED + f,
                                       'mps', 'zero-shot', query_chunk=2318)
                p = probability(model.predict_proba(features.iloc[query])[:, 1], len(query))
                model.release()
                del model
                base.release_memory()
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, context=context, query=query, labels=labels[context],
                                        p=p, seed=SEED+f, checkpoint=base.WEIGHT_SHA)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key, expected in [('context', context), ('query', query), ('labels', y[context]),
                                      ('seed', SEED+f), ('checkpoint', base.WEIGHT_SHA)]:
                    np.testing.assert_array_equal(z[key], expected)
                scores[b] = probability(z['p'], len(b))
            inner[b] = f
            print('outer', outer, 'inner', f, 'complete', flush=True)
        assert np.isfinite(scores).all() and (inner >= 0).all()
        path = OUT / f'outer{outer}.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, t=t, q=q, labels=labels[t], fold=inner, p=scores,
                                passenger=raw.PassengerId.iloc[t].to_numpy(str), checkpoint=base.WEIGHT_SHA)
        path.with_suffix('.tmp').replace(path)


if __name__ == '__main__':
    run()
