"""Train inverse origin models on the old split with all query bookings excluded."""
import hashlib
import numpy as np
import torch

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT, data
from inverse_tabicl_manifest_n26 import conditional_observation, masked_view
from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer


def run():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    raw, truth, groups, _ = data()
    masked, attribute = masked_view(compact_features(raw), 'home_planet')
    with np.load(ROOT / 'artifacts/nov25_historical_nine/historical.npz') as history:
        rows, fold = history['rows'], history['fold']
        np.testing.assert_array_equal(history['passenger'], raw.PassengerId.iloc[rows])
    out = ROOT / 'artifacts/nov26_historical_inverse_planet'
    out.mkdir(exist_ok=True)
    for k in np.unique(fold):
        held = rows[fold == k]
        train = rows[(fold != k) & ~np.isin(groups[rows], groups[held])]
        assert not np.isin(groups[train], groups[held]).any()
        seed = 24671 + int(k)
        path = out / f'{k}.npz'
        if not path.exists():
            prior = infer(masked, truth, train, held, seed)
            real, levels = conditional_observation(masked, attribute, truth, train, held, seed, False)
            sham, other_levels = conditional_observation(masked, attribute, truth, train, held, seed, True)
            np.testing.assert_array_equal(levels, other_levels)
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, q=held, t=train, prior=prior, observed=real, shuffled=sham,
                                    passenger=raw.PassengerId.iloc[held].to_numpy(str), levels=levels,
                                    train_y=truth[train], seed=seed, checkpoint=DIGEST)
            temp.replace(path)
        with np.load(path) as cache:
            for key, expected in [('t',train),('q',held),('train_y',truth[train]),('seed',seed),
                                  ('checkpoint',DIGEST),('passenger',raw.PassengerId.iloc[held])]:
                np.testing.assert_array_equal(cache[key], expected)
        print('inverse planet', k, len(train), len(held), flush=True)


if __name__ == '__main__':
    run()
