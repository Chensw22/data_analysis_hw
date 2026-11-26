"""Evaluate agreement between the manifest model and the latent difficulty model on held bookings."""
import json

import numpy as np
import pandas as pd

from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT, ROOT
from nori_fare_party_n26 import FEES


def evaluate():
    neighbor_dir = ROOT/'artifacts/nov26_nori_adjacent_manifest'
    with np.load(OUT/'panel.npz') as z:
        ids, y, fold, known, common, group, base = [z[k] for k in
                    ('passenger','y','fold','known','common','groups','anchor')]
    n = len(y)
    base = base[:n]
    with np.load(neighbor_dir/'aligned.npz') as z:
        for k, value in [('passenger',ids[:n]), ('y',y), ('fold',fold), ('known',known),
                         ('reference161',base)]:
            np.testing.assert_array_equal(z[k],value)
        near, placebo, raw = [((z[k]>=.5)!=base)&z['observed'] for k in ('real','permuted','raw')]
    size = pd.Series(group[:n]).groupby(group[:n]).transform('size').to_numpy()
    scopes = dict(all=known, shared=common, solo=known&(size==1), pairs=known&(size==2),
                  larger=known&(size>=3))
    passenger = pd.read_csv(ROOT/'data/raw/train.csv', usecols=['PassengerId','CryoSleep',*FEES])
    np.testing.assert_array_equal(ids[:n],passenger.PassengerId)
    paid = passenger[FEES].gt(0).any(axis=1).to_numpy()
    scopes['shared_awake_paid_called_negative'] = common & passenger.CryoSleep.eq(False).to_numpy() & paid & ~base
    scopes['shared_sleep_no_positive_called_positive'] = common & passenger.CryoSleep.eq(True).to_numpy() & ~paid & base
    result = {}
    latent_votes = []
    for split in ('shuffled','order_sorted'):
        with np.load(OUT/f'{split}.npz') as z:
            simple,latent,logistic = ((z['p'][:n]>=.5)!=base[:,None]).T
        latent_votes.append(latent)
        rules = dict(neighbor_and_difficulty=near&latent, placebo_and_difficulty=placebo&latent,
                     original_and_difficulty=raw&latent,
                     both_new_information=near&~placebo&~raw&latent&~simple&~logistic)
        result[split]={k:{label:summary(act,base!=y,mask,fold) for label,mask in scopes.items()}
                       for k,act in rules.items()}
    path = ROOT/'reports/nov23_matched_learning_curve/paired_neighbor_difficulty.json'
    for split,r in result.items():
        print(split,[(k,v['all']['fixed'],v['all']['harmed']) for k,v in r.items()])
    stable = np.logical_and.reduce(latent_votes)
    partitions = dict(both=near&stable, neighbor_only=near&~stable, difficulty_only=stable&~near, neither=~near&~stable)
    result['coverage'] = {}
    for label,mask in scopes.items():
        cells = {k:summary(v,base!=y,mask,fold) for k,v in partitions.items()}
        assert sum(s['fixed'] for s in cells.values()) == np.count_nonzero(mask&(base!=y))
        assert sum(s['harmed'] for s in cells.values()) == np.count_nonzero(mask&(base==y))
        result['coverage'][label] = {k:dict(incumbent_wrong=s['fixed'],incumbent_correct=s['harmed'])
                                     for k,s in cells.items()}
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    evaluate()
