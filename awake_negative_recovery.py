"""Recover awake false negatives using local expenditure analogues and an additive outcome model."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from confirm_sleeper_scope import confirm
from exact_fold_error_selector import cached, counts
from exaone_validation_data import compact_features
from export_nov21_error_batch import test_probabilities
import export_consensus_routes as route
import relational_error_corrector as ref
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
OUT = ART / 'nov21_fn_recovery'
BILLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def matrices(frame):
    bill = frame[BILLS].to_numpy(float)
    missing = np.isnan(bill)
    bill = np.nan_to_num(bill)
    logbill = np.log1p(bill) / 6
    age = frame.Age.fillna(28).to_numpy() / 40
    categories = {'HomePlanet': ['Earth', 'Europa', 'Mars'], 'Deck': list('ABCDEFGT'),
                  'Side': ['P', 'S'], 'Destination': ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'],
                  'CryoSleep': ['False', 'True'], 'VIP': ['True', 'False']}
    flags = np.column_stack([frame[c].eq(value) for c, values in categories.items()
                             for value in values + ['__MISSING__']])
    # Compare positive-spend patterns and log amounts separately. Port/starboard and deck
    # can identify passengers with similar journeys without using names or party labels.
    profile = np.column_stack([flags, age, frame.Age.isna(), logbill, (bill > 0) * .5, missing * .5])
    beneficial = np.log1p(bill[:, 1] + bill[:, 2]) / 6
    adverse = np.log1p(bill[:, 0] + bill[:, 3] + bill[:, 4]) / 6
    difference = beneficial - adverse
    berth = frame.CabinNumber.fillna(900).to_numpy() / 1000
    additive = np.column_stack([profile, np.maximum(logbill - .9, 0), np.maximum(logbill - 1.25, 0),
        beneficial, adverse, np.abs(difference), beneficial * adverse, frame.Age.fillna(28) < 13,
        np.maximum(age-.5, 0), berth, flags * berth[:,None], flags * difference[:,None]])
    return profile, additive


def witnesses(train, y, query):
    t = train.CryoSleep.ne('True').to_numpy()
    tx, ta = matrices(train.iloc[np.flatnonzero(t)])
    qx, qa = matrices(query)
    nn = KNeighborsClassifier(n_neighbors=31, algorithm='brute', n_jobs=1).fit(tx, y[t])
    lr = LogisticRegression(C=1., max_iter=600).fit(ta, y[t])
    return np.column_stack([nn.predict_proba(qx)[:,1], lr.predict_proba(qa)[:,1]])


def decisions(frame, base, p, historical, witness):
    awake_negative = frame.CryoSleep.ne('True').to_numpy() & ~base & (p < .5)
    votes = (historical < .5).sum(1)
    # Neighbour counts can expose local pockets missed by all global model families.
    nearest = awake_negative & (votes == 5) & (witness[:,0] >= 19/31) & (witness[:,1] >= .5)
    # Pool additive evidence with the baseline probability for marginal negatives.
    # Neighbour majority and a majority of historical negatives define the target.
    additive = awake_negative & (votes >= 3) & (witness[:,0] > .5) & (witness[:,1] + p >= 1)
    return {'v131': base | nearest, 'v132': base | additive}


def run():
    inp, _, fold, bases, _ = ref.overlap.load_authenticated()
    dev, y = inp['dev'], inp['y'].astype(bool)
    x = inp['features'].iloc[dev]
    p = cached('v108_exact_oof/oof.npz','v108',dev,fold)
    original = cached('sleeper_consensus_scope.npz','original',dev,fold)
    manifest = cached('sleeper_manifest_risk.npz','manifest_berth_party_a',dev,fold)
    base = confirm(p>=.5,original,manifest)
    h = np.column_stack([bases[k] for k in FAMILIES])
    scores = np.empty((len(dev),2))
    for k in np.unique(fold):
        t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        t = t[~np.isin(inp['groups'][t], inp['groups'][q])]
        assert not np.intersect1d(inp['groups'][t],inp['groups'][q]).size
        scores[q] = witnesses(x.iloc[t], y[t], x.iloc[q])
    hard = pd.read_csv(ART/'error_rescue_20261117/catalog_rows.csv')
    np.testing.assert_array_equal(hard.row_id, dev)
    hard = hard.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp,bases,hard)
    predictions = decisions(x,base,p,h,scores)
    rows = pd.DataFrame(dict(row=dev, target=y, v128=base, hard=hard, persistent=persistent, fold=fold, **predictions))
    OUT.mkdir(exist_ok=True)
    rows.to_csv(OUT/'oof.csv',index=False)
    outcome = {k: counts(d,base,y,fold,hard,persistent) for k,d in predictions.items()}
    train = pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda s:s!='Name')
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda s:s!='Name')
    ids = test.PassengerId.to_numpy(str)
    q = compact_features(test)
    w = witnesses(compact_features(train), train.Transported.to_numpy(bool), q)
    deployed,_ = route.official_decisions('proposal_v128_manifest_confirmation',ids)
    tp = test_probabilities(route.BASE,ids)
    hist = np.column_stack([test_probabilities(k,ids) for k in route.HISTORICAL])
    for key, d in decisions(q,deployed,tp,hist,w).items():
        outcome[key]['test_changes'] = int((d!=deployed).sum())
        pd.DataFrame(dict(PassengerId=ids,Transported=d)).to_csv(OUT/(key+'.csv'),index=False)
    ref.write(ROOT/'reports/nov21_fn_recovery/result.json',dict(baseline_accuracy=float((y==base).mean()),results=outcome,
        development='Original three assessment folds. Removed all group overlaps from each expert training partition. Expert fits only on other folds, using all non-observed-sleepers. Hard mask used for evaluation only. Final fit on 8693 training rows after applying same observable filter.'))
    print(outcome,flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
