"""Infer the recorded destination and deck jointly from an independent order's spending."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import DECKS, FEES, OUT as DECK, build_features

OUT = ROOT/'artifacts/nov26_fare_routes'
REPORT = ROOT/'reports/nov23_matched_learning_curve/fare_route_inference.json'


def summarize(mass, route, deck, dest):
    joint = mass.reshape(len(DECKS), -1)
    d = joint if deck < 0 else joint[deck:deck+1]
    destination = d.sum(0) / d.sum()
    decks = joint.sum(1) if dest < 0 else joint[:, dest]
    deck_probability = decks[deck] / decks.sum() if deck >= 0 else np.nan
    observed = destination[dest] if dest >= 0 else np.nan
    rate = mass[route] if route >= 0 else np.nan
    return np.r_[destination, observed, deck_probability, np.log(rate)]


def prepare():
    raw, y, _, _ = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns.tolist())
    people = pd.concat([raw, test], ignore_index=True)
    group = people.PassengerId.str.split('_').str[0].to_numpy()
    home = people.HomePlanet.fillna('?').to_numpy()
    destinations = pd.Index(sorted(people.Destination.dropna().unique()))
    dest = destinations.get_indexer(people.Destination)
    deck = pd.Index(DECKS).get_indexer(people.Cabin.str.split('/').str[0])
    route = deck * len(destinations) + dest
    route[(deck < 0) | (dest < 0)] = -1
    fee = people[FEES]
    total = fee.fillna(0).sum(axis=1).to_numpy(int)
    count = fee.gt(0).sum(axis=1).to_numpy()
    valid = (count > 0) & fee.notna().all(axis=1).to_numpy()
    cost = 2 * total + count
    sort_fold = np.empty(len(cost), int)
    columns = ['route_to_' + d for d in destinations] + ['route_registered_destination',
              'route_registered_deck', 'registered_route_log_density']
    variants = {k: np.full((len(cost), len(columns)), np.nan) for k in ['broad', 'exact', 'displaced']}
    width = len(DECKS) * len(destinations)
    for k, (t, q) in enumerate(GroupKFold(5, shuffle=True, random_state=9664).split(cost, groups=group)):
        assert not np.isin(group[t], group[q]).any()
        sort_fold[q] = k
        mature = t[valid[t] & (route[t] >= 0)]
        for h in np.unique(home[q]):
            pool = mature[home[mature] == h]
            if len(pool) < 25:
                pool = mature
            bank = pool[np.argsort(cost[pool], kind='stable')]
            prices = cost[bank]
            prior = .00002 * (1 + np.bincount(route[bank], minlength=width))
            for row in q[(home[q] == h) & valid[q]]:
                idx = np.searchsorted(prices, [cost[row] - 350, cost[row] + 351])
                near = bank[idx[0]:idx[1]]
                offset = cost[near] - cost[row]
                displaced = 2 * (((total[row] + 23) % 64) - (total[row] % 64))
                def smooth(sigma, shift):
                    weight = np.exp(-.5 * ((offset - shift) / sigma)**2) / (np.sqrt(2*np.pi) * sigma)
                    return np.bincount(route[near], weights=weight, minlength=width)
                broad = smooth(42, 0) + prior
                exact = .78 * smooth(2.4, 0) + .22 * broad
                moved_wide = smooth(42, displaced) + prior
                fake = broad * (.78 * smooth(2.4, displaced) + .22 * moved_wide) / moved_wide
                for key, mass in [('broad', broad), ('exact', exact), ('displaced', fake)]:
                    variants[key][row] = summarize(mass, route[row], deck[row], dest[row])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'routes.npz', passenger=people.PassengerId.to_numpy(str),
                        columns=columns, valid=valid, **variants)
    selected = np.flatnonzero(valid & (route >= 0))
    rates = {}
    for name, mat in variants.items():
        correct = mat[selected, :len(destinations)].argmax(1) == dest[selected]
        rates[name] = dict(rows=len(selected), accuracy=float(correct.mean()),
                           fold_accuracy=[float(correct[sort_fold[selected]==k].mean()) for k in range(5)],
                           nll=float(-np.log(mat[selected, len(destinations)]).mean()))
    (OUT/'destination.json').write_text(json.dumps(rates, indent=2)+'\n')
    print(rates, flush=True)


def route_features(raw, group):
    deck = build_features(raw, group)['fine']
    with np.load(OUT/'routes.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(raw)], raw.PassengerId)
        return {key: pd.concat([deck, pd.DataFrame(z[key][:len(raw)], columns=z['columns'])], axis=1)
                for key in ('broad', 'exact', 'displaced')}


def train(arch):
    raw, y, group, _ = data()
    variants = route_features(raw, group)
    if arch == 'neural':
        import torch
        from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
        torch.set_num_threads(2)
    else:
        from solo_selection_likelihood_n26 import fit_model
    for path in sorted(DECK.glob('neural_*.npz')):
        output = OUT/path.name.replace('neural', arch)
        with np.load(path) as base:
            t, q = base['t'], base['q']
            np.testing.assert_array_equal(y[t], base['y'])
        assert not np.isin(group[t], group[q]).any()
        k = int(path.stem.split('_')[-1])
        if not output.exists():
            predictions = {}
            for name, x in variants.items():
                if arch == 'neural':
                    predictions[name] = infer(x, y, t, q, 5583+k)
                else:
                    predictions[name] = fit_model(x.iloc[t], y[t]).predict_proba(x.iloc[q])[:,1]
            with output.with_suffix('.tmp').open('wb') as fp:
                np.savez_compressed(fp, t=t, q=q, **predictions)
            output.with_suffix('.tmp').replace(output)
        print(output.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'tree', 'neural'])
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    else:
        train(action)
