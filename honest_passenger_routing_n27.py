"""Learn which whole passenger populations should take an independent classifier's decision."""
import json
import numpy as np
import pandas as pd

from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, count, cohort

OUT = ART / 'nov27_honest_passenger_routing'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/honest_passenger_routing.json'


def best_action(rows, utility, edits):
    # Protect small, high-variance groups when comparing policies to doing nothing.
    gains = utility[rows].sum(axis=0)
    score = gains - np.sqrt(edits[rows].sum(axis=0))
    winner = score.argmax()
    return winner, score[winner]


def learn(structure, estimate, ask, features, edits, utility, depth):
    if depth:
        valid = []
        _, baseline = best_action(structure, utility, edits)
        for key, field in enumerate(features.T):
            left = field[structure]
            if min(left.sum(), (~left).sum()) < 45:
                continue
            gain = sum(best_action(r, utility, edits)[1] for r in (structure[left], structure[~left]))
            valid.append((gain, key))
        if valid and max(valid)[0] > baseline + np.sqrt(len(structure)) / 4:
            field = features[:, max(valid)[1]]
            out = np.zeros(len(ask), bool)
            for value in (False, True):
                take = field[ask] == value
                out[take] = learn(structure[field[structure] == value], estimate[field[estimate] == value],
                                  ask[take], features, edits, utility, depth - 1)
            return out
    action, _ = best_action(estimate, utility, edits)
    return edits[ask, action]


def run():
    raw, y, g, fold = data()
    _, blind, known, fresh = cohort(raw, y, fold)
    n = len(y)
    frame = pd.concat([raw, pd.read_csv(ROOT / 'data/raw/test.csv', usecols=raw.columns)], ignore_index=True)
    ids = frame.PassengerId.to_numpy(str)
    paid = frame[FEES].gt(0).any(axis=1)
    sleep = frame.CryoSleep.eq(True)
    group = frame.PassengerId.str.split('_').str[0]
    size = group.map(group.value_counts())
    with np.load(ART / 'nov27_dissent_likelihood_transfer/aligned.npz') as anchor:
        np.testing.assert_array_equal(anchor['passenger'], ids)
        np.testing.assert_array_equal(anchor['y'], y)
        first = anchor['first']
        prior = {s: anchor[s + '_median'] for s in ('random', 'contiguous')}
    with np.load(ART / 'nov26_latent_passenger_difficulty/panel.npz') as panel:
        np.testing.assert_array_equal(panel['passenger'], ids)
        p = panel['scores']
        opinions = np.column_stack([first, np.median(p[:, :18], axis=1) >= .5, p[:, -3:] >= .5,
                                    np.median(p[:, -3:], axis=1) >= .5])
    with np.load(ART / 'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'], ids)
        outsiders = []
        for key in ('dpt', 'tabnet'):
            a = z[key]
            outsiders.append((np.median(a, axis=1) if a.ndim == 2 else a) >= .5)
        opinions = np.column_stack([opinions, *outsiders])
    features = np.column_stack([first, paid, sleep, size > 1, size > 2, frame.HomePlanet.eq('Earth'),
                                frame.HomePlanet.eq('Europa'), frame.Cabin.str.startswith('G/').fillna(False),
                                frame.Destination.eq('TRAPPIST-1e'), frame.Age.lt(18),
                                frame[FEES].isna().any(axis=1), frame.Spa.gt(0), frame.VRDeck.gt(0),
                                frame.FoodCourt.gt(0)])
    edits = opinions != first[:, None]
    utility = edits[:n] * np.where(first[:n] == y, -1, 1)[:, None]
    scopes = {'all': known, 'unanimous': blind, 'fresh': fresh & known}
    report = {'groups': {}, 'rules': {}}
    saved = {}
    for scheme, protected in prior.items():
        current = first ^ protected
        err = current[:n] != y
        part = splits(g, fold, scheme)
        for key, mask in [('all', known), ('awake_paying', frame.CryoSleep.eq(False).to_numpy()[:n] & paid.to_numpy()[:n]),
                          ('asleep_unpaid', sleep.to_numpy()[:n] & ~paid.to_numpy()[:n]), ('former_unanimous', blind)]:
            valid = known & mask
            disagree = opinions[:n, 1:] != current[:n, None]
            shared = valid & ~disagree.any(axis=1)
            report['groups'].setdefault(key, {})[scheme] = dict(rows=int(valid.sum()), errors=int((valid & err).sum()),
                all_seven_agree_rows=int(shared.sum()), all_seven_wrong=int((shared & err).sum()),
                one_or_more_recovers=int((valid & err & ~shared).sum()), one_or_more_harms=int((valid & ~err & ~shared).sum()))
        for depth in (0, 1, 2):
            edits_all = np.zeros(len(ids), bool)
            test_votes = []
            for k in range(8):
                for j in range(4):
                    others = np.flatnonzero(known & (fold == k) & (part != j))
                    estimate = others[part[others] == ((j + 1) % 4)]
                    structure = others[part[others] != ((j + 1) % 4)]
                    q = np.flatnonzero((fold == k) & (part == j))
                    assert not np.isin(g[structure], g[estimate]).any()
                    assert not np.isin(g[others], g[q]).any()
                    request = np.r_[q, np.arange(n, len(ids))]
                    a = learn(structure, estimate, request, features, edits, utility, depth)
                    edits_all[q] = a[:len(q)]
                    test_votes.append(a[len(q):])
            edits_all[n:] = np.mean(test_votes, axis=0) > .5
            a = edits_all & ~protected
            delta = np.where(err, 1, -1) * a[:n]
            tag = f'depth_{depth}'
            saved[scheme + '_' + tag] = a
            report['rules'].setdefault(tag, {})[scheme] = dict(
                scopes={s:count(a[:n], mask, err) for s,mask in scopes.items()},
                folds=[int(delta[known & (fold == j)].sum()) for j in range(8)],
                test_changes=int(a[n:].sum()))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'aligned.npz', passenger=ids, fold=fold, y=y, **saved)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(report['groups'])
    print(report['rules'])


if __name__ == '__main__':
    run()
