"""Cross-fit a discriminative combination of the class evidence from individual manifest fields and pairs."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT
from manifest_tree_mixture_n27 import class_tables, discretize
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs

MODES = ('single_fields', 'registration_by_activity', 'all_field_pairs')


def evidence(x, y, train, query, pairs):
    blocks = []
    for label in (False, True):
        marginals, factors, _ = class_tables(x, train[y[train] == label], x.max(axis=0) + 1, pairs)
        blocks.append(np.column_stack(
            [np.log(table[x[query, j]]) for j, table in enumerate(marginals)]
            + [np.log(table[x[query, a], x[query, b]]) for table, (a, b) in zip(factors, pairs)]))
    return blocks[1] - blocks[0]


def train_predict(x, y, groups, train, query):
    width = x.shape[1]
    pairs = list(zip(*np.triu_indices(width, 1)))
    reference = np.empty((len(train), width + len(pairs)))
    query_inputs = []
    for bank, held in GroupKFold(3, shuffle=True, random_state=293).split(train, groups=groups[train]):
        source = train[bank]
        destinations = np.r_[train[held], query]
        assert not np.isin(groups[source], groups[destinations]).any()
        features = evidence(x, y, source, destinations, pairs)
        reference[held] = features[:len(held)]
        query_inputs.append(features[len(held):])
    # CryoSleep is field 3. The last five fields are facility charges.
    activity = {3, *range(width - len(FEES), width)}
    cross = [width + j for j, (a, b) in enumerate(pairs) if (a in activity) != (b in activity)]
    column_sets = [np.arange(width), np.r_[np.arange(width), cross], np.arange(reference.shape[1])]
    outputs = []
    for cols in column_sets:
        model = LogisticRegression(C=0.1, max_iter=400).fit(reference[:, cols], y[train])
        outputs.append(np.mean([model.predict_proba(z[:, cols])[:, 1] for z in query_inputs], axis=0))
    return np.column_stack(outputs)


def run():
    raw, frame, y, groups, fold, first, correction, _, ordinary, _ = inputs()
    n = len(y)
    ids = frame.PassengerId.to_numpy(str)
    with np.load(ART/'nov27_manifest_tree_mixture/scores.npz') as z:
        for key, value in [('passenger', ids), ('fold', fold), ('y', y)]:
            np.testing.assert_array_equal(z[key], value)
        contiguous = z['contiguous']
    parent = first ^ correction
    wrong = parent[:n] != y
    _, _, known, fresh = cohort(raw, y, fold)
    shared = known & ((ordinary[:n] >= 0.5) == parent[:n, None]).all(axis=1)
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    scopes = dict(all=known, fresh=known & fresh, unanimous=shared,
                  awake_paid=shared & raw.CryoSleep.eq(False).to_numpy() & paid,
                  sleep_zero=shared & raw.CryoSleep.eq(True).to_numpy() & ~paid)
    direction = np.where(parent, 1., -1.)
    odds = lambda p: logit(np.clip(p, 0.002, 0.998))
    base_margin = direction * np.abs(odds(np.median(ordinary[:, 1:4], axis=1)))
    x = discretize(frame)
    keys = frame.PassengerId.str.split('_').str[0].to_numpy()
    saved, report = {}, {}
    for scheme, partition in [('random', fold), ('ordered', contiguous)]:
        scores = np.empty((len(frame), len(MODES)))
        for k in range(9):
            train = np.flatnonzero(partition != k)
            query = np.flatnonzero(partition == k) if k < 8 else np.arange(n, len(frame))
            scores[query] = train_predict(x, y, keys, train, query)
        saved[scheme] = scores
        votes = scores >= 0.5
        report[scheme] = dict(scores={name:dict(accuracy=float(np.mean(votes[:n, j] == y)),
                                 log_loss=float(log_loss(y, scores[:n, j]))) for j, name in enumerate(MODES)}, rules={})
        rules = {}
        for j in (1, 2):
            change = votes[:, j] != parent
            rules[MODES[j]] = change
            rules[MODES[j]+'_extra'] = change & ((base_margin + odds(scores[:, j]) - odds(scores[:, 0]) >= 0) != parent)
        rules['both_increment'] = rules['registration_by_activity_extra'] & rules['all_field_pairs_extra']
        for name, flag in rules.items():
            saved[scheme+'_'+name] = flag
            delta = flag[:n] * np.where(wrong, 1, -1)
            report[scheme]['rules'][name] = dict(
                scopes={scope:count(flag[:n], rows, wrong) for scope, rows in scopes.items()},
                fold_net=[int(delta[known & (partition == k)].sum()) for k in range(8)], test=int(flag[n:].sum()))
    np.testing.assert_allclose(saved['random'][n:], saved['ordered'][n:])
    report['both_protocols'] = {}
    for name in rules:
        repeated = saved['random_'+name][:n] & saved['ordered_'+name][:n]
        report['both_protocols'][name] = {scope:count(repeated, rows, wrong) for scope, rows in scopes.items()}
    out = ART/'nov27_field_pair_evidence'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz', passenger=ids, y=y, fold=fold, contiguous=contiguous, **saved)
    (ROOT/'reports/nov23_matched_learning_curve/field_pair_evidence.json').write_text(json.dumps(report, indent=2)+'\n')
    for scheme in ('random', 'ordered'):
        print(scheme, report[scheme]['scores'], report[scheme]['rules']['both_increment'], flush=True)
    print('repeated', report['both_protocols'])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
