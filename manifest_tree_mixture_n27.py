"""Average class-conditional manifest densities over all spanning trees of the fields."""
import json

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs


def discretize(frame):
    cabin = frame.Cabin.str.split('/', expand=True)
    groups = frame.PassengerId.str.split('_').str[0]
    fields = [frame.HomePlanet, frame.Destination, frame.VIP, frame.CryoSleep, cabin[0], cabin[2]]
    for values, edges in [(frame.Age, [6, 13, 18, 25, 35, 55]),
                          (pd.to_numeric(cabin[1]), [150, 350, 550, 800, 1100, 1450]),
                          (groups.map(groups.value_counts()), [2, 3, 5])]:
        fields.append(pd.cut(values, [-np.inf, *edges, np.inf], right=False))
    for name in FEES:
        fields.append(pd.cut(frame[name], [-np.inf, 0, 80, 300, 700, 1200, 2500, np.inf]))
    # Missing is an explicit observed field state, shared by all three models.
    return np.column_stack([pd.factorize(field, sort=True)[0] + 1 for field in fields])


def tree_log_sum(weights):
    laplacian = -weights.copy()
    diagonal = np.arange(weights.shape[-1])
    laplacian[..., diagonal, diagonal] = weights.sum(axis=-1)
    sign, value = np.linalg.slogdet(laplacian[..., :-1, :-1])
    assert (sign > 0).all() and np.isfinite(value).all()
    return value


def class_tables(x, train, sizes, pairs):
    # The uniform pseudo-count has the same total in every pair and singleton.
    mass = 8.0
    marginals = [(np.bincount(x[train, j], minlength=size) + mass/size) / (len(train) + mass)
                 for j, size in enumerate(sizes)]
    factors, information = [], []
    for a, b in pairs:
        shape = (sizes[a], sizes[b])
        bins = x[train, a] * sizes[b] + x[train, b]
        counts = np.bincount(bins, minlength=shape[0]*shape[1]).reshape(shape)
        joint = (counts + mass/counts.size) / (len(train) + mass)
        factor = joint / np.outer(marginals[a], marginals[b])
        factors.append(factor)
        information.append(np.sum(joint * np.log(factor)))
    return marginals, factors, information


def predict(x, y, train, query):
    width = x.shape[1]
    sizes = x.max(axis=0) + 1
    left, right = np.triu_indices(width, 1)
    pairs = list(zip(left, right))
    tables = [class_tables(x, train[y[train] == label], sizes, pairs) for label in [False, True]]
    prevalence = np.mean(y[train])
    info = (1 - prevalence) * np.array(tables[0][2]) + prevalence * np.array(tables[1][2])
    # Temper the preference for dependent pairs. Classes share this graph prior.
    weights = np.zeros((width, width))
    weights[left, right] = weights[right, left] = np.exp(12 * info)
    mst = minimum_spanning_tree(-weights).toarray() != 0
    selected = (mst | mst.T)[left, right]
    assert selected.sum() == width - 1
    denom = tree_log_sum(weights)
    by_class = []
    for label, (marginals, factors, _) in enumerate(tables):
        prior = prevalence if label else 1 - prevalence
        independent = np.log(prior) + sum(np.log(m[x[query, j]]) for j, m in enumerate(marginals))
        ratio = np.column_stack([factor[x[query, a], x[query, b]] for factor, (a, b) in zip(factors, pairs)])
        tree = independent + np.log(ratio[:, selected]).sum(axis=1)
        mixture = independent.copy()
        for chunk in np.array_split(np.arange(len(query)), max(1, (len(query)+255)//256)):
            adjusted = np.zeros((len(chunk), width, width))
            adjusted[:, left, right] = adjusted[:, right, left] = ratio[chunk] * weights[left, right]
            mixture[chunk] += tree_log_sum(adjusted) - denom
        by_class.append(np.column_stack([independent, tree, mixture]))
    return expit(by_class[1] - by_class[0])


def run():
    # For three fields the nonnegative determinant must equal the sum over three trees.
    example = np.array([[0., 2., 3.], [2., 0., 5.], [3., 5., 0.]])
    np.testing.assert_allclose(np.exp(tree_log_sum(example)), 2*3 + 2*5 + 3*5)
    raw, frame, y, groups, fold, first, correction, _, ordinary, _ = inputs()
    _, _, known, fresh = cohort(raw, y, fold)
    anchor = first ^ correction
    n = len(y)
    wrong = anchor[:n] != y
    common = known & ((ordinary[:n] >= 0.5) == anchor[:n, None]).all(axis=1)
    positive = raw[FEES].gt(0).any(axis=1).to_numpy()
    scopes = dict(all=known, fresh=known & fresh, unanimous=common,
                  awake_paid=common & raw.CryoSleep.eq(False).to_numpy() & positive,
                  sleep_zero=common & raw.CryoSleep.eq(True).to_numpy() & ~positive)
    assert sum(wrong & scopes['awake_paid']) == 476 and sum(wrong & scopes['sleep_zero']) == 300
    x = discretize(frame)
    order = frame.PassengerId.str.split('_').str[0].to_numpy()
    contiguous = np.zeros(n, dtype=int)
    for i, part in enumerate(np.array_split(np.unique(groups), 8)):
        contiguous[np.isin(groups, part)] = i
    labels = ['independent', 'single_tree', 'tree_mixture']
    saved, report = {}, {}
    for scheme, partition in [('random', fold), ('ordered', contiguous)]:
        scores = np.empty((len(frame), len(labels)))
        for k in range(9):
            train = np.flatnonzero(partition != k)
            query = np.flatnonzero(partition == k) if k < 8 else np.arange(n, len(frame))
            assert not np.isin(order[query], order[train]).any()
            scores[query] = predict(x, y, train, query)
        saved[scheme] = scores
        votes = scores >= 0.5
        mixture_change = votes[:, 2] != anchor
        opinion = np.median(ordinary[:, 1:4], axis=1)
        odds = lambda p: logit(np.clip(p, 0.002, 0.998))
        baseline = np.where(anchor, 1., -1.) * abs(odds(opinion))
        delta = odds(scores[:, 2]) - odds(scores[:, 1])
        rules = {
            'mixture': mixture_change,
            'unique_to_mixture': mixture_change & (votes[:, 1] == anchor) & (votes[:, 0] == anchor),
            'increment_vs_tree': mixture_change & ((baseline + delta >= 0) != anchor),
        }
        report[scheme] = dict(scores={label: dict(accuracy=float(np.mean(votes[:n, j] == y)),
                                  log_loss=float(log_loss(y, scores[:n, j]))) for j, label in enumerate(labels)},
                              edits={})
        for name, mask in rules.items():
            saved[scheme+'_'+name] = mask
            net = mask[:n] * np.where(wrong, 1, -1)
            report[scheme]['edits'][name] = dict(
                scopes={scope: count(mask[:n], rows, wrong) for scope, rows in scopes.items()},
                fold_net=[int(net[known & (partition == k)].sum()) for k in range(8)], test=int(mask[n:].sum()))
    np.testing.assert_allclose(saved['random'][n:], saved['ordered'][n:])
    report['both_protocols'] = {}
    for name in rules:
        repeated = saved['random_'+name][:n] & saved['ordered_'+name][:n]
        report['both_protocols'][name] = {scope: count(repeated, rows, wrong) for scope, rows in scopes.items()}
    path = ART/'nov27_manifest_tree_mixture'
    path.mkdir(exist_ok=True)
    np.savez_compressed(path/'scores.npz', passenger=frame.PassengerId.to_numpy(str), y=y,
                        fold=fold, contiguous=contiguous, **saved)
    (ROOT/'reports/nov23_matched_learning_curve/manifest_tree_mixture.json').write_text(json.dumps(report, indent=2)+'\n')
    for scheme in ('random', 'ordered'):
        print(scheme, report[scheme]['scores'], report[scheme]['edits']['increment_vs_tree'], flush=True)
    print('both', report['both_protocols'])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
