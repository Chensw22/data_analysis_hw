"""Test whether outcome dependence transfers companion evidence to held bookings."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs, summary
from full_roster_booking_validation_n23 import OUT, ROOT, data
from matched_bill_pair_states_n26 import vectors


def couple(a, b, log_odds):
    # One free cell determines the two-by-two table after both margins are fixed.
    low, high = np.maximum(0, a+b-1), np.minimum(a, b)
    ratio = np.exp(log_odds)
    for _ in range(38):
        mid = (low+high)/2
        below = mid*(1-a-b+mid) < ratio*(a-mid)*(b-mid)
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    both = (low+high)/2
    return np.stack([1-a-b+both, b-both, a-both, both], axis=-1)


def tree_posterior(size, edges, ratios, dependence, strong, independent):
    # Use the highest mutual information spanning tree, determined without outcomes.
    components = np.arange(size)
    states = (np.arange(2**size)[:, None] >> np.arange(size)) & 1
    potential = np.zeros(len(states))
    for e in np.argsort(-dependence, kind='stable'):
        i, j = edges[e]
        if components[i] == components[j]:
            continue
        components[components == components[j]] = components[i]
        potential += ratios[e, 2*states[:, i] + states[:, j]]
    assert len(set(components)) == 1
    p = softmax(states @ logit(strong) + potential) @ states
    check = softmax(states @ logit(independent) + potential) @ states
    np.testing.assert_allclose(check, independent, atol=1e-8)
    return p


def train_and_predict(raw, y, groups, fold):
    geom, bills = vectors(raw)
    x = np.column_stack([geom, bills])
    parties = [np.array(ix) for ix in pd.Series(np.arange(len(y))).groupby(groups).indices.values() if len(ix)>1]
    edges = np.concatenate([list(combinations(rows, 2)) for rows in parties])
    left, right = np.concatenate([edges, edges[:, ::-1]]).T
    assert (fold[left] == fold[right]).all()
    peer_count = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()-1
    features = np.column_stack([x[left], x[right], x[left]-x[right], peer_count[left]])
    pair_class = 2*y[left].astype(int)+y[right]
    with np.load(OUT/'mitra96.npz') as z:
        for key, expected in [('passenger', raw.PassengerId.to_numpy(str)), ('y', y), ('fold', fold)]:
            np.testing.assert_array_equal(z[key], expected)
        strong = z['p'].clip(.003, .997)
    joint = np.full((len(left), 4), np.nan)
    weak = np.zeros(len(y))
    for k in np.unique(fold):
        train, test = fold!=k, fold==k
        assert not np.isin(groups[train],groups[test]).any()
        opts = dict(max_iter=95, max_leaf_nodes=13, l2_regularization=4., early_stopping=False, random_state=1405)
        weak[test] = HistGradientBoostingClassifier(**opts).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        bank, query = train[left], test[left]
        model = HistGradientBoostingClassifier(**opts).fit(features[bank], pair_class[bank],
                    sample_weight=1/peer_count[left[bank]])
        np.testing.assert_array_equal(model.classes_, np.arange(4))
        joint[query] = model.predict_proba(features[query])
    n = len(edges)
    joint = (joint[:n] + joint[n:, [0, 2, 1, 3]])/2
    np.testing.assert_allclose(joint.sum(1), 1)
    odds = np.log(joint.clip(1e-12)) @ [1, -1, -1, 1]
    a, b = weak.clip(.003, .997)[edges].T
    independent = couple(a, b, np.zeros(n))
    coupled = couple(a, b, odds)
    ratios = np.log(coupled/independent)
    information = (coupled*ratios).sum(1)
    output = strong.copy()
    offset = 0
    for members in parties:
        local_edges = np.array(list(combinations(range(len(members)), 2)))
        span = slice(offset, offset+len(local_edges))
        np.testing.assert_array_equal(members[local_edges], edges[span])
        output[members] = tree_posterior(len(members), local_edges, ratios[span], information[span],
                            strong[members], weak[members].clip(.003, .997))
        offset += len(local_edges)
    assert offset == n
    labels = 2*y[edges[:,0]].astype(int)+y[edges[:,1]]
    def nll(table):
        return float(-np.log(table[np.arange(n),labels]).mean())
    diagnostics = dict(unordered_pairs=n, bookings=len(parties), weak_independent_pair_nll=nll(independent),
                       weak_dependent_pair_nll=nll(coupled), strong_individual_accuracy=float(((strong>=.5)==y).mean()))
    return strong, output, peer_count>0, diagnostics


def run():
    raw, y, g, fold = data()
    p, updated, multi, report = train_and_predict(raw, y, g, fold)
    _, current, _ = inputs()
    parent = current['parent']
    change = (p>=.5)!=(updated>=.5)
    policies = dict(relational_prediction=(updated>=.5)!=parent,
                    only_transferred_evidence=change&((updated>=.5)!=parent))
    wrong = parent!=y
    known = current['known']
    report['grouped_results'] = {}
    for name, mask in [('multi',known&multi), ('multi_unanimous',current['common']&multi),
                       ('multi_new_orders',known&current['new']&multi)]:
        report['grouped_results'][name] = {key:summary(value,wrong,mask,fold) for key,value in policies.items()}
        report['grouped_results'][name]['strong_before'] = summary((p>=.5)!=parent,wrong,mask,fold)
        report['grouped_results'][name]['after_vs_before'] = summary(change,(p>=.5)!=y,mask,fold)
    report['scope'] = ('Eight folds hold out complete bookings. Strong margins are adapted Mitra; weak margins and '
                       'pair interactions are trained on raw covariates. New orders are a subset of these same folds. '
                       'Common is agreement of eleven frozen candidate versions. Pair NLL weights unordered pairs equally.')
    target = ROOT/'reports/nov23_matched_learning_curve/booking_tree_messages.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
