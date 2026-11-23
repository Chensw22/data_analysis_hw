"""Learn when to follow a contrary neighbor vote using orders unseen by both stages."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data


def load_scores(raw, y, fold, path, keys):
    with np.load(path) as bank:
        for name, expected in [('passenger', raw.PassengerId), ('y', y), ('fold', fold)]:
            np.testing.assert_array_equal(bank[name], expected)
        return np.column_stack([bank[k] for k in keys])


def fit_queries(x, truth, threshold, conflict, train, query):
    ids = train[conflict[train]]
    ask = query[conflict[query]]
    assert len(np.unique(truth[ids])) == 2
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.10, max_iter=500))
    model.fit(x[ids], truth[ids])
    return ask[model.predict_proba(x[ask])[:, 1] >= threshold]


def run():
    raw, y, groups, fold = data()
    scores = np.column_stack([load_scores(raw, y, fold, OUT / f'{name}.npz', ['p'])[:, 0]
                              for name in (*FAMILIES, 'mitra96')])
    neighbors = load_scores(raw, y, fold, ROOT / 'artifacts/nov23_full_roster_metric_neighbors/oof.npz',
                            ['fixed', 'learned', 'focus'])
    base = scores[:, -1] >= .5
    pick = neighbors[:, -1] >= .5
    conflict = base != pick
    common = (base[:, None] == (scores >= .5)).all(1)
    orient = (2*pick.astype(int)-1)[:, None]
    base_inputs = np.column_stack([orient * logit(scores.clip(.001, .999)), pick])
    evidence = np.column_stack([base_inputs, orient*logit(neighbors.clip(.001, .999))])
    bills = raw[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
    _, codes, size = np.unique(groups, return_inverse=True, return_counts=True)
    context = np.column_stack([evidence, np.log1p(bills.fillna(0)), bills.isna(), raw.Age.fillna(-1),
                               raw.CryoSleep.eq(True), raw.CryoSleep.isna(), size[codes]])
    policies = {k: base.copy() for k in ['base_scores', 'neighbor_scores', 'neighbor_and_fields']}
    split_counts = []
    for outer in np.unique(fold):
        rows = np.flatnonzero(fold == outer)
        for a, b in GroupKFold(3, shuffle=True, random_state=SEED).split(rows, groups=groups[rows]):
            train, query = rows[a], rows[b]
            assert not np.isin(groups[query], groups[train]).any()
            split_counts.append(int(conflict[train].sum()))
            for name, x in zip(policies, [base_inputs, evidence, context], strict=True):
                swap = fit_queries(x, pick == y, .5, conflict, train, query)
                policies[name][swap] = pick[swap]
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as bank:
        dev = np.sort(bank['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    def counts(policy, mask):
        fixed = (policy == y) & (base != y) & mask
        harmed = (policy != y) & (base == y) & mask
        return dict(fixed=int(fixed.sum()), harmed=int(harmed.sum()))
    all_rows = np.ones(len(y), bool)
    result = dict(disagreements=int(conflict.sum()), raw_on_disagreement=counts(pick, conflict),
                  calibration_disagreements_per_split=split_counts, policies={})
    for name, policy in policies.items():
        delta = (policy == y).astype(int)-(base == y)
        result['policies'][name] = dict(all=counts(policy, all_rows), unanimity=counts(policy, common),
                historical=counts(policy, hard), additional=counts(policy, ~np.isin(np.arange(len(y)), dev)),
                accuracy=float((policy == y).mean()), group_bootstrap_pp=interval(delta, groups),
                fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)])
    path = ROOT / 'reports/nov23_matched_learning_curve/metric_adjudication.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(path.read_text())


if __name__ == '__main__':
    run()
