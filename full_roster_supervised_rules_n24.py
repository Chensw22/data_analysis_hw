"""Learn shallow supervised rules and compare same-order and disjoint-order coefficient fits."""
import json

import numpy as np
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities


def design(raw):
    cost = raw[BILLS].fillna(0)
    total = cost.sum(axis=1).to_numpy()
    party = raw.PassengerId.str.split('_').str[0]
    return np.column_stack([passenger_vectors(raw), np.log1p(total) / 6,
                            cost.to_numpy() / np.maximum(1, total[:, None]),
                            np.log1p(party.map(party.value_counts()).to_numpy())])


def score(x, y, group, t, q):
    scores = {'boosted': [], 'same_orders': [], 'disjoint_orders': []}
    rule_counts = []
    for build, estimate in GroupKFold(2, shuffle=True, random_state=929).split(t, groups=group[t]):
        build, estimate = t[build], t[estimate]
        assert not np.isin(group[build], group[estimate]).any()
        forest = GradientBoostingClassifier(n_estimators=110, max_depth=3, min_samples_leaf=16,
                    learning_rate=.055, subsample=.8, random_state=716).fit(x[build], y[build])
        scores['boosted'].append(forest.predict_proba(x[q])[:, 1])
        ids = np.r_[build, estimate, q]
        subset = x[ids]
        # Each non-root node contributes a conjunction of all preceding splits.
        blocks = [sparse.csr_matrix(subset)]
        for tree in forest.estimators_.ravel():
            blocks.append(tree.decision_path(subset)[:, 1:])
        matrix = sparse.hstack(blocks, format='csr')
        ranges = np.split(np.arange(len(ids)), [len(build), len(build)+len(estimate)])
        for key, fit_rows in zip(['same_orders', 'disjoint_orders'], ranges[:2], strict=True):
            model = LogisticRegression(solver='liblinear', l1_ratio=1., C=.12, random_state=451,
                                       max_iter=1000).fit(matrix[fit_rows], y[ids[fit_rows]])
            assert model.n_iter_[0] < model.max_iter
            scores[key].append(model.predict_proba(matrix[ranges[2]])[:, 1])
            rule_counts.append(int(np.count_nonzero(model.coef_[0, x.shape[1]:])))
    return {k: np.mean(v, axis=0) for k, v in scores.items()}, rule_counts


def fit_folds(x, y, group, rows, fold):
    predictions = {k: np.zeros(len(rows)) for k in ('boosted', 'same_orders', 'disjoint_orders')}
    sizes = []
    for f in np.unique(fold):
        q = rows[fold == f]
        t = rows[(fold != f) & ~np.isin(group[rows], group[q])]
        assert not np.isin(group[t], group[q]).any()
        result, counts = score(x, y, group, t, q)
        for k in predictions:
            predictions[k][fold == f] = result[k]
        sizes.append(dict(training=len(t), query=len(q), nonzero_rules=counts))
    return predictions, sizes


def compare(p, base, y, group, fold, mask):
    d = ((p == y).astype(int) - (base == y))[mask]
    return dict(rows=int(mask.sum()), accuracy=float((p[mask]==y[mask]).mean()),
                fixed=int((d==1).sum()), harmed=int((d==-1).sum()),
                by_fold=[int(d[fold[mask]==f].sum()) for f in np.unique(fold[mask])], ci_pp=interval(d,group[mask]))


def run():
    _, y, g, rows, oldfold, _, champion, _, _, _, _, old, _ = aligned()
    raw, all_y, groups, fold = data()
    names, half, full = banks(raw, all_y, fold)
    base = latest_reference(all_y, groups, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, all_y, groups, fold)
    np.testing.assert_array_equal(gate>=.5, base)
    new_orders = ~np.isin(np.arange(len(raw)), rows) & ~np.isin(groups, g)
    assert new_orders.sum() == 1044
    shared = ((full>=.5) != all_y[:, None]).all(axis=1)
    paired = shared[rows] & (champion!=y)
    assert shared.sum()==257 and paired.sum()==201
    x = design(raw)
    historic, old_sizes = fit_folds(x, all_y, groups, rows, oldfold)
    current, sizes = fit_folds(x, all_y, groups, np.arange(len(raw)), fold)
    report = dict(historical_folds=old_sizes, eight_folds=sizes, methods={})
    for key in historic:
        methods = {}
        for name, weight in [('standalone',1.),('quarter',.25),('half',.5)]:
            a = historic[key]>=.5
            b = current[key]>=.5
            if weight < 1:
                a = np.where(champion==(old[:,0]>=.5), ((1-weight)*old[:,0]+weight*historic[key])>=.5,champion)
                b = ((1-weight)*gate+weight*current[key])>=.5
            methods[name] = dict(historical=compare(a,champion,y,g,oldfold,np.ones(len(y),bool)),
                    eightfold=compare(b,base,all_y,groups,fold,np.ones(len(raw),bool)),
                    new_orders=compare(b,base,all_y,groups,fold,new_orders),
                    common_fixed=int(((b==all_y)&shared).sum()),
                    common_fixed_both=int(((a==y)&(b[rows]==y)&paired).sum()),
                    harmed_both=int(((a!=y)&(b[rows]!=y)&(champion==y)&(base[rows]==y)).sum()))
        report['methods'][key] = methods
        print(key,methods['quarter'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/supervised_rules.json'
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
