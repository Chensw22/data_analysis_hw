"""Learn the current correction model's residuals on all passengers, with inner predictions."""
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_unanimous_error_risk_n23 import CATEGORICAL, ROOT, banks, data, inputs, latest_reference


def choose_reference(scores, design, proposal, y, t, q):
    initial = scores[:, 4] >= .5
    learner = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
    learner.fit(design[t & proposal], (y != initial)[t & proposal])
    flips = np.zeros(len(scores), bool)
    flips[q & proposal] = learner.predict_proba(design[q & proposal])[:, 1] >= .5
    return (initial ^ flips)[q]


def orient(frame, majority, reference, count):
    x = frame.copy()
    margins = x.columns[x.columns.str.startswith('ExpertMargin')]
    x[margins] = x[margins].to_numpy() * np.where(majority == reference, 1, -1)[:, None]
    x['ReferencePositive'] = reference.astype(int)
    x['AgreementCount'] = count
    return x


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    train, tr_vote, count = inputs(raw, half)
    outer, vote, other_count = inputs(raw, full)
    outer = orient(outer, vote, reference, other_count)
    evidence = [names.index(k) for k in EVIDENCE]
    proposal, design = proposal_design(half[:, :5], half[:, evidence])
    modes = ('linear', 'tree', 'consensus')
    preds = {k: reference.copy() for k in modes}
    for f in np.unique(fold):
        t, q = (fold != f) & (fold % 2 == f % 2), fold == f
        ref_t = np.zeros(len(y), bool)
        for h in np.unique(fold[t]):
            a, b = t & (fold != h), t & (fold == h)
            assert not np.isin(group[a], group[b | q]).any()
            ref_t[b] = choose_reference(half, design, proposal, y, a, b)
        x = orient(train.loc[t], tr_vote[t], ref_t[t], count[t])
        prep = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
            ('num', StandardScaler(), x.columns.difference(CATEGORICAL).tolist())
        ])
        a, b = prep.fit_transform(x), prep.transform(outer.loc[q])
        target = ref_t[t] != y[t]
        flags = []
        for k, model in [
            ('linear', LogisticRegression(C=.1, max_iter=500)),
            ('tree', HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=7, learning_rate=.06,
                                                   min_samples_leaf=35, l2_regularization=2,
                                                   early_stopping=False, random_state=23))]:
            correction = model.fit(a, target).predict_proba(b)[:, 1] >= .5
            preds[k][q] ^= correction
            flags.append(correction)
        preds['consensus'][q] ^= flags[0] & flags[1]
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    persistent = np.zeros(len(y), bool)
    persistent[dev] = hard_cohorts(dev, y[dev])[2]
    all_missed = ((full >= .5) != y[:, None]).all(axis=1)
    report = {}
    for k,pred in preds.items():
        delta = (pred == y).astype(int) - (reference == y)
        report[k] = dict(accuracy=float((pred == y).mean()), fixed=int((delta == 1).sum()),
                         harmed=int((delta == -1).sum()), persistent_fixed=int(((delta == 1) & persistent).sum()),
                         of_257=int(((pred == y) & all_missed).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
                         fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
                         interval_pp=interval(delta, group))
    (ROOT / 'reports/nov23_matched_learning_curve/direct_reference_risk.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
