"""Choose the scope of a correction on bookings independent of each outer fold."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_agreement_error_risk_n23 import agreement_band, fit_risk, risk_scores
from full_roster_unanimous_error_risk_n23 import ROOT, banks, data, inputs, latest_reference


def choose_scope(train, train_vote, bands, y, group, fold, proposal, design, base, t):
    delta = np.zeros(len(y), int)
    evaluated = np.zeros(len(y), bool)
    for h in np.unique(fold[t]):
        a, b = t & (fold != h) & (fold % 2 == h % 2), t & (fold == h)
        assert not np.isin(group[a], group[b]).any()
        m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
        m.fit(design[a & proposal], y[a & proposal] != base[a & proposal])
        predicted = base.copy()
        selected = b & proposal
        predicted[selected] ^= m.predict_proba(design[selected])[:, 1] >= .5
        for band in range(4):
            qq = b & (bands == band)
            tt = a & (np.abs(bands - band) <= 1)
            prob = fit_risk(train, train, train_vote, y, tt, qq)
            flip = (prob >= .5) & (predicted[qq] == train_vote[qq])
            delta[qq] = np.where(flip, np.where(predicted[qq] == y[qq], -1, 1), 0)
            evaluated[qq] = True
    np.testing.assert_array_equal(evaluated, t)
    gains = np.array([delta[t & (bands == k)].sum() for k in range(4)])
    # Prefer less intervention when two candidate cutoffs give identical gain.
    cut = max(range(5), key=lambda k: (gains[k:].sum(), k))
    return gains, cut


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    train, train_vote, count = inputs(raw, half)
    query, vote, query_count = inputs(raw, full)
    bands, validation_bands = agreement_band(count), agreement_band(query_count)
    risk = risk_scores(train, query, train_vote, count, query_count, y, group, fold, 'nearest')
    change = (risk >= .5) & (reference == vote)
    evidence = [names.index(k) for k in EVIDENCE]
    proposal, design = proposal_design(half[:, :5], half[:, evidence])
    base = half[:, 4] >= .5
    policies = {label: reference.copy() for label in ('learn_bands', 'learn_cutoff', 'exploratory_16_plus')}
    inner = []
    for f in np.unique(fold):
        t = (fold % 2 == f % 2) & (fold != f)
        q = fold == f
        gains, cut = choose_scope(train, train_vote, bands, y, group, fold, proposal, design, base, t)
        inner.append(dict(outer_fold=int(f), gains=gains.tolist(), cutoff=cut))
        for key, enabled in [('learn_bands', np.where(gains > 0)[0]), ('learn_cutoff', range(cut, 4)),
                             ('exploratory_16_plus', [2, 3])]:
            active = q & change & np.isin(validation_bands, list(enabled))
            policies[key][active] ^= True
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    gains, cut = choose_scope(train, train_vote, bands, y, group, fold, proposal, design, base,
                              np.ones(len(y), bool))
    output = dict(inner=inner, deployment_scope=dict(gains=gains.tolist(), cutoff=cut), results={})
    for name, pred in policies.items():
        delta = (pred == y).astype(int) - (reference == y)
        output['results'][name] = dict(
            accuracy=float((pred == y).mean()), fixed=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
            fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
            additional_net=int(delta.sum() - delta[dev].sum()), booking_interval_pp=interval(delta, group),
        )
    (ROOT / 'reports/nov23_matched_learning_curve/nested_error_scope.json').write_text(
        json.dumps(output, indent=2) + '\n'
    )
    print(output)


if __name__ == '__main__':
    run()
