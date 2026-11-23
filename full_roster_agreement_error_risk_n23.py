"""Test residual prediction on the full range of expert agreement."""
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_unanimous_error_risk_n23 import CATEGORICAL, ROOT, banks, data, inputs, latest_reference


def agreement_band(count):
    return np.select([count == 19, count >= 16, count >= 13], [3, 2, 1], default=0)


def risk_scores(train, query, train_vote, count, other_count, y, group, fold, mode):
    tr_band, q_band = agreement_band(count), agreement_band(other_count)
    scores = np.full(len(y), np.nan)
    for f in np.unique(fold):
        donors = (fold % 2 == f % 2) & (fold != f)
        qfold = fold == f
        assert not np.isin(group[donors], group[qfold]).any()
        for band in np.unique(q_band[qfold]):
            q = qfold & (q_band == band)
            if mode == 'all':
                t = donors
                q = qfold
            elif mode == 'nearest':
                t = donors & (np.abs(tr_band - band) <= 1)
            else:
                t = donors & (tr_band == band)
            prep = ColumnTransformer([
                ('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
                ('numeric', StandardScaler(), train.columns.difference(CATEGORICAL).tolist()),
            ])
            model = make_pipeline(prep, LogisticRegression(C=.1, max_iter=500))
            model.fit(train.loc[t], y[t] != train_vote[t])
            scores[q] = model.predict_proba(query.loc[q])[:, 1]
            if mode == 'all':
                break
    assert np.isfinite(scores).all()
    return scores


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    train, train_vote, count = inputs(raw, half)
    query, vote, other_count = inputs(raw, full)
    band = agreement_band(other_count)
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    persistent = np.zeros(len(y), bool)
    persistent[dev] = hard_cohorts(dev, y[dev])[2]
    report = {'bands': [], 'policies': {}}
    for b in range(4):
        mask = band == b
        report['bands'].append(dict(rows=int(mask.sum()), vote_errors=int((mask & (vote != y)).sum()),
                                    reference_errors=int((mask & (reference != y)).sum())))
    for mode in ['all', 'nearest', 'same']:
        risk = risk_scores(train, query, train_vote, count, other_count, y, group, fold, mode)
        for action in ['only_correct_vote', 'replace']:
            prediction = reference.copy()
            flips = risk >= .5
            prediction[flips] = ~vote[flips]
            if action == 'replace':
                prediction[~flips] = vote[~flips]
            delta = (prediction == y).astype(int) - (reference == y)
            fixed, harmed = delta == 1, delta == -1
            report['policies'][f'{mode}_{action}'] = dict(
                accuracy=float(np.mean(prediction == y)), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                band_fixed=[int(fixed[band == b].sum()) for b in range(4)],
                band_harmed=[int(harmed[band == b].sum()) for b in range(4)],
                fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
                additional_net=int(delta.sum()-delta[dev].sum()),
                persistent_correct=int(((prediction == y) & persistent).sum()),
                persistent_fixed=int((fixed & persistent).sum()), persistent_harmed=int((harmed & persistent).sum()),
                booking_interval_pp=interval(delta, group),
            )
    path = ROOT / 'reports/nov23_matched_learning_curve/agreement_error_risk.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(report['bands'])
    for k,v in report['policies'].items():
        print(k,round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['band_fixed'],v['band_harmed'],v['additional_net'])


if __name__ == '__main__':
    run()
