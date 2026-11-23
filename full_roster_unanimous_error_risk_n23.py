"""Cross-fit probability of error on passengers with unanimous expert predictions."""
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference


def inputs(raw, scores):
    votes = (scores >= .5).sum(axis=1)
    direction = votes > scores.shape[1] // 2
    count = np.maximum(votes, scores.shape[1] - votes)
    x = compact_features(raw)
    for name in BILLS:
        x[name] = np.log1p(x[name])
    x['BookingSize'] = raw.PassengerId.str.split('_').str[0].map(
        raw.PassengerId.str.split('_').str[0].value_counts()
    )
    x['VotePositive'] = direction.astype(int)
    margins = np.where(direction[:, None], scores - .5, .5 - scores)
    for j, values in enumerate(margins.T):
        x[f'ExpertMargin{j}'] = values
    numeric = x.columns.difference(CATEGORICAL)
    x[numeric] = x[numeric].fillna(-1)
    return x, direction, count


def evaluate():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    training, train_vote, train_agreement = inputs(raw, half)
    query, query_vote, query_agreement = inputs(raw, full)
    unanimous = query_agreement == len(names)
    mistakes = query_vote != y
    paid = raw[BILLS].fillna(0).sum(axis=1).to_numpy() > 0
    assert unanimous.sum() == 3632 and mistakes[unanimous].sum() == 257
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        additional = np.ones(len(y), bool)
        additional[z['development']] = False
    results = {}
    for min_votes in [19, 16]:
        risk = np.full(len(y), np.nan)
        top_decile = np.zeros(len(y), bool)
        for f in np.unique(fold):
            t = (fold % 2 == f % 2) & (fold != f) & (train_agreement >= min_votes)
            q = (fold == f) & unanimous
            assert not np.isin(group[q], group[t]).any()
            prep = ColumnTransformer([
                ('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
                ('numeric', StandardScaler(), training.columns.difference(CATEGORICAL).tolist()),
            ])
            model = make_pipeline(prep, LogisticRegression(C=.1, max_iter=500))
            model.fit(training.loc[t], y[t] != train_vote[t])
            risk[q] = model.predict_proba(query.loc[q])[:, 1]
            qidx = np.flatnonzero(q)
            highest_first = qidx[np.argsort(-risk[q], kind='stable')]
            top_decile[highest_first[:int(np.ceil(q.sum() / 10))]] = True
        assert np.isfinite(risk[unanimous]).all()
        pred = reference.copy()
        selected = unanimous & (risk >= .5)
        pred[selected] = ~query_vote[selected]
        delta = (pred == y).astype(int) - (reference == y)
        results[str(min_votes)] = dict(
            accuracy=float((pred == y).mean()), selected=int(selected.sum()),
            fixed=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
            additional_fixed=int(((delta == 1) & additional).sum()),
            additional_harmed=int(((delta == -1) & additional).sum()),
            folds=[int(delta[fold == f].sum()) for f in np.unique(fold)],
            ranked_decile_size=int(top_decile.sum()),
            ranked_decile_errors=int((top_decile & mistakes).sum()),
            ranked_decile_paid_predicted_stayed_errors=int((top_decile & paid & ~query_vote & mistakes).sum()),
            ranked_decile_paid_predicted_stayed_total=int((top_decile & paid & ~query_vote).sum()),
        )
    table = []
    for direction in [False, True]:
        for positive_spending in [False, True]:
            mask = unanimous & (query_vote == direction) & (paid == positive_spending)
            table.append(dict(predicted_transported=direction, recorded_spending=positive_spending,
                              total=int(mask.sum()), wrong=int(mistakes[mask].sum())))
    out = dict(unanimous_groups=table, crossfit_risk=results)
    (ROOT / 'reports/nov23_matched_learning_curve/unanimous_error_risk.json').write_text(
        json.dumps(out, indent=2) + '\n'
    )
    print(results)


if __name__ == '__main__':
    evaluate()
