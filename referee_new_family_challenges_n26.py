"""Train on cases where a new architecture challenges the established vote, then test whole bookings."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs, summary
from full_roster_booking_validation_n23 import data
from joint_new_architectures_n26 import ART, REPORT, load_models


def scale_predictions(raw, y, groups, fold, full):
    columns = []
    for size in [384, 1536]:
        pred = np.full(len(y), np.nan)
        folder = ART / 'nov26_nori'
        if size == 1536:
            folder = folder / 'context_1536'
        for k in np.unique(fold):
            q = np.flatnonzero(fold == k)
            with np.load(folder / f'fold{k}.npz') as z:
                np.testing.assert_array_equal(z['q'], q)
                np.testing.assert_array_equal(z['ids'], raw.PassengerId.iloc[q].to_numpy(str))
                np.testing.assert_array_equal(z['labels'], y[z['context']])
                assert len(np.unique(z['context'])) == size and not np.isin(groups[q], groups[z['context']]).any()
                pred[q] = z['p']
        columns.append(pred)
    return np.column_stack([*columns, full])


def run():
    raw, y, g, f = data()
    _, cur, _ = inputs()
    old, new = load_models(raw, y, g, f, 0)
    old_scores = np.column_stack(list(old.values()))
    new_scores = np.column_stack(list(new.values()))
    odds = lambda p: logit(p.clip(.003, .997))
    historical = odds(old_scores)
    # A label-independent anchor lets the same referee learn from calibration passengers
    # without importing another calibration passenger's label through the queued release.
    ref = historical.mean(1) >= 0
    sign = np.where(ref, -1., 1.)
    previous = historical * sign[:, None]
    challengers = odds(new_scores) * sign[:, None]
    contest = (challengers > 0).any(axis=1)
    sequence = odds(scale_predictions(raw, y, g, f, new_scores[:, 0])) * sign[:, None]
    old_info = np.column_stack([ref, previous.mean(1), previous.std(1), previous.max(1),
                                previous[:, 0], (previous > 0).mean(1)])
    full = np.column_stack([old_info, challengers, sequence[:, 1] - sequence[:, 0],
                            sequence[:, 2] - sequence[:, 1]])
    assert np.isfinite(full).all()
    scopes = dict(all=cur['known'], common=cur['common'], fresh=cur['known'] & cur['new'])
    report = dict(anchor_models=list(old), new_models=list(new), protocols={})
    target = y != ref
    for protocol in ['shuffled', 'ordered']:
        predictions = np.zeros((len(y), 2), bool)
        covered = np.zeros(len(y), bool)
        train_counts = []
        for k in np.unique(f):
            ix = np.flatnonzero(f == k)
            if protocol == 'ordered':
                splits = [(ix[~np.isin(g[ix], b)], ix[np.isin(g[ix], b)]) for b in np.array_split(np.unique(g[ix]), 4)]
            else:
                splits = [(ix[t], ix[q]) for t, q in GroupKFold(4, shuffle=True, random_state=412+int(k)).split(ix, groups=g[ix])]
            for t, q in splits:
                assert not np.isin(g[t], g[q]).any() and not covered[q].any()
                covered[q] = True
                t, q = t[contest[t]], q[contest[q]]
                train_counts.append(len(t))
                for j, x in enumerate([old_info, full]):
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
                    model.fit(x[t], target[t])
                    predictions[q, j] = model.predict(x[q])
        assert covered.all()
        results = {}
        for name, scope in scopes.items():
            compatible = scope & (ref == cur['parent'])
            base = cur['parent']
            a = compatible & predictions[:, 0]
            b = compatible & predictions[:, 1]
            results[name] = dict(eligible=int((compatible & contest).sum()),
                available=summary(contest, y != base, compatible, f),
                baseline=summary(a, y != base, scope, f),
                referee=summary(b, y != base, scope, f),
                incremental=summary(a != b, (base ^ a) != y, scope, f),
                exclusive=summary(b & ~a, y != base, scope, f))
        report['protocols'][protocol] = dict(train_range=[min(train_counts), max(train_counts)], scopes=results)
    (REPORT.parent / 'referee_new_family_challenges.json').write_text(json.dumps(report, indent=2)+'\n')
    for key, item in report['protocols'].items():
        print(key, [(name, scores['referee']['fixed'], scores['referee']['harmed'], scores['incremental']['net'])
                    for name, scores in item['scopes'].items()])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
