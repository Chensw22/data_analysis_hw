"""Test whether solo-only models can recognize errors when denied different inputs."""
import json

import numpy as np
from catboost import CatBoostClassifier

from broad_geographic_transfer_n23 import BILLS
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, SEED, data


def run():
    raw, y, groups, fold = data()
    _, order, size = np.unique(groups, return_inverse=True, return_counts=True)
    solo = size[order] == 1
    x = compact_features(raw)
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as bank:
        dev = np.sort(bank['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    cols = []
    for family in (*FAMILIES, 'mitra96'):
        with np.load(OUT / f'{family}.npz') as bank:
            np.testing.assert_array_equal(bank['passenger'], raw.PassengerId)
            np.testing.assert_array_equal(bank['fold'], fold)
            np.testing.assert_array_equal(bank['y'], y)
            cols.append(bank['p'])
    scores = np.column_stack(cols)
    common = ((scores >= .5) != y[:, None]).all(1)
    p = scores[:, -1]
    baseline = p >= .5
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    profiles = {}
    for bill, payment in [(paid, 'positive'), (~paid, 'no_known_positive')]:
        for decision in [False, True]:
            votes = ((scores >= .5) == decision).all(1)
            selected = solo & bill & votes
            profiles[f'{payment}_unanimous_{decision}'] = dict(rows=int(selected.sum()),
                errors=int((selected & common).sum()), historical_errors=int((selected & common & hard).sum()))
    numerically_confident = (np.abs(scores-.5) >= .25).all(1)
    result = dict(solo=int(solo.sum()), profiles=profiles, solo_shared=int((common & solo).sum()),
                  confident_solo_shared=int((common & solo & numerically_confident).sum()))
    margins = np.abs(scores-.5).min(axis=1)
    result['confidence_by_direction'] = {}
    for cutoff in [.10, .25, .40]:
        by_direction = {}
        for decision in [False, True]:
            selected = solo & ((scores >= .5) == decision).all(1) & (margins >= cutoff)
            by_direction[str(decision)] = dict(correct=int((selected & ~common).sum()),
                                                wrong=int((selected & common).sum()))
        result['confidence_by_direction'][str(cutoff)] = by_direction
    views = dict(full=list(x), no_bills=[k for k in x if k not in BILLS],
                 no_context=['Age', 'CryoSleep', 'VIP', *BILLS])
    predictions = {}
    for name, fields in views.items():
        pred = p.copy()
        for f in np.unique(fold):
            train = solo & (fold != f)
            query = solo & (fold == f)
            model = CatBoostClassifier(iterations=420, depth=5, learning_rate=.045,
                        cat_features=[k for k in fields if k in CATEGORICAL],
                        allow_writing_files=False, verbose=False, random_seed=SEED, thread_count=4)
            model.fit(x.loc[train, fields], y[train].astype(int))
            pred[query] = model.predict_proba(x.loc[query, fields])[:, 1]
        predictions[name] = pred
    left, right = predictions['no_bills'], predictions['no_context']
    agree = (left >= .5) == (right >= .5)
    policies = dict(full_solo=predictions['full'] >= .5,
                    average_baseline_full=(p+predictions['full']) >= 1,
                    two_views_agree=np.where(agree, left >= .5, baseline),
                    average_three=(p+left+right) >= 1.5)
    result['policies'] = {}
    for name, decision in policies.items():
        assert np.array_equal(decision[~solo], baseline[~solo])
        rescue = (baseline != y) & (decision == y)
        harm = (baseline == y) & (decision != y)
        result['policies'][name] = dict(repaired=int(rescue.sum()), newly_wrong=int(harm.sum()),
            accuracy=float((decision == y).mean()), shared_repaired=int((rescue & common).sum()),
            historical_repaired=int((rescue & hard).sum()), historical_harmed=int((harm & hard).sum()),
            fold_net=[int(rescue[fold == f].sum()-harm[fold == f].sum()) for f in np.unique(fold)])
    path = ROOT / 'reports/nov23_matched_learning_curve/full_roster_solo_views.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(path.read_text())


if __name__ == '__main__':
    run()
