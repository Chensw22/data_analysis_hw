"""Export the fixed random 52-field three-tree ensemble against the primary v131 release."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from export_budget_quarter_n23 import source
from full_roster_permutation_selection_n24 import (
    OUT, ROOT, ensemble_probability, feature_bank, fit_ensemble,
)
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

VERSION = 'n24_random_subspace'
NAME = 'proposal_' + VERSION


def export():
    train_path = ROOT / 'data/raw/train.csv'
    test_path = ROOT / 'data/raw/test.csv'
    train, test = [pd.read_csv(path, usecols=lambda col: col != 'Name') for path in (train_path, test_path)]
    train_groups, test_groups = [frame.PassengerId.str.split('_').str[0] for frame in (train, test)]
    assert not set(train_groups) & set(test_groups)
    x, fields, _ = feature_bank(train)
    test_x, test_fields, _ = feature_bank(test)
    cols = np.random.default_rng(1216).permutation(len(fields))[:52]
    chosen = [fields[col] for col in cols]
    query = pd.DataFrame(test_x, columns=test_fields).reindex(columns=chosen, fill_value=0).to_numpy()
    x = x[:, cols]
    y = train.Transported.to_numpy(int)
    with np.load(OUT / 'full_0.npz') as fold:
        np.testing.assert_array_equal(fold['cols'][1], cols)
        np.testing.assert_array_equal(fold['features'], fields)
        training, validation = fold['train'], fold['query']
        model = fit_ensemble(x[training], y[training], 497)
        np.testing.assert_array_equal(ensemble_probability(model, x[validation]), fold['prob'][:, 2])

    out = ROOT / 'artifacts' / NAME
    out.mkdir(exist_ok=True)
    parent = {}
    base, _, parent['v96'] = source('proposal_online_first_counts15', test)
    incumbent, champion, parent['v131'] = source('proposal_v131_party_consensus', test)
    locked = (base >= .5) != champion
    assert locked.sum() == 61

    def predict(models):
        return np.where(locked, incumbent, .75 * base + .25 * ensemble_probability(models, query))

    models = fit_ensemble(x, y, 497)
    file = out / 'ensemble.joblib'
    joblib.dump(dict(fields=chosen, models=models), file)
    p = predict(models)
    restored = joblib.load(file)
    assert restored['fields'] == chosen
    repeat = predict(restored['models'])
    np.testing.assert_array_equal(p, repeat)
    hard = p >= .5
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(hard[locked], champion[locked])
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=hard)).to_csv(out/'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=p, reload_probability=repeat)).to_csv(
        out/'test_probabilities.csv', index=False)
    check = validate_submission(out/'submission.csv', test_path)
    receipts = list((ROOT/'reports').glob('*/online_attempts/*.json'))
    assert all(json.loads(path.read_text()).get('file_sha256') != check['sha256'] for path in receipts)
    change = hard != champion
    assert change.any()
    manifest = dict(**check, version=VERSION, model_sha256=sha256(file), reload_difference=0,
                    parents=parent, training_rows=len(train), locked_v131=int(locked.sum()),
                    v131_changes=int(change.sum()), changed_to_true=int((change & hard).sum()),
                    changed_to_false=int((change & ~hard).sum()), receipts_checked=len(receipts),
                    source_train_sha256=sha256(train_path), source_test_sha256=sha256(test_path),
                    test_zero_filled_fields=[k for k in chosen if k not in test_fields])
    selection = dict(version=VERSION, fields=chosen, draw_seed=1216, model_seed=497, challenger_weight=.25,
                     evidence='reports/nov23_matched_learning_curve/permutation_selection.json',
                     mode='random/quarter')
    for directory in [out, ROOT/'reports'/NAME]:
        write(directory/'manifest.json', manifest)
        write(directory/'selection.json', selection)
    print(manifest)


if __name__ == '__main__':
    with threadpool_limits(3):
        export()
