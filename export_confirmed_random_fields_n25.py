"""Export random-field edits endorsed by two complementary feature partitions."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import OUT as LOCAL_FOLDS
from export_budget_quarter_n23 import ART, ROOT, source
from full_roster_permutation_selection_n24 import ensemble_probability, feature_bank, fit_ensemble
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

NAME = 'proposal_n25_confirmed_random_fields'
RANDOM = 'proposal_n24_random_subspace'


def export():
    train_path, test_path = (ROOT/'data/raw'/f'{split}.csv' for split in ('train', 'test'))
    train, test = (pd.read_csv(path, usecols=lambda col: col != 'Name') for path in (train_path, test_path))
    x, fields, _ = feature_bank(train)
    xt, test_fields, _ = feature_bank(test)
    query = pd.DataFrame(xt, columns=test_fields).reindex(columns=fields, fill_value=0).to_numpy()
    y = train.Transported.to_numpy(int)
    perm = np.random.default_rng(1216).permutation(len(fields))
    chosen, rest = perm[:52], perm[52:]
    model_path = ART/RANDOM/'ensemble.joblib'
    random_manifest = json.loads((ROOT/'reports'/RANDOM/'manifest.json').read_text())
    assert sha256(model_path) == random_manifest['model_sha256']
    original = joblib.load(model_path)
    assert original['fields'] == [fields[i] for i in chosen]
    parents = {}
    native, native_hard, parents['native'] = source('proposal_online_first_counts15', test)
    _, v131, parents['v131'] = source('proposal_v131_party_consensus', test)
    nine_p, nine, parents['nine'] = source('proposal_n24_selected_dissent', test)
    _, prior, parents['random'] = source(RANDOM, test)
    mixed = .75*native + .25*ensemble_probability(original['models'], query[:, chosen])
    original_hard = mixed >= .5
    np.testing.assert_array_equal(np.where(native_hard == v131, original_hard, v131), prior)
    free = (native_hard == v131) & (v131 == nine)
    out = ART/NAME
    out.mkdir(exist_ok=True)
    models = []
    for layout, order in enumerate((rest, np.random.default_rng(11939).permutation(rest))):
        suffix = '_repeat' if layout else ''
        with np.load(LOCAL_FOLDS/f'full_0{suffix}.npz') as fold:
            np.testing.assert_array_equal(fold['features'], fields)
            np.testing.assert_array_equal(fold['permutation'], np.r_[chosen, order])
            ensemble = []
            for j, cols in enumerate(np.array_split(order, 3)):
                trained = fit_ensemble(x[fold['t']][:, cols], y[fold['t']], 497)
                np.testing.assert_array_equal(ensemble_probability(trained, x[fold['q']][:, cols]), fold['p'][:, j])
                ensemble.append((cols, fit_ensemble(x[:, cols], y, 497)))
            models.append(ensemble)

    def predict(layouts):
        endorsed = free.copy()
        for layout in layouts:
            votes = np.column_stack([ensemble_probability(model, query[:, cols]) >= .5 for cols, model in layout])
            endorsed &= (votes == original_hard[:, None]).sum(1) >= 2
        return np.where(endorsed, mixed, nine_p)

    file = out/'ensemble.joblib'
    joblib.dump(dict(fields=fields, layouts=models), file)
    p = predict(models)
    restored = joblib.load(file)
    assert restored['fields'] == fields
    repeat = predict(restored['layouts'])
    np.testing.assert_array_equal(p, repeat)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    hard = p >= .5
    changes = hard != nine
    assert not (changes & ((prior == v131) | ~free)).any()
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=hard)).to_csv(out/'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=p, reload_probability=repeat)).to_csv(
        out/'test_probabilities.csv', index=False)
    check = validate_submission(out/'submission.csv', test_path)
    distances = {}
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('evaluation', {}).get('Status') != 'COMPLETE':
            continue
        name = path.parents[1].name
        submitted = ART/name/'submission.csv'
        assert sha256(submitted) == receipt['file_sha256']
        frame = pd.read_csv(submitted)
        np.testing.assert_array_equal(frame.PassengerId, test.PassengerId)
        distances[name] = int((frame.Transported != hard).sum())
    nearest = min(distances.values())
    manifest = dict(**check, version=NAME.removeprefix('proposal_'), parents=parents,
                    model_sha256=sha256(file), original_model_sha256=sha256(model_path),
                    source_train_sha256=sha256(train_path), source_test_sha256=sha256(test_path),
                    training_rows=len(y), reload_difference=0, nine_changes=int(changes.sum()),
                    changed_to_true=int((changes & hard).sum()), changed_to_false=int((changes & ~hard).sum()),
                    closest_submitted_changes=nearest, completed_exports=len(distances),
                    nearest_submissions=sorted(name for name, d in distances.items() if d == nearest))
    selection = dict(version=manifest['version'], mode='both_partitions_confirmed', original_fields=52, original_draw_seed=1216,
                     complementary_permutation_seed=11939, model_seed=497, challenger_weight=.25,
                     evidence='reports/nov23_matched_learning_curve/disjoint_tree_fields.json')
    for directory in (out, ROOT/'reports'/NAME):
        write(directory/'manifest.json', manifest)
        write(directory/'selection.json', selection)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    with threadpool_limits(3):
        export()
