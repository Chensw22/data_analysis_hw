"""Freeze the broad, paired change induced by exact fare information in TabICL."""
import numpy as np
import pandas as pd

from export_consensus_review_n26 import ART, dig, write
from full_roster_booking_validation_n23 import ROOT, data
from kaggle_workflow import validate_submission
from price_implied_deck_n26 import OUT as STUDY, build_features


NAME = 'proposal_n26_fare_deck'
PARENT = 'proposal_n26_consensus_review'


def combine(parent, raw, fine, eligible):
    alter = eligible & ((raw >= .5) == (parent >= .5)) & ((fine >= .5) != (parent >= .5))
    return np.where(alter, fine, parent)


def run():
    import json
    import torch
    from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer
    assert dig(WEIGHT) == DIGEST
    torch.set_num_threads(2)
    raw, y, groups, folds = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns.tolist())
    all_rows = pd.concat([raw, test], ignore_index=True)
    all_groups = all_rows.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(all_groups[len(y):], all_groups[:len(y)]).any()
    x = build_features(all_rows, all_groups)
    query = np.arange(len(y), len(all_rows))
    train = np.arange(len(y))
    out = ART/NAME
    out.mkdir(exist_ok=True)
    with np.load(STUDY/'neural_full_0.npz') as z:
        errors = {}
        scores = {}
        for n in ('raw', 'fine'):
            replay = infer(x[n], y, z['t'], z['q'], 5583)
            errors[n] = float(np.max(abs(replay - z[n])))
            np.testing.assert_allclose(replay, z[n], atol=3e-5, rtol=0)
            np.testing.assert_array_equal(replay >= .5, z[n] >= .5)
            scores[n] = infer(x[n], y, train, query, 5583)
    parent_dir = ART/PARENT
    parent_file = parent_dir/'submission.csv'
    parent_digest = dig(parent_file)
    assert parent_digest == json.loads((ROOT/'reports'/PARENT/'manifest.json').read_text())['sha256']
    parent_frame = pd.read_csv(parent_dir/'test_probabilities.csv', float_precision='round_trip')
    original = pd.read_csv(parent_file)
    np.testing.assert_array_equal(parent_frame.PassengerId, test.PassengerId)
    np.testing.assert_array_equal(original.PassengerId, test.PassengerId)
    parent = parent_frame.probability.to_numpy()
    np.testing.assert_array_equal(parent >= .5, original.Transported)
    nine = pd.read_csv(ART/'proposal_n24_selected_dissent/submission.csv')
    np.testing.assert_array_equal(nine.PassengerId, test.PassengerId)
    with np.load(STUDY/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'][query], test.PassengerId)
        eligible = z['eligible'][query]
    model_path = out/'scores.npz'
    np.savez_compressed(model_path, passenger=test.PassengerId.to_numpy(str), parent=parent,
                        eligible=eligible, **scores)
    p = combine(parent, scores['raw'], scores['fine'], eligible)
    with np.load(model_path) as z:
        repeat = combine(z['parent'], z['raw'], z['fine'], z['eligible'])
    np.testing.assert_array_equal(p, repeat)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=p>=.5)).to_csv(out/'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=p, reload_probability=repeat)).to_csv(out/'test_probabilities.csv', index=False)
    checked = validate_submission(out/'submission.csv', ROOT/'data/raw/test.csv')
    assert all(json.loads(f.read_text())['file_sha256'] != checked['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    changed = (p >= .5) != (parent >= .5)
    with np.load(parent_dir/'test_inputs.npz') as z:
        np.testing.assert_array_equal(z['passenger'], test.PassengerId)
        consensus = z['agreement'] & ((z['parent_probability'] >= .5) == nine.Transported.to_numpy())
    manifest = dict(**checked, version=NAME.removeprefix('proposal_'), training_rows=len(y),
                    parents={PARENT:parent_digest}, model_sha256=dig(model_path), checkpoint_sha256=DIGEST,
                    reload_difference=0, replay_max_error=errors, changes=int(changed.sum()),
                    unanimous_rows=int(consensus.sum()), unanimous_changes=int((changed & consensus).sum()),
                    eligible=int(eligible.sum()))
    selection = dict(version=manifest['version'], rule='For a fully recorded positive bill, replace the parent decision when raw TabICL agrees with it and the fare-deck TabICL reverses it.',
                     evidence='reports/nov23_matched_learning_curve/price_implied_deck.json')
    for path in (out, ROOT/'reports'/NAME):
        write(path/'manifest.json', manifest)
        write(path/'selection.json', selection)
    print(manifest, flush=True)


if __name__ == '__main__':
    run()
