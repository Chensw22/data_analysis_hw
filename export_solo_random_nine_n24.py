"""Compose three frozen passenger predictions and verify the release against official receipts."""
import json

import numpy as np
import pandas as pd

from export_budget_quarter_n23 import ART, ROOT, source
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

VERSION = 'n24_solo_random_nine'
NAME = 'proposal_' + VERSION
PARENTS = dict(champion='proposal_v131_party_consensus', nine='proposal_n24_selected_dissent',
               random_fields='proposal_n24_random_subspace')
RULE = 'Prefer a nine-vote change, else use random fields for single-person bookings, else use v131.'


def freeze(test):
    parents = {}
    for role, name in PARENTS.items():
        path = ART/name
        if role == 'nine':
            manifest = json.loads((ROOT/'reports'/name/'manifest.json').read_text())
            digest = manifest['sha256']
            assert sha256(path/'ensemble.joblib') == manifest['model_sha256']
        else:
            _, _, digest = source(name, test)
        parents[role] = dict(name=name, submission=digest, scores=sha256(path/'test_probabilities.csv'))
    return dict(version=VERSION, rule=RULE, parents=parents)


def compose(recipe, test):
    assert recipe['version'] == VERSION and recipe['rule'] == RULE
    scores = {}
    for role, entry in recipe['parents'].items():
        path = ART/entry['name']
        assert entry['name'] == PARENTS[role] and sha256(path/'submission.csv') == entry['submission']
        assert sha256(path/'test_probabilities.csv') == entry['scores']
        frame = pd.read_csv(path/'test_probabilities.csv', float_precision='round_trip')
        hard = pd.read_csv(path/'submission.csv')
        for ids in (frame.PassengerId, hard.PassengerId):
            np.testing.assert_array_equal(ids, test.PassengerId)
        np.testing.assert_array_equal(frame.probability, frame.reload_probability)
        assert frame.probability.between(0, 1).all()
        np.testing.assert_array_equal(frame.probability >= .5, hard.Transported)
        scores[role] = frame.probability.to_numpy()
    champion = scores['champion'] >= .5
    nine_change = (scores['nine'] >= .5) != champion
    group = test.PassengerId.str.split('_').str[0]
    solo = ~group.duplicated(keep=False).to_numpy()
    random_change = solo & ((scores['random_fields'] >= .5) != champion)
    pred = np.where(nine_change, scores['nine'], np.where(solo, scores['random_fields'], scores['champion']))
    return pred, champion, nine_change, random_change


def export():
    test_file = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_file, usecols=['PassengerId'])
    out = ART/NAME
    recipe = freeze(test)
    model_file = out/'composition.json'
    write(model_file, recipe)
    pred, champion, nine, random = compose(recipe, test)
    reload_pred = compose(json.loads(model_file.read_text()), test)[0]
    np.testing.assert_array_equal(pred, reload_pred)
    hard = pred >= .5
    changed = hard != champion
    assert (nine.sum(), random.sum(), (nine & random).sum(), changed.sum()) == (10, 60, 0, 70)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=pred, reload_probability=reload_pred)
                 ).to_csv(out/'test_probabilities.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=hard)).to_csv(out/'submission.csv', index=False)
    checked = validate_submission(out/'submission.csv', test_file)
    distances = {}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        name = receipt.parents[1].name
        if name == NAME or record.get('evaluation', {}).get('Status') != 'COMPLETE':
            continue
        path = ART/name/'submission.csv'
        assert sha256(path) == record['file_sha256']
        previous = pd.read_csv(path)
        np.testing.assert_array_equal(previous.PassengerId, test.PassengerId)
        distances[name] = int((hard != previous.Transported).sum())
    nearest = min(distances.values())
    assert nearest > 0
    manifest = dict(**checked, version=VERSION, parents={k:v['submission'] for k,v in recipe['parents'].items()},
                    model_sha256=sha256(model_file), reload_difference=0, training_rows=8693, additional_fit_rows=0,
                    v131_changes=int(changed.sum()), to_true=int((changed & hard).sum()),
                    to_false=int((changed & ~hard).sum()), overlap=int((nine & random).sum()),
                    nine_changes=int(nine.sum()), solo_random_changes=int(random.sum()),
                    closest_submitted_changes=nearest, completed_exports=len(distances),
                    nearest_submissions=[n for n in sorted(distances) if distances[n] == nearest])
    selection = dict(**recipe, threshold=.5, evidence='reports/nov23_matched_learning_curve/solo_release_replay.json')
    for directory in (out, ROOT/'reports'/NAME):
        write(directory/'manifest.json', manifest)
        write(directory/'selection.json', selection)
    print(manifest)


if __name__ == '__main__':
    export()
