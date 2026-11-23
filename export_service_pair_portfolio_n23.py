"""Freeze the service-pair vote plus changes corroborated by two other model families."""
import json
import numpy as np
import pandas as pd

from broad_partition_evidence_n23 import ART, ROOT, write
from export_five_release_majority_n23 import load_votes
from kaggle_workflow import sha256, validate_submission
from service_pair_portfolio_n23 import NAMES, actions, test_actions

VERSION = 'n23_service_pair_corroborated'
NAME = 'proposal_' + VERSION
INCUMBENT = 'proposal_v131_party_consensus'
RULE = 'pair_plus_corroborated'


def forecast(recipe, ids):
    assert list(recipe['members']) == [INCUMBENT, *NAMES] and recipe['rule'] == RULE
    votes = load_votes(recipe, ids)
    ref = votes[:, 0]
    return ref ^ actions(votes[:, 1:] != ref[:, None])[RULE]


def export():
    ids = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['PassengerId']).PassengerId
    ref, expected = test_actions()
    reports = ROOT / 'reports'
    recipe = dict(rule=RULE, members={})
    for name in [INCUMBENT, *NAMES]:
        recipe['members'][name] = sha256(ART / name / 'submission.csv')
    receipt, = (reports / INCUMBENT / 'online_attempts').glob('*.json')
    assert recipe['members'][INCUMBENT] == json.loads(receipt.read_text())['file_sha256']
    out = ART / NAME
    out.mkdir(exist_ok=True)
    model = out / 'recipe.json'
    write(model, recipe)
    vote = forecast(recipe, ids)
    repeated = forecast(json.loads(model.read_text()), ids)
    np.testing.assert_array_equal(vote, repeated)
    np.testing.assert_array_equal(vote, ref ^ expected[RULE])
    pd.DataFrame(dict(PassengerId=ids, Transported=vote)).to_csv(out / 'submission.csv', index=False)
    # The ensemble outputs decisions; 0/1 below reproduces those decisions exactly.
    pd.DataFrame(dict(PassengerId=ids, probability=vote.astype(float),
                      reload_probability=repeated.astype(float))).to_csv(out / 'test_probabilities.csv', index=False)
    manifest = validate_submission(out / 'submission.csv', ROOT / 'data/raw/test.csv')
    receipts = [json.loads(p.read_text())['file_sha256'] for p in reports.glob('*/online_attempts/*.json')]
    assert manifest['sha256'] not in receipts and manifest['sha256'] not in recipe['members'].values()
    manifest.update(version=VERSION, model_sha256=sha256(model), reload_difference=0,
                    previous_receipts=len(receipts), v131_changes=int((vote != ref).sum()),
                    added_to_pair=int((expected[RULE] & ~expected['pair']).sum()))
    assert manifest['v131_changes'] == 174 and manifest['added_to_pair'] == 24
    for folder in (out, reports / NAME):
        write(folder / 'recipe.json', recipe)
        write(folder / 'manifest.json', manifest)
        write(folder / 'selection.json', dict(version=VERSION, threshold=.5))
    print(manifest)


if __name__ == '__main__':
    export()
