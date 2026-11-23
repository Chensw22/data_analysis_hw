"""Export queued corrections plus additional changes supported by two model families."""
import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from export_five_release_majority_n23 import load_votes
from kaggle_workflow import sha256, validate_submission
from pending_family_coverage_n23 import ART, DONOR, REPORT, ROOT, read, rules, write

VERSION = 'n23_pending_family_support'
NAME = 'proposal_'+VERSION
BASELINE = 'proposal_v131_party_consensus'


def predict(recipe, ids):
    members = recipe['members']
    matrix = load_votes(recipe, ids)
    votes = dict(zip(members, matrix.T))
    ref = votes.pop(BASELINE)
    masks, _ = rules(ref, votes)
    return ref ^ masks[recipe['rule']], ref, votes


def export():
    coverage = read(REPORT/'coverage.json')
    test_path = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_path, usecols=['PassengerId'])
    _, official, digest = source(BASELINE, test)
    recipe = dict(rule='queued_or_two', members={BASELINE: digest, **coverage['member_hashes']})
    out = ART/NAME
    out.mkdir(exist_ok=True)
    vote, ref, members = predict(recipe, test.PassengerId)
    np.testing.assert_array_equal(ref, official)
    model = out/'recipe.json'
    write(model, recipe)
    reload, _, _ = predict(read(model), test.PassengerId)
    np.testing.assert_array_equal(reload, vote)
    assert np.all(vote[members[DONOR] != ref] == members[DONOR][members[DONOR] != ref])
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=vote)).to_csv(out/'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=vote.astype(float),
                      reload_probability=reload.astype(float))).to_csv(out/'test_probabilities.csv', index=False)
    manifest = validate_submission(out/'submission.csv', test_path)
    receipts = [read(p)['file_sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json')]
    distances = {k: int((vote != v).sum()) for k, v in members.items()}
    assert manifest['sha256'] not in receipts and min(distances.values()) > 0
    total = int((vote != ref).sum())
    assert total == coverage['test_changes'][recipe['rule']]
    manifest.update(version=VERSION, model_sha256=sha256(model), v131_changes=total, reload_difference=0,
                    previous_receipts=len(receipts), candidate_distances=distances)
    for path in [out, ROOT/'reports'/NAME]:
        write(path/'manifest.json', manifest)
        write(path/'selection.json', dict(version=VERSION, threshold=.5))
        write(path/'recipe.json', recipe)
    print(manifest)


if __name__ == '__main__':
    export()
