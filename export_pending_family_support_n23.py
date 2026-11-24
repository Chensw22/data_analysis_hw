"""Export queued corrections plus additional changes supported by two model families."""
import argparse

import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from export_five_release_majority_n23 import load_votes
from kaggle_workflow import sha256, validate_submission
from pending_family_coverage_n23 import ART, DONOR, REPORT, ROOT, read, rules, write

VERSIONS = {'queued_or_two': 'n23_pending_family_support', 'two_families': 'n24_two_family_support'}
BASELINE = 'proposal_v131_party_consensus'


def predict(recipe, ids):
    members = recipe['members']
    matrix = load_votes(recipe, ids)
    votes = dict(zip(members, matrix.T))
    ref = votes.pop(BASELINE)
    masks, _ = rules(ref, votes)
    return ref ^ masks[recipe['rule']], ref, votes


def export(rule='queued_or_two'):
    version = VERSIONS[rule]
    name = 'proposal_' + version
    coverage = read(REPORT/'coverage.json')
    test_path = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_path, usecols=['PassengerId'])
    _, official, digest = source(BASELINE, test)
    recipe = dict(rule=rule, members={BASELINE: digest, **coverage['member_hashes']})
    out = ART/name
    out.mkdir(exist_ok=True)
    vote, ref, members = predict(recipe, test.PassengerId)
    np.testing.assert_array_equal(ref, official)
    model = out/'recipe.json'
    write(model, recipe)
    reload, _, _ = predict(read(model), test.PassengerId)
    np.testing.assert_array_equal(reload, vote)
    if rule == 'queued_or_two':
        assert np.all(vote[members[DONOR] != ref] == members[DONOR][members[DONOR] != ref])
    else:
        _, count = rules(ref, members)
        np.testing.assert_array_equal(vote != ref, count >= 2)
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=vote)).to_csv(out/'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=vote.astype(float),
                      reload_probability=reload.astype(float))).to_csv(out/'test_probabilities.csv', index=False)
    manifest = validate_submission(out/'submission.csv', test_path)
    receipts = [read(p)['file_sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json')]
    distances = {k: int((vote != v).sum()) for k, v in members.items()}
    assert manifest['sha256'] not in receipts and min(distances.values()) > 0
    total = int((vote != ref).sum())
    assert total == coverage['test_changes'][recipe['rule']]
    manifest.update(version=version, model_sha256=sha256(model), v131_changes=total, reload_difference=0,
                    previous_receipts=len(receipts), candidate_distances=distances)
    for path in [out, ROOT/'reports'/name]:
        write(path/'manifest.json', manifest)
        write(path/'selection.json', dict(version=version, threshold=.5))
        write(path/'recipe.json', recipe)
    print(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule', choices=VERSIONS, default='queued_or_two')
    export(parser.parse_args().rule)
