"""Export the full-population personal classifier blended into the incumbent replay."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_partition_evidence_n23 import ART, ROOT, write
from broad_two_source_manifest_n23 import fit_model
from broad_within_booking_bill_matching_n23 import encode
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256, validate_submission

VERSION = 'n23_personal_bill_quarter'
NAME = 'proposal_' + VERSION


def export():
    train = pd.read_csv(ROOT/'data/raw/train.csv', usecols=['PassengerId', 'Transported'])
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId'])
    frame, groups, size, _ = load_roster()
    people = pd.concat([train.PassengerId, test.PassengerId], ignore_index=True)
    np.testing.assert_array_equal(groups, people.str.split('_').str[0].astype(int))
    person, bills = encode(frame)
    features = np.column_stack([person, bills, np.log(size)])
    base, _, anchor_hash = source('proposal_v108_party_bills', test)
    _, incumbent, incumbent_hash = source('proposal_v131_party_consensus', test)
    locked = incumbent != (base >= .5)
    directory = ART / NAME
    directory.mkdir(exist_ok=True)
    model = fit_model(features[:len(train)], train.Transported.to_numpy(int))
    path = directory / 'personal.cbm'
    model.save_model(str(path))

    def predict(fitted):
        personal = fitted.predict_proba(features[len(train):], thread_count=2)[:, 1]
        return np.where(locked, incumbent.astype(float), .75 * base + .25 * personal)

    probability = predict(model)
    reload = predict(CatBoostClassifier().load_model(str(path)))
    np.testing.assert_array_equal(probability, reload)
    assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
    vote = probability >= .5
    np.testing.assert_array_equal(vote[locked], incumbent[locked])
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=probability,
                      reload_probability=reload)).to_csv(directory/'test_probabilities.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=vote)).to_csv(directory/'submission.csv', index=False)
    validation = validate_submission(directory/'submission.csv', ROOT/'data/raw/test.csv')
    differences = {}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        assert record.get('file_sha256') != validation['sha256'], receipt
        previous = ART / receipt.parent.parent.name / 'submission.csv'
        if previous.exists() and sha256(previous) == record.get('file_sha256'):
            other = pd.read_csv(previous)
            np.testing.assert_array_equal(other.PassengerId, test.PassengerId)
            differences[receipt.parent.parent.name] = int((vote != other.Transported).sum())
    nearest = min(differences.values())
    assert nearest > 0
    assert differences['proposal_v131_party_consensus'] == (vote != incumbent).sum()
    report = dict(**validation, version=VERSION, training_rows=len(train), anchor_v108=anchor_hash,
                  incumbent_v131=incumbent_hash, locked=int(locked.sum()), changed=int((vote != incumbent).sum()),
                  model_sha256=sha256(path), reload_difference=0, previous_files=len(differences),
                  closest_changes=nearest, closest=[k for k,v in differences.items() if v==nearest])
    for parent in [directory, ROOT/'reports'/NAME]:
        write(parent/'manifest.json', report)
        write(parent/'selection.json', dict(version=VERSION, threshold=.5))
    print(report)


if __name__ == '__main__':
    with threadpool_limits(2):
        export()
