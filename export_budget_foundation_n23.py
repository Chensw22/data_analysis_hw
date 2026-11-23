"""Export the pretrained budget view on top of the actual v131 release."""
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from broad_partition_evidence_n23 import ART, ROOT, write
from broad_pretrained_spending_views_n23 import frames, predict
from export_budget_quarter_n23 import source
from exaone_validation_data import CATEGORICAL
from kaggle_workflow import sha256, validate_submission
from tabicl_routed_context_n23 import DIGEST, WEIGHT

VERSION = 'n23_incumbent_quarter_tabicl_budget'
NAME = 'proposal_'+VERSION


def run():
    assert sha256(WEIGHT) == DIGEST
    train = pd.read_csv(ROOT/'data/raw/train.csv')
    test = pd.read_csv(ROOT/'data/raw/test.csv')
    assert not set(train.PassengerId.str.split('_').str[0]) & set(test.PassengerId.str.split('_').str[0])
    frame = frames(pd.concat([train.drop(columns='Transported'), test], ignore_index=True))[0]
    out = ART/NAME
    out.mkdir(exist_ok=True)
    labels = train.Transported.to_numpy()
    t = np.arange(len(train))
    q = np.arange(len(train), len(frame))
    # TabICL has a frozen checkpoint. Its fitted context is the ordered training rows and their outcomes.
    context_file = out/'training_context.csv'
    frame.iloc[t].assign(Transported=labels).to_csv(context_file, index=False)
    context = pd.read_csv(context_file, dtype={k: str for k in CATEGORICAL}, float_precision="round_trip")
    replay_labels = context.pop('Transported').to_numpy()
    np.testing.assert_array_equal(replay_labels, labels)
    pd.testing.assert_frame_equal(frame.iloc[t], context)
    second = pd.concat([context, frame.iloc[q]], ignore_index=True)
    base, _, base_sha = source('proposal_v108_party_bills', test)
    _, ref, ref_sha = source('proposal_v131_party_consensus', test)
    def calculate(x, y):
        p = predict(x, y, t, q, 0)
        return np.where((base >= .5) == ref, .75*base + .25*p, ref.astype(float))
    probs, replay = calculate(frame, labels), calculate(second, replay_labels)
    np.testing.assert_array_equal(probs, replay)
    csv = out/'submission.csv'
    vote = probs >= .5
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=vote)).to_csv(csv, index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=probs, reload_probability=replay)).to_csv(
        out/'test_probabilities.csv', index=False)
    differences = {}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        if record.get('evaluation', {}).get('Status') != 'COMPLETE':
            continue
        archive = ART/receipt.parents[1].name/'submission.csv'
        if not archive.is_file() or sha256(archive) != record['file_sha256']:
            continue
        old = pd.read_csv(archive)
        np.testing.assert_array_equal(old.PassengerId, test.PassengerId)
        differences[receipt.parents[1].name] = int((old.Transported != vote).sum())
    assert min(differences.values()) > 0
    check = validate_submission(csv, ROOT/'data/raw/test.csv')
    manifest = dict(**check, version=VERSION, model_sha256=sha256(context_file), checkpoint_sha256=DIGEST,
                    reload_difference=0, seed=73, estimators=2, fit_people=len(train),
                    v108_hash=base_sha, v131_hash=ref_sha, retained_overrides=int(((base >= .5) != ref).sum()),
                    differences=differences)
    for folder in [out, ROOT/'reports'/NAME]:
        write(folder/'manifest.json', manifest)
        write(folder/'selection.json', dict(version=VERSION, sha256=check['sha256']))
    print(check, len(differences), min(differences.values()), differences['proposal_v131_party_consensus'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
