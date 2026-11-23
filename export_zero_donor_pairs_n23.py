"""Export verified pair-model decisions with limited false-to-true zero-bill changes."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_geographic_transfer_n23 import BILLS
from broad_partition_evidence_n23 import ART, ROOT, write
from broad_service_bag_trees_n23 import fit
from broad_zero_spend_analogues_n23 import simulate
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256, validate_submission
from zero_donor_pair_overlay_n23 import additional

VERSION = 'n23_service_pair_zero_donor'
NAME = 'proposal_' + VERSION
PARENT = 'proposal_n23_service_pair_change_150'


def export():
    train = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=['PassengerId','Transported'])
    test_path = ROOT / 'data/raw/test.csv'
    test = pd.read_csv(test_path, usecols=['PassengerId'])
    out = ART / NAME
    out.mkdir(exist_ok=True)
    raw, groups, size, _ = load_roster()
    raw['BookingSize'] = size
    ids = pd.concat([train.PassengerId,test.PassengerId], ignore_index=True)
    np.testing.assert_array_equal(groups, ids.str.split('_').str[0].astype(int))
    t = np.arange(len(train))
    q = np.arange(len(train), len(raw))
    assert not np.intersect1d(groups[t], groups[q]).size
    source_folder = ART / PARENT
    parent_manifest = json.loads((ROOT/'reports'/PARENT/'manifest.json').read_text())
    parent_digest = sha256(source_folder/'submission.csv')
    assert parent_manifest['sha256'] == parent_digest
    parent_hard = pd.read_csv(source_folder/'submission.csv')
    parent_soft = pd.read_csv(source_folder/'test_probabilities.csv',float_precision='round_trip')
    for parent in (parent_hard,parent_soft):
        np.testing.assert_array_equal(parent.PassengerId, test.PassengerId)
    p = parent_soft.probability.to_numpy()
    np.testing.assert_array_equal(p, parent_soft.reload_probability)
    np.testing.assert_array_equal(p>=.5,parent_hard.Transported.to_numpy())
    base, _, anchor = source('proposal_v108_party_bills',test)
    _, ref, incumbent = source('proposal_v131_party_consensus',test)
    assert parent_manifest['v108'] == anchor and parent_manifest['v131'] == incumbent
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    support = t[paid[t]]
    target = q[~paid[q] & ~ref & (ref == (base >= .5))]
    y = train.Transported.to_numpy()
    model = fit(raw.iloc[support], y[support], np.ones(len(support)))
    model_path = out / 'paid_bills.cbm'
    model.save_model(str(model_path))

    def decision(fitted):
        scores = np.full(len(q), np.nan)
        scores[target-len(t)] = simulate(raw, groups, y, t, target, paid, fitted)['matched']
        extra = additional(base, ref, p >= .5, scores, q, .10)
        # The selection above returns a binary action. 0.5 is the positive decision cutoff.
        return np.where(extra, .5, p), extra, scores

    probability, extra, scores = decision(model)
    replay, _, reloaded_scores = decision(CatBoostClassifier().load_model(str(model_path)))
    np.testing.assert_array_equal(scores,reloaded_scores)
    np.testing.assert_array_equal(probability,replay)
    np.testing.assert_array_equal((probability>=.5) != (p>=.5), extra)
    assert np.isfinite(probability).all() and extra.any()
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=probability,
                      reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=probability>=.5)).to_csv(out/'submission.csv',index=False)
    check = validate_submission(out/'submission.csv',test_path)
    receipts = list((ROOT/'reports').glob('*/online_attempts/*.json'))
    assert not any(json.loads(path.read_text()).get('file_sha256') == check['sha256'] for path in receipts)
    result = dict(**check, version=VERSION, model_sha256=sha256(model_path), reload_difference=0,
                  parents={PARENT:parent_digest, 'v108':anchor, 'v131':incumbent},
                  model_training_rows=len(support), eligible_test=len(target), test_budget=len(target)//10,
                  additional_changes=int(extra.sum()), v131_changes=int(((probability>=.5)!=ref).sum()),
                  parent_probability_sha256=sha256(source_folder/'test_probabilities.csv'), receipts_checked=len(receipts))
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',result)
        write(folder/'selection.json',dict(version=VERSION, threshold=.5, direction='false_to_true', fraction=.10))
    print(result)


if __name__ == '__main__':
    with threadpool_limits(2):
        export()
