"""Export the quarter-weight registration and bill tree on the complete labeled roster."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_partition_evidence_n23 import ART, ROOT
from export_service_bag_n23 import publish_blend
from registration_bill_links_n23 import fields, fitted_tree, level_map


def predict(bundle, frame):
    register, bills, _ = fields(frame, bundle['levels'])
    return expit(bundle['tree'].decision_function(np.column_stack([register, bills])))


def export():
    version = 'n23_registration_tree_quarter'
    train = pd.read_csv(ROOT/'data/raw/train.csv', usecols=['PassengerId', 'Transported'])
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId'])
    frame, groups, size, audit = load_roster()
    assert audit['test_group_overlap'] == 0
    ids = pd.concat([train.PassengerId, test.PassengerId], ignore_index=True)
    np.testing.assert_array_equal(groups, ids.str.split('_').str[0].astype(int))
    frame['BookingSize'] = size
    levels = level_map(frame)
    register, bills, categories = fields(frame.iloc[:len(train)], levels)
    tree = fitted_tree(np.column_stack([register, bills]), train.Transported,
                       np.r_[categories, np.zeros(bills.shape[1], bool)])
    bundle = dict(tree=tree, levels=levels)
    path = ART/('proposal_'+version)/'registration_tree.joblib'
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    query = frame.iloc[len(train):]
    alt = predict(bundle, query)
    replay = predict(joblib.load(path), query)
    earlier = json.loads((ROOT/'reports/nov23_recent_family_transfer/pending_diversity.json').read_text())
    comparators = [*earlier['files'], 'proposal_n23_large_booking_union', 'proposal_n23_all_booking_union']
    publish_blend(version, path, test, alt, replay, len(train), comparators, 'pending_union_changes')


if __name__ == '__main__':
    with threadpool_limits(2):
        export()
