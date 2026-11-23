"""Export the two locally positive coverage levels from the same verified full model."""
import json
from shutil import copyfile

import numpy as np
import pandas as pd

from broad_partition_evidence_n23 import ART, ROOT
from broad_service_pairs_n23 import REPORT
from export_service_bag_n23 import NAME as LINEAR, PERSONAL, THREE, publish_blend
from export_service_bag_tree_n23 import NAME as TREE
from kaggle_workflow import sha256
from service_pair_change_budget_n23 import changed


def export():
    folder = ART / 'nov23_service_pair_full'
    results = json.loads((REPORT / 'test_budget.json').read_text())
    training = results['training']
    model = folder / 'pairs.cbm'
    assert sha256(model) == training['model_sha256']
    test = pd.read_csv(ROOT / 'data/raw/test.csv', usecols=['PassengerId'])
    with np.load(folder / 'probabilities.npz') as cached:
        np.testing.assert_array_equal(cached['id'], test.PassengerId)
        for key in ('change_150', 'quarter'):
            version = 'n23_service_pair_' + key
            out = ART / ('proposal_' + version)
            out.mkdir(exist_ok=True)
            path = out / model.name
            copyfile(model, path)
            assert sha256(path) == training['model_sha256']
            plan = results['test'][key]
            weight = plan['weight']
            expect = changed(cached['pair'], cached['base'], cached['ref'], weight)
            publish_blend(version, path, test, cached['pair'], cached['reload'],
                          training['training_rows'], (PERSONAL, THREE, LINEAR, TREE),
                          'five_candidate_union', weight=weight)
            found = pd.read_csv(out / 'submission.csv')
            np.testing.assert_array_equal(found.Transported, expect)
            manifest = json.loads((out / 'manifest.json').read_text())
            assert manifest['v131_changes'] == plan['changes']
            assert manifest['v108'] == training['anchor'] and manifest['v131'] == training['incumbent']


if __name__ == '__main__':
    export()
