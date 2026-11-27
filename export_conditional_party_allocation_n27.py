"""Freeze the conditional allocation increment on the officially scored 16-edit parent."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from conditional_party_allocation_n27 import transfer
from export_budget_quarter_n23 import source
from export_consensus_review_n26 import dig, write
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import ART
from unanimous_error_reach_n27 import ROOT, inputs

NAME = 'proposal_n27_conditional_party_allocation'
PARENT = 'proposal_n27_directional_foundation'


def apply(parent, prior_odds, scores, ids):
    orders = list(pd.Series(np.arange(len(ids))).groupby(pd.Series(ids).str.split('_').str[0]).indices.values())
    probability = expit(prior_odds)
    conditioned = transfer(probability, scores[:, 1], orders)
    increment = transfer(probability, prior_odds + scores[:, 1] - scores[:, 0], orders)
    changed = ((conditioned >= .5) != (parent >= .5)) & ((increment >= .5) != (parent >= .5))
    return np.where(changed, increment, parent)


def run():
    _, frame, y, _, fold, first, changed, _, opinions, _ = inputs()
    n = len(y)
    test = frame.iloc[n:]
    ids = test.PassengerId.to_numpy(str)
    parent, decision, digest = source(PARENT, test)
    np.testing.assert_array_equal((first ^ changed)[n:], decision)
    prior_odds = (np.where(decision, 1., -1.)
                  * np.maximum(1e-6, abs(logit(np.median(opinions[n:, 1:4], axis=1).clip(.002, .998)))))
    raw = ART/'nov27_conditional_party_allocation/aligned.npz'
    with np.load(raw) as bank:
        np.testing.assert_array_equal(bank['passenger'], frame.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(bank['y'], y)
        np.testing.assert_array_equal(bank['fold'], fold)
        scores = bank['random_scores'][n:]
        np.testing.assert_allclose(scores, bank['ordered_scores'][n:])
        flag = bank['random_supported_increment'][n:]
        np.testing.assert_array_equal(flag, bank['ordered_supported_increment'][n:])
    dest = ART/NAME
    dest.mkdir(exist_ok=True)
    model = dest/'inputs.npz'
    np.savez_compressed(model, passenger=ids, parent=parent, prior_odds=prior_odds, scores=scores)
    p = apply(parent, prior_odds, scores, ids)
    with np.load(model) as z:
        reload = apply(z['parent'], z['prior_odds'], z['scores'], z['passenger'])
    np.testing.assert_array_equal(p, reload)
    np.testing.assert_array_equal((p >= .5) != decision, flag)
    csv = dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids, Transported=p >= .5)).to_csv(csv, index=False)
    pd.DataFrame(dict(PassengerId=ids, probability=p, reload_probability=reload)).to_csv(dest/'test_probabilities.csv', index=False)
    verified = validate_submission(csv, ROOT/'data/raw/test.csv')
    comparisons = []
    for receipt in sorted((ROOT/'reports').glob('*/online_attempts/*.json')):
        r = json.loads(receipt.read_text())
        if r.get('evaluation', {}).get('Status') != 'COMPLETE':
            continue
        other_csv = ART/receipt.parents[1].name/'submission.csv'
        assert verified['sha256'] != r['file_sha256']
        if other_csv.is_file() and dig(other_csv) == r['file_sha256']:
            other = pd.read_csv(other_csv)
            np.testing.assert_array_equal(other.PassengerId, ids)
            comparisons.append(dict(proposal=receipt.parents[1].name,
                                    distance=int((other.Transported != (p >= .5)).sum()),
                                    overlapping_edits=int((flag & (other.Transported != decision)).sum())))
    assert sum(flag) == 56 and len(comparisons) > 0
    manifest = dict(**verified, version=NAME.removeprefix('proposal_'), parents={PARENT:digest}, training_rows=n,
                    changed=int(flag.sum()), changed_orders=int(test.PassengerId[flag].str[:4].nunique()),
                    reload_difference=0, model_sha256=dig(model), source_file='export_conditional_party_allocation_n27.py')
    selection = dict(version=manifest['version'], rule='Apply the conditional allocation decision when the incremental allocation relative to the matched ordinary classifier agrees.',
                     source='reports/nov23_matched_learning_curve/conditional_party_allocation.json',
                     compared_scored_files=len(comparisons), nearest_scored=sorted(comparisons,key=lambda v:v['distance'])[:5],
                     reserved_for_public='Two full booking holdouts were net positive; earlier held-out orders contained no edited passengers.')
    for folder in [dest, ROOT/'reports'/NAME]:
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json', manifest)
        write(folder/'selection.json', selection)
    print(manifest, selection, flush=True)


if __name__ == '__main__':
    run()
