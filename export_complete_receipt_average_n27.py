"""Reproduce the calibration of the average five facility deletions on complete paid bills."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from complete_receipt_counterfactual_n27 import ART, ROOT, STORE, REPORT, FEES
from export_consensus_review_n26 import dig, write
from export_directional_foundation_n27 import NAME as BASE
from kaggle_workflow import validate_submission

NAME = 'proposal_n27_complete_receipt_average'


def apply(parent, paid, native, bank_curves, weights):
    p = bank_curves.mean(axis=0)
    delta = logit(p[:,1:].mean(axis=1).clip(.005,.995)) - logit(p[:,0].clip(.005,.995))
    shifted = np.nan_to_num(delta)*weights.mean()
    changed = paid & (shifted*np.where(parent>=.5, -1., 1.) > abs(native))
    return np.where(changed, expit(native+shifted), parent)


def run():
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId',*FEES])
    ids = test.PassengerId.to_numpy(str)
    paid = test[FEES].notna().all(axis=1).to_numpy() & test[FEES].gt(0).any(axis=1).to_numpy()
    parent_csv = ART/BASE/'submission.csv'
    parent_path = ART/BASE/'test_probabilities.csv'
    parent_file = pd.read_csv(parent_path,float_precision='round_trip')
    np.testing.assert_array_equal(parent_file.PassengerId, ids)
    np.testing.assert_array_equal(parent_file.probability,parent_file.reload_probability)
    parent = parent_file.probability.to_numpy()
    np.testing.assert_array_equal(parent>=.5, pd.read_csv(parent_csv).Transported)
    manifest_parent = json.loads((ROOT/'reports'/BASE/'manifest.json').read_text())
    assert dig(parent_csv)==manifest_parent['sha256']
    records = {}
    expected = None
    for protocol in ('random','ordered'):
        report = REPORT if protocol=='random' else REPORT.with_name(REPORT.stem+'_ordered_refit.json')
        data = json.loads(report.read_text())
        for key in ('average_random_calibrated','average_ordered_calibrated'):
            value = data['results'][key]
            assert value['scopes']['paid']['fixed'] > value['scopes']['paid']['harmed']
            records[protocol+'_'+key] = value
        with np.load(STORE/f'aligned_{protocol}_2.npz') as z:
            n = len(z['y'])
            np.testing.assert_array_equal(z['ids'][n:], ids)
            np.testing.assert_array_equal(z['anchor'][n:], parent>=.5)
            weights = z['weights_average_random']
            np.testing.assert_array_equal(weights,z['weights_average_ordered'])
            native = z['native'][n:]
            curves = np.stack([z[f'curve_bank{i}'][n:] for i in range(2)])
            action = z['average_random_calibrated'][n:]
            np.testing.assert_array_equal(action,z['average_ordered_calibrated'][n:])
            p = apply(parent,paid,native,curves,weights)
            np.testing.assert_array_equal((p>=.5)!=(parent>=.5), action)
            if expected is None:
                expected = p
            else:
                np.testing.assert_array_equal(p,expected)
    destination = ART/NAME
    destination.mkdir(exist_ok=True)
    model = destination/'inputs.npz'
    np.savez_compressed(model,ids=ids,parent=parent,paid=paid,native=native,bank_curves=curves,weights=weights)
    with np.load(model) as z:
        np.testing.assert_array_equal(z['ids'],ids)
        repeat = apply(**{field:z[field] for field in ['parent','paid','native','bank_curves','weights']})
    np.testing.assert_array_equal(repeat,expected)
    output = destination/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=expected>=.5)).to_csv(output,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=expected,reload_probability=repeat)).to_csv(destination/'test_probabilities.csv',index=False)
    check = validate_submission(output,ROOT/'data/raw/test.csv')
    assert all(json.loads(r.read_text())['file_sha256']!=check['sha256'] or r.parents[1].name==NAME
               for r in (ROOT/'reports').glob('*/online_attempts/*.json'))
    peers = {}
    for folder in ('proposal_n26_champion_fare_six','proposal_n27_independent_tabnet','proposal_n27_joint_masked_fare'):
        rows = pd.read_csv(ART/folder/'submission.csv')
        np.testing.assert_array_equal(rows.PassengerId,ids)
        differences = rows.Transported.to_numpy()!=(parent>=.5)
        peers[folder] = dict(overlap=int((differences&action).sum()),changed_from_candidate=int((differences!=action).sum()))
    manifest = dict(**check,version=NAME.removeprefix('proposal_'),training_rows=n,parents={BASE:dig(parent_csv)},
                    parent_probabilities_sha256=dig(parent_path),source_file='export_complete_receipt_average_n27.py',
                    model_sha256=dig(model),reload_difference=0,changed=int(action.sum()),
                    changed_orders=int(pd.Series(ids[action]).str[:4].nunique()),overlap=peers)
    selection = dict(version=manifest['version'],rule='Apply the mean change in log odds under five equally weighted service deletions, using nonnegative shrinkage calibrated within held-out orders. Preserve the reference prediction when the change stays on its existing side.',
                     results=records,weights=weights.tolist())
    for folder in (destination,ROOT/'reports'/NAME):
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    print(check['sha256'],manifest['changed'],peers)


if __name__=='__main__':
    run()
