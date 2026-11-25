"""Freeze the union of two independently tested corrections to the nine-vote release."""
import json
import numpy as np
import pandas as pd

from kaggle_workflow import sha256,validate_submission
from relational_error_corrector import write
from full_roster_booking_validation_n23 import ROOT

ART=ROOT/'artifacts'
VERSION='n25_confirmed_awake_union'
NAME='proposal_'+VERSION
MEMBERS=('proposal_n24_selected_dissent','proposal_n25_confirmed_random_fields','proposal_n25_awake_personal_tabicl')


def read_probs(spec,ids):
    result=[]
    for key in MEMBERS:
        folder=ART/key
        for filename,digest in spec[key].items():
            assert sha256(folder/filename)==digest
        hard=pd.read_csv(folder/'submission.csv')
        soft=pd.read_csv(folder/'test_probabilities.csv',float_precision='round_trip')
        np.testing.assert_array_equal(hard.PassengerId,ids)
        np.testing.assert_array_equal(soft.PassengerId,ids)
        np.testing.assert_array_equal(soft.probability,soft.reload_probability)
        np.testing.assert_array_equal(soft.probability>=.5,hard.Transported)
        result.append(soft.probability.to_numpy())
    return result


def compose(values):
    base,tree,net=values
    tree_edit=(tree>=.5)!=(base>=.5)
    net_edit=(net>=.5)!=(base>=.5)
    return np.where(tree_edit,tree,np.where(net_edit,net,base))


def export():
    test_path=ROOT/'data/raw/test.csv'
    ids=pd.read_csv(test_path,usecols=['PassengerId']).PassengerId
    out=ART/NAME
    out.mkdir(exist_ok=True)
    spec={}
    for key in MEMBERS:
        d=ART/key
        spec[key]={name:sha256(d/name) for name in ['submission.csv','test_probabilities.csv']}
        assert spec[key]['submission.csv']==json.loads((ROOT/'reports'/key/'manifest.json').read_text())['sha256']
    path=out/'composition.json'
    write(path,spec)
    source=read_probs(spec,ids)
    p=compose(source)
    replay=compose(read_probs(json.loads(path.read_text()),ids))
    np.testing.assert_array_equal(p,replay)
    hard=p>=.5
    changes=hard!=(source[0]>=.5)
    original=[(v>=.5)!=(source[0]>=.5) for v in source[1:]]
    np.testing.assert_array_equal(changes,original[0]|original[1])
    assert tuple(map(np.count_nonzero,original))==(30,135) and (original[0]&original[1]).sum()==11
    pd.DataFrame(dict(PassengerId=ids,Transported=hard)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=p,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',test_path)
    receipts=[json.loads(f.read_text()) for f in (ROOT/'reports').glob('*/online_attempts/*.json')]
    assert all(r['file_sha256']!=check['sha256'] for r in receipts)
    manifest=dict(**check,version=VERSION,model_sha256=sha256(path),reload_difference=0,training_rows=8693,
                  additional_fit_rows=0,parents={key:s['submission.csv'] for key,s in spec.items()},
                  nine_changes=int(changes.sum()),changed_to_true=int((hard&changes).sum()),
                  changed_to_false=int((~hard&changes).sum()),overlap=11)
    selection=dict(version=VERSION,rule='Retain all edits from either member; both always agree on any shared edit.',
                   evidence='reports/nov23_matched_learning_curve/confirmed_awake_union.json')
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    print(check)


if __name__=='__main__':
    export()
