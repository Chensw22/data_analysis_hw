"""Export every distinct label change from the two frozen candidate files against the same reference."""
import json
import numpy as np
import pandas as pd

from export_confirmed_awake_union_n25 import ART,ROOT,compose
from kaggle_workflow import sha256,validate_submission
from relational_error_corrector import write

NAME='proposal_n26_joint_dispute_awake'
MEMBERS=('proposal_n24_selected_dissent','proposal_n26_dispute_band','proposal_n25_confirmed_awake_union')


def load(spec,ids):
    values=[]
    for name in MEMBERS:
        folder=ART/name
        for file,digest in spec[name].items():
            assert sha256(folder/file)==digest
        hard=pd.read_csv(folder/'submission.csv')
        soft=pd.read_csv(folder/'test_probabilities.csv',float_precision='round_trip')
        for frame in (hard,soft):
            np.testing.assert_array_equal(frame.PassengerId,ids)
        np.testing.assert_array_equal(soft.probability,soft.reload_probability)
        np.testing.assert_array_equal(soft.probability>=.5,hard.Transported)
        values.append(soft.probability.to_numpy())
    return values


def run():
    test=ROOT/'data/raw/test.csv'
    ids=pd.read_csv(test,usecols=['PassengerId']).PassengerId
    spec={name:{file:sha256(ART/name/file) for file in ('submission.csv','test_probabilities.csv')} for name in MEMBERS}
    for name in MEMBERS:
        assert spec[name]['submission.csv']==json.loads((ROOT/'reports'/name/'manifest.json').read_text())['sha256']
    out=ART/NAME
    out.mkdir(exist_ok=True)
    file=out/'composition.json'
    write(file,spec)
    inputs=load(spec,ids)
    p=compose(inputs)
    replay=compose(load(json.loads(file.read_text()),ids))
    np.testing.assert_array_equal(replay,p)
    base,left,right=[v>=.5 for v in inputs]
    changed=(p>=.5)!=base
    np.testing.assert_array_equal(changed,(left!=base)|(right!=base))
    assert changed.sum()==214 and np.sum((left!=base)&(right!=base))==27
    pd.DataFrame(dict(PassengerId=ids,Transported=p>=.5)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=p,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',test)
    assert all(json.loads(f.read_text())['file_sha256']!=check['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    known=json.loads((ROOT/'reports/nov23_matched_learning_curve/joint_candidate_coverage.json').read_text())['competition']
    assert known['all_three']==int(changed.sum())
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),model_sha256=sha256(file),parents={n:v['submission.csv'] for n,v in spec.items()},
        reload_difference=0,training_rows=8693,additional_fit_rows=0,nine_changes=int(changed.sum()),overlap=27,
        to_true=int((changed&~base).sum()),to_false=int((changed&base).sum()))
    selection=dict(version=manifest['version'],rule='Union of label edits relative to the same scored nine-vote parent',
        members=MEMBERS,evidence='reports/nov23_matched_learning_curve/joint_candidate_coverage.json')
    for directory in (out,ROOT/'reports'/NAME):
        write(directory/'manifest.json',manifest)
        write(directory/'selection.json',selection)
    print(manifest)


if __name__=='__main__':
    run()
