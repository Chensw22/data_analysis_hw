"""Test the registration/spending model when the seven reference opinions disagree."""
import json

import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from export_consensus_review_n26 import dig, write
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import ART,cohort,count
from unanimous_error_reach_n27 import ROOT,inputs

NAME='proposal_n28_divided_manifest_axes'
PARENT='proposal_n27_directional_foundation'


def apply(parent, votes, models):
    decision=parent>=.5
    divided=(votes!=votes[:,:1]).any(1)
    flags=divided & ((models[:,0]>=.5)==decision) & ((models[:,2]>=.5)==decision) & ((models[:,1]>=.5)!=decision)
    return np.where(flags,models[:,1],parent)


def run():
    raw,frame,y,_,fold,first,changed,_,opinions,_=inputs()
    base=first^changed
    n=len(y)
    _,_,known,fresh=cohort(raw,y,fold)
    votes=opinions>=.5
    test=frame.iloc[n:]
    parent,hard,digest=source(PARENT,test)
    np.testing.assert_array_equal(base[n:],hard)
    selection={}
    with np.load(ART/'nov28_shared_manifest_spending_axes/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        np.testing.assert_array_equal(z['fold'],fold)
        np.testing.assert_array_equal(z['y'],y)
        for tag in ['random','ordered']:
            scores=z[tag]
            p=apply(base,votes,scores)
            flag=(p>=.5)!=base
            wrong=base[:n]!=y
            selection[tag]=dict(all=count(flag[:n],known,wrong),fresh=count(flag[:n],fresh&known,wrong),
                                 accuracy=float(np.mean((p[:n][known]>=.5)==y[known])))
        test_scores=z['random'][n:]
        np.testing.assert_allclose(test_scores,z['ordered'][n:])
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    np.savez_compressed(model,passenger=test.PassengerId.to_numpy(str),parent=parent,votes=votes[n:],scores=test_scores)
    p=apply(parent,votes[n:],test_scores)
    with np.load(model) as z:
        repeat=apply(z['parent'],z['votes'],z['scores'])
    np.testing.assert_array_equal(p,repeat)
    csv=dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=repeat)).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(path.read_text()).get('file_sha256')!=check['sha256'] for path in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),model_sha256=dig(model),parents={PARENT:digest},source_file='export_divided_manifest_axes_n28.py',
                  reload_difference=0,changed=int(np.sum((p>=.5)!=hard)),training_rows=n)
    selection.update(version=manifest['version'],rule='Only seven-way divided opinions; baseline and shuffled-axis trees agree with parent, true-axis tree opposes it.',
                     limitation='The split by agreement was inspected on the same validation rows before freezing.')
    for folder in [dest,ROOT/'reports'/NAME]:
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    print(manifest,selection)


if __name__=='__main__':
    run()
