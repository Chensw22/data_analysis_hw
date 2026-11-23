"""Export the validated equal-probability combination of three recorded corrections."""
import json
import numpy as np
import pandas as pd

from broad_partition_evidence_n23 import ART, ROOT, write
from broad_three_corrections_replay_n23 import NAMES
from kaggle_workflow import sha256,validate_submission

VERSION = 'n23_three_corrections_mean'
NAME = 'proposal_'+VERSION


def export():
    test_file = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_file,usecols=['PassengerId'])
    parents = {}
    probabilities,reloaded = [],[]
    for name in NAMES:
        folder = ART/name
        metadata = json.loads((ROOT/'reports'/name/'manifest.json').read_text())
        assert sha256(folder/'submission.csv')==metadata['sha256']
        csv = pd.read_csv(folder/'submission.csv')
        path = folder/'test_probabilities.csv'
        soft = pd.read_csv(path,float_precision='round_trip')
        for frame in [csv,soft]:
            np.testing.assert_array_equal(frame.PassengerId,test.PassengerId)
        for col in ('probability','reload_probability'):
            values = soft[col].to_numpy()
            assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
            np.testing.assert_array_equal(values>=.5,csv.Transported)
        np.testing.assert_array_equal(soft.probability,soft.reload_probability)
        probabilities.append(soft.probability.to_numpy())
        reloaded.append(soft.reload_probability.to_numpy())
        parents[name] = dict(probability_sha256=sha256(path),submission_sha256=metadata['sha256'])
    p = np.column_stack(probabilities).mean(1)
    reload = np.column_stack(reloaded).mean(1)
    np.testing.assert_array_equal(p,reload)
    out = ART/NAME
    out.mkdir(exist_ok=True)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=reload)).to_csv(out/'test_probabilities.csv',index=False)
    csv = out/'submission.csv'
    vote = p>=.5
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=vote)).to_csv(csv,index=False)
    check = validate_submission(csv,test_file)
    np.testing.assert_array_equal(pd.read_csv(out/'test_probabilities.csv',float_precision='round_trip').probability,p)
    previous = {}
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        receipt = json.loads(path.read_text())
        assert receipt.get('file_sha256')!=check['sha256'],path
        name = path.parent.parent.name
        old = ART/name/'submission.csv'
        if old.exists() and sha256(old)==receipt.get('file_sha256'):
            frame = pd.read_csv(old)
            np.testing.assert_array_equal(frame.PassengerId,test.PassengerId)
            previous[name] = int((frame.Transported!=vote).sum())
    parent_changes = {k:int(np.sum((v>=.5)!=vote)) for k,v in zip(parents,probabilities)}
    assert min(previous.values())>0 and min(parent_changes.values())>0
    result = dict(**check,version=VERSION,parents=parents,parent_changes=parent_changes,reload_difference=0,
                  previous_files=len(previous),nearest=min(previous.values()),
                  closest=[k for k,v in previous.items() if v==min(previous.values())],
                  v131_changes=previous['proposal_v131_party_consensus'])
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',result)
        write(folder/'selection.json',dict(version=VERSION,threshold=.5))
    print(result)


if __name__=='__main__':
    export()
