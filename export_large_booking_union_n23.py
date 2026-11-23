"""Export the verified queued predictions plus linear-only corrections on parties of three or more."""
import json
import numpy as np
import pandas as pd

from broad_partition_evidence_n23 import ART,ROOT,write
from export_budget_quarter_n23 import source
from export_five_release_majority_n23 import load_votes
from kaggle_workflow import sha256,validate_submission
from pending_submission_diversity_n23 import DONOR,LINEAR

VERSION='n23_large_booking_union'
NAME='proposal_'+VERSION
INCUMBENT='proposal_v131_party_consensus'


def predict(recipe,ids):
    raw=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']).PassengerId
    labels=pd.concat([raw,ids],ignore_index=True).str.split('_').str[0]
    size=labels.map(labels.value_counts()).iloc[len(raw):].to_numpy()
    assert list(recipe['members'])==[INCUMBENT,DONOR,LINEAR]
    ref,donor,linear=load_votes(recipe,ids).T
    extra=(size>=recipe['minimum_booking_size'])&(donor==ref)&(linear!=ref)
    return donor^extra,ref,extra


def export():
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=['PassengerId'])
    _,official,h=source(INCUMBENT,test)
    recipe=dict(minimum_booking_size=3,members={INCUMBENT:h})
    for parent in (DONOR,LINEAR):
        manifest=json.loads((ROOT/'reports'/parent/'manifest.json').read_text())
        recipe['members'][parent]=manifest['sha256']
    out=ART/NAME
    out.mkdir(exist_ok=True)
    vote,ref,extra=predict(recipe,test.PassengerId)
    np.testing.assert_array_equal(ref,official)
    model=out/'recipe.json'
    write(model,recipe)
    reload,_,_=predict(json.loads(model.read_text()),test.PassengerId)
    np.testing.assert_array_equal(vote,reload)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=vote)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=vote.astype(float),
                      reload_probability=reload.astype(float))).to_csv(out/'test_probabilities.csv',index=False)
    manifest=validate_submission(out/'submission.csv',test_path)
    earlier=json.loads((ROOT/'reports/nov23_recent_family_transfer/pending_diversity.json').read_text())['files']
    receipts=[json.loads(p.read_text())['file_sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json')]
    assert manifest['sha256'] not in receipts and manifest['sha256'] not in [m['sha256'] for m in earlier.values()]
    manifest.update(version=VERSION,model_sha256=sha256(model),reload_difference=0,previous_receipts=len(receipts),
                    pending_files_checked=len(earlier),added_to_queued=int(extra.sum()),v131_changes=int((vote!=ref).sum()))
    assert manifest['v131_changes']==160+manifest['added_to_queued']
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'recipe.json',recipe)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',dict(version=VERSION,threshold=.5))
    print(manifest)


if __name__=='__main__':
    export()
