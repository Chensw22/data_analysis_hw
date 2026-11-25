"""Check how many of each frozen release's changed labels must be correct to exceed 85 percent."""
import json
from decimal import Decimal
import numpy as np
import pandas as pd

from audit_flip_capacity_n23 import correct_count,required,same_denominator
from full_roster_booking_validation_n23 import ROOT
from kaggle_workflow import sha256

NAMES=('proposal_n25_confirmed_random_fields','proposal_n25_awake_personal_tabicl',
       'proposal_n25_confirmed_awake_union','proposal_n25_awake_error_confirmation')
BASE='proposal_n24_selected_dissent'


def run():
    data=ROOT/'data/raw/test.csv'
    ids=pd.read_csv(data,usecols=['PassengerId']).PassengerId
    n=len(ids)
    officially_scored={}
    for file in (ROOT/'reports').glob('*/online_attempts/*.json'):
        r=json.loads(file.read_text())
        if r.get('evaluation',{}).get('Status')=='COMPLETE':
            officially_scored[file.parents[1].name]=r
    recorded=json.loads((ROOT/'reports/nov23_flip_capacity/primary_scores.json').read_text())['public_scores']
    score_values=set(recorded.values())|{r['evaluation']['Public Score'] for r in officially_scored.values()}
    digits={int(Decimal(str(s))*100000) for s in score_values}
    assert same_denominator(digits,n,'either')==[n]
    score=officially_scored[BASE]['evaluation']['Public Score']
    base_count=correct_count(score,n)
    def read(name):
        p=ROOT/'artifacts'/name/'submission.csv'
        expected=json.loads((ROOT/'reports'/name/'manifest.json').read_text())['sha256']
        assert sha256(p)==expected
        frame=pd.read_csv(p)
        np.testing.assert_array_equal(frame.PassengerId,ids)
        return frame.Transported.to_numpy()
    baseline=read(BASE)
    goal=required(n)
    report={}
    for name in NAMES:
        cuts=int((read(name)!=baseline).sum())
        minimum=(goal-base_count+cuts+1)//2
        report[name]=dict(changes=cuts,maximum_correct=base_count+cuts,maximum_accuracy=(base_count+cuts)/n,
                          target_possible=minimum<=cuts,minimum_correct_changes_for_target=minimum if minimum<=cuts else None)
    out=dict(reference_score=score,reference_correct=base_count,compatible_scoring_denominator=n,
             distinct_official_scores=len(digits),target_correct=goal,minimum_net_gain=goal-base_count,proposals=report)
    (ROOT/'reports/nov23_matched_learning_curve/queued_target_capacity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    run()
