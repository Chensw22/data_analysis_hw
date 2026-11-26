"""Bound each queued file's target reach using differences from scored files and aggregate accuracy."""
import json
from decimal import Decimal

import pandas as pd

from audit_flip_capacity_n23 import correct_count,required,same_denominator
from full_roster_booking_validation_n23 import ROOT
from six_way_against_scored_nine_n26 import test_decisions
from rebase_fare_six_n26 import NAME as FIRST
from export_joint_fare_foundations_n26 import NAME as SECOND


def run():
    report=ROOT/'reports/nov23_matched_learning_curve'
    best=json.loads((ROOT/'reports/current_best_submission.json').read_text())
    batch=json.loads((report/'nov26_official_queue_results.json').read_text())
    files={k:(r['proposal'],r['ref'],r['public']) for k,r in batch['scores'].items()}
    files['incumbent']=best['proposal'],best['submission_ref'],best['public_score']
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    n=len(ids)
    scores={}
    votes={}
    for key,(name,ref,score) in files.items():
        rows=[json.loads(p.read_text()) for p in (ROOT/'reports'/name/'online_attempts').glob('*.json')]
        receipt,=[r for r in rows if str(r['submission_ref'])==str(ref)]
        e=receipt['evaluation']
        assert e['Status']=='COMPLETE' and str(e['Submission Ref'])==str(ref) and float(e['Public Score'])==score
        votes[key],fingerprint=test_decisions(name,ids)
        assert fingerprint==receipt['file_sha256']
        scores[key]=correct_count(str(score),n)
    den=same_denominator([int(Decimal(str(s))*100000) for _,_,s in files.values()],n,'either')
    assert den==[n]
    result={}
    for name in [FIRST,SECOND]:
        decisions,_=test_decisions(name,ids)
        n_changes=int((decisions!=votes['incumbent']).sum())
        upper={k:min(n,correct+int((decisions!=votes[k]).sum())) for k,correct in scores.items()}
        maximum=min(upper.values())
        result[name]=dict(changes=n_changes,minimum_correct_edits_to_exceed_85=(required(n)-scores['incumbent']+n_changes+1)//2,
            upper_correct=maximum,upper_accuracy=maximum/n,tightest_scored_files=[k for k,v in upper.items() if v==maximum],
            can_rule_out_85_from_counts=maximum<required(n))
    out=dict(test_rows=n,compatible_denominators=den,incumbent_correct=scores['incumbent'],target_correct=required(n),files=result)
    (report/'clean_fare_target_capacity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    run()
