"""Compare two precommitted public fare runs with their booking-held gains."""
import json
from decimal import Decimal
from pathlib import Path
import pandas as pd

from audit_flip_capacity_n23 import correct_count
from kaggle_workflow import sha256

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/'reports/nov23_matched_learning_curve'
NAMES=['proposal_n24_selected_dissent','proposal_n26_champion_fare_six','proposal_n26_joint_fare_foundations']


def read(path):
    return json.loads(path.read_text())


def run():
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    capacity=read(STUDY/'clean_fare_target_capacity.json')
    assert capacity['compatible_denominators']==[len(ids)]
    predictions=[]
    results=[]
    for name in NAMES:
        root=ROOT/'reports'/name
        summary=read(root/'followup_result.json')
        rows=[read(p) for p in (root/'online_attempts').glob('*.json')]
        result,=[r for r in rows if r['submission_ref']==summary['ref']]
        assert result['evaluation']['Status']=='COMPLETE'
        scored=Decimal(result['evaluation']['Public Score'])
        assert scored==Decimal(str(summary['score']))
        csv=ROOT/'artifacts'/name/'submission.csv'
        assert sha256(csv)==result['file_sha256']
        df=pd.read_csv(csv)
        assert df.PassengerId.equals(ids) and df.Transported.dtype==bool
        predictions.append(df.Transported)
        correct=correct_count(str(scored),len(ids))
        results.append(dict(name=name,ref=summary['ref'],score=float(scored),correct=correct,
                            rank_at_scoring=summary['rank']))
    local1=read(STUDY/'rebase_fare_six.json')['local']['full']['joint']
    local2=read(STUDY/'paired_fare_foundations.json')['fare_pair']
    for i,(local,fresh) in enumerate([(local1,'outside_historical_orders'),(local2,'fresh')],1):
        changed=int((predictions[i]!=predictions[i-1]).sum())
        delta=results[i]['correct']-results[i-1]['correct']
        assert abs(delta)<=changed and (changed+delta)%2==0
        results[i]['versus_preceding']=dict(changed=changed,public_net=delta,public_fixed=(changed+delta)//2,
            public_harmed=(changed-delta)//2,local_net=local['all']['net'],local_rows=local['all']['rows'],
            historical_unseen_net=local[fresh]['net'],historical_unseen_rows=local[fresh]['rows'])
    assert results[1]['versus_preceding']['changed']==111 and results[2]['versus_preceding']['changed']==50
    best=max(results,key=lambda r:r['correct'])
    target=(85*len(ids))//100+1
    out=dict(rows=len(ids),results=results,correct_needed_to_exceed_85=target,remaining_from_best=target-best['correct'],
             note='Aggregate corrected/harmed counts assume the previously checked compatible scoring denominator. No test outcomes are used in training.')
    (STUDY/'scored_fare_n27.json').write_text(json.dumps(out,indent=2)+'\n')
    print([(r['score'],r.get('versus_preceding')) for r in results],out['remaining_from_best'])


if __name__=='__main__':
    run()
