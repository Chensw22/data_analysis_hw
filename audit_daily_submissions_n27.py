"""Verify scored files, changed decisions, rank snapshots and public gains for a Kaggle day."""
import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
from audit_flip_capacity_n23 import correct_count
from kaggle_workflow import sha256

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/'reports/nov23_matched_learning_curve'


def run(day, verify_live=False):
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    n=len(ids)
    assert json.loads((STUDY/'clean_fare_target_capacity.json').read_text())['compatible_denominators']==[n]
    linked={}
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        attempt=json.loads(path.read_text())
        if attempt.get('evaluation',{}).get('Status')=='COMPLETE':
            linked[attempt['file_sha256']]=(path,attempt)
    results=[]
    def decisions(path,digest):
        csv=ROOT/'artifacts'/path.parents[1].name/'submission.csv'
        assert sha256(csv)==digest
        frame=pd.read_csv(csv)
        assert frame.PassengerId.equals(ids) and frame.Transported.dtype==bool
        return frame.Transported
    daily=sorted([(p,a) for p,a in linked.values() if a['started_at_utc'][:10]==day],key=lambda v:v[1]['started_at_utc'])
    for path,a in daily:
        folder=path.parents[1]
        manifest=json.loads((folder/'manifest.json').read_text())
        parent_options=manifest.get('parents',{})
        parents={name:linked[digest] for name,digest in parent_options.items() if digest in linked}
        assert parents, folder
        name, (parent_path,parent)=max(parents.items(),key=lambda item:Decimal(item[1][1]['evaluation']['Public Score']))
        after=decisions(path,a['file_sha256'])
        before=decisions(parent_path,parent['file_sha256'])
        score=a['evaluation']['Public Score']
        old=parent['evaluation']['Public Score']
        changes=int((after!=before).sum())
        net=correct_count(score,n)-correct_count(old,n)
        assert changes>=abs(net) and (changes+net)%2==0
        results.append(dict(proposal=folder.name,ref=a['submission_ref'],score=score,compared_parent=name,
            parent_score=old,changed=changes,net=net,fixed=(changes+net)//2,harmed=(changes-net)//2,
            scored_at=a['last_checked_at_utc'],account_best_rank=a['leaderboard_snapshot']['account_best_rank'],
            team_count=a['leaderboard_snapshot']['team_count'],commit=a['commit'],sha256=a['file_sha256']))
    best=json.loads((ROOT/'reports/current_best_submission.json').read_text())
    threshold=85*n//100+1
    report=dict(day_utc=day,rows=n,completed=len(results),current_best=best['public_score'],required_correct=threshold,
        remaining_to_exceed_85=threshold-correct_count(str(best['public_score']),n),results=results,
        scope='Aggregates inferred from official scores under the previously checked compatible denominator; comparison uses the highest-scoring recorded parent. No passenger-specific test labels are inferred.')
    output=STUDY/('official_submissions_'+day+'.json')
    if verify_live:
        from await_service_pair_slot_n23 import available
        from await_half_roster_slot_n23 import snapshot
        if datetime.now(timezone.utc).date().isoformat()!=day:
            raise ValueError('Live balance is only available for the current UTC date.')
        api,limits=available()
        rank,third=snapshot(api)
        assert limits['numToday']==len(results)
        report['official_snapshot']=dict(at_utc=datetime.now(timezone.utc).isoformat(),limits=limits,
            account_rank=rank,third_place_score=third)
    elif output.exists():
        prior=json.loads(output.read_text())
        if prior['results']==results and 'official_snapshot' in prior:
            report['official_snapshot']=prior['official_snapshot']
    output.write_text(json.dumps(report,indent=2)+'\n')
    print([(r['ref'],r['score'],r['changed'],r['net']) for r in results])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('day',default='2026-11-27',nargs='?')
    parser.add_argument('--verify-live',action='store_true')
    args=parser.parse_args()
    run(args.day,args.verify_live)
