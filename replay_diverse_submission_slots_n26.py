"""Retrospective comparison of local ranking and prediction diversity for two or three slots."""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT
from kaggle_workflow import sha256

REPORT=ROOT/'reports/nov23_matched_learning_curve'
ART=ROOT/'artifacts'


def verified(name,digest,ids):
    path=ART/name/'submission.csv'
    assert sha256(path)==digest
    frame=pd.read_csv(path)
    np.testing.assert_array_equal(frame.PassengerId,ids)
    assert frame.Transported.dtype==bool
    return frame.Transported.to_numpy()


def choose(scores,votes,slots):
    ranks=sorted(scores,key=lambda n:(-scores[n],n))
    pool=[n for n in ranks if scores[ranks[0]]-scores[n]<=.01+1e-12]
    if len(pool)<slots:return None
    picked=[pool[0]]
    while len(picked)<slots:
        remaining=[n for n in pool if n not in picked]
        # Ties use the preset local rank, never a public score.
        picked.append(max(remaining,key=lambda n:min(np.count_nonzero(votes[n]!=votes[m]) for m in picked)))
    return pool,{'top_local':pool[:slots],'most_different':picked}


def historical(batches,inventory,pred,slots,strict_ids):
    result={}
    for key,b in batches.items():
        if key.startswith('n25_eight_known'):continue
        versions=b['versions']
        if strict_ids and key.startswith('early_common_folds'):
            versions={n:v for n,v in versions.items() if v['evidence']['identity']==v['evidence']['baseline_identity']=='row IDs'}
        local={n:v['local_net']/v['local_rows'] for n,v in versions.items()}
        if len(local)<=slots:continue
        selection=choose(local,pred,slots)
        if not selection:continue
        pool,picks=selection
        if len(pool)<=slots:continue
        base={v['baseline'] for v in versions.values()}
        assert len(base)==1
        anchor=inventory[base.pop()]['score']
        score={n:inventory[n]['score'] for n in pool}
        best=max(score.values())
        out={}
        for mode,names in picks.items():
            best_chosen=max(score[n] for n in names)
            out[mode]=dict(names=names,best_score=best_chosen,hit_best=best_chosen==best,
                           improved=best_chosen>anchor,regret_pp=100*(best-best_chosen))
        lottery=[max(score[n] for n in (pool[0],*pair)) for pair in combinations(pool[1:],slots-1)]
        result[key]=dict(eligible=len(pool),best_available=best,reference_score=anchor,choices=out,
                         random_with_local_first=dict(hit_best=float(np.mean(np.array(lottery)==best)),
                             improved=float(np.mean(np.array(lottery)>anchor)),mean_regret_pp=float(100*(best-np.mean(lottery)))))
    summary={}
    for method in ('top_local','most_different','random_with_local_first'):
        vals=[v[method] if method=='random_with_local_first' else v['choices'][method] for v in result.values()]
        summary[method]=dict(groups=len(vals),sum_hit_best=float(sum(v['hit_best'] for v in vals)),
                             sum_improved=float(sum(v['improved'] for v in vals)),
                             avg_gap_to_best_pp=float(np.mean([v.get('regret_pp',v.get('mean_regret_pp')) for v in vals])) if vals else None)
    return dict(batches=result,summary=summary)


def run():
    source=json.loads((REPORT/'cross_family_submission_choices.json').read_text())
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    pred={n:verified(n,v['sha256'],ids) for n,v in source['inventory'].items()}
    result={}
    for slots in (2,3):
        for strict in (False,True):
            result[f'{slots}_slots_'+('explicit_ids' if strict else 'all')]=historical(
                source['batches'],source['inventory'],pred,slots,strict)
    frozen=json.loads((REPORT/'frozen_queue_forecast.json').read_text())
    pending=frozen['versions']
    votes={n:verified(v['proposal'],v['sha256'],ids) for n,v in pending.items()}
    local={n:v['local_accuracy'] for n,v in pending.items()}
    distance={n:{m:int(np.count_nonzero(votes[n]!=votes[m])) for m in pending} for n in pending}
    result['pending']=dict(local_accuracy=local,distances=distance,
                           hypothetical={str(k):choose(local,votes,k)[1] for k in (2,3)})
    result['scope']='Within a retrospective common-date, common-baseline, common-OOF batch, retain candidates within one local accuracy point of the best. Analyze only opportunities with more eligible files than slots. The first choice is always the local leader; subsequent choices use only local accuracy or actual unlabeled differences. The outcome metric is best official score obtained with the budget. Families and dates are dependent; file availability may have differed within a day. Existing queue order is untouched.'
    print({k:v['summary'] for k,v in result.items() if 'summary' in v},flush=True)
    (REPORT/'replay_diverse_submission_slots.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()
