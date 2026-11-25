"""Freeze a comparable local ranking before the ten queued files receive official scores."""
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from export_tabnet_attention_n26 import NAME
from frozen_queue_shared_errors_n26 import RELEASES
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import ROOT,RESULT,collect

FILE=RESULT.parent/'frozen_queue_forecast.json'
NAMES={**RELEASES,'tabnet':NAME}


def gains(num,den,edits,n):
    if np.any((den==0)&(edits>0)):
        raise ValueError('A changed direction is missing from this resample.')
    return 100*(np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den!=0)*edits).sum(-1)/n


def forecast():
    runs,y,g,fold,v=collect()
    items=runs['full']
    q=np.concatenate([a[0] for a in items])
    new=np.concatenate([a[2]^(a[1]&a[3]['actual_and_scores']) for a in items])
    assert len(np.unique(q))==len(q)==8532
    local=np.column_stack([v[q],new])
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    files=[test_decisions(name,ids) for name in NAMES.values()]
    test=np.column_stack([h for h,_ in files])
    known=json.loads((ROOT/'reports/current_best_submission.json').read_text())
    assert known['csv_sha256']==files[0][1]
    target=[[np.sum((test[:,j]!=test[:,0])&(test[:,j]==val)) for val in (False,True)] for j in range(1,test.shape[1])]
    target=np.array(target)
    buckets,group=np.unique(g[q],return_inverse=True)
    count=np.zeros((len(buckets),len(target),2))
    signed=np.zeros_like(count)
    delta=(local[:,1:]==y[q,None]).astype(int)-(local[:,:1]==y[q,None])
    for j in range(len(target)):
        for side in (0,1):
            changed=(local[:,j+1]!=local[:,0])&(local[:,j+1]==side)
            count[:,j,side]=np.bincount(group,weights=changed,minlength=len(buckets))
            signed[:,j,side]=np.bincount(group,weights=delta[:,j]*changed,minlength=len(buckets))
    point=gains(signed.sum(0),count.sum(0),target,len(ids))
    rng=np.random.default_rng(24719)
    simulations=[]
    for _ in range(100):
        samples=rng.integers(len(buckets),size=(20,len(buckets)))
        den=count[samples].sum(1)
        num=signed[samples].sum(1)
        keep=~np.any((den==0)&(target>0),axis=(1,2))
        simulations.extend(gains(num[keep],den[keep],target,len(ids)))
    draws=np.array(simulations)
    assert len(draws)>=1900
    results={}
    for j,(short,name) in enumerate(list(NAMES.items())[1:]):
        # Refuse to replace the pre-score record with one that has already observed outcomes.
        if any((ROOT/'reports'/name/'online_attempts').glob('*.json')):
            raise RuntimeError(f'Official receipt already exists for {name}; use --evaluate instead.')
        best=draws.max(1,keepdims=True)
        winners=np.isclose(draws,best,rtol=0,atol=1.e-9)
        interval=np.quantile(draws[:,j],[.025,.975])
        r=dict(proposal=name,sha256=files[j+1][1],changed_on_test=int(target[j].sum()),local_rows=len(q),
               local_fixed=int((delta[:,j]>0).sum()),local_harmed=int((delta[:,j]<0).sum()),
               local_accuracy=float(np.mean(local[:,j+1]==y[q])),conditional_gain_pp=float(point[j]),
               order_bootstrap_gain_pp=interval.tolist(),
               conditional_public_score=float(known['public_score']+point[j]/100),
               share_top_in_bootstrap=float((winners[:,j]/winners.sum(1)).mean()))
        results[short]=r
    prior=json.loads((RESULT.parent/'tabnet_release_target_gap.json').read_text())
    assert results['tabnet']['local_fixed']==prior['local']['full']['fixed'] and results['tabnet']['local_harmed']==prior['local']['full']['harmed']
    output=dict(reference=known['proposal'],official_ref=known['submission_ref'],starting_public_score=known['public_score'],
                bootstrap_count=len(draws),local_bookings=len(buckets),test_rows=len(ids),versions=results,
                order_by_conditional_mean=sorted(results,key=lambda k:-results[k]['conditional_public_score']),
                note='Forecast uses the local net gain per changed upward/downward prediction applied to the actual number of test changes in that direction. Resampling identical whole orders across candidates captures local uncertainty and dependencies, excluding model selection, training and domain shift. Top shares describe local resampling only and do not estimate the probability of winning the competition.')
    FILE.write_text(json.dumps(output,indent=2)+'\n')
    print([(s,round(results[s]['conditional_public_score'],5)) for s in output['order_by_conditional_mean']],flush=True)


def evaluate():
    frozen=json.loads(FILE.read_text())
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    scored=[]
    for short,item in frozen['versions'].items():
        name=item['proposal']
        _,sha=test_decisions(name,ids)
        if sha!=item['sha256']:
            raise ValueError(f'{name} no longer matches the frozen forecast.')
        for file in (ROOT/'reports'/name/'online_attempts').glob('*.json'):
            receipt=json.loads(file.read_text())
            if receipt['file_sha256']==sha and receipt.get('evaluation',{}).get('Status')=='COMPLETE':
                scored.append(dict(version=short,ref=receipt['submission_ref'],predicted=item['conditional_public_score'],
                                   actual=float(receipt['evaluation']['Public Score'])))
    assert len({r['version'] for r in scored})==len(scored)
    print(json.dumps(dict(scored=len(scored),pending=len(frozen['versions'])-len(scored),results=scored),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',action='store_true')
    with threadpool_limits(1):
        if parser.parse_args().evaluate:
            evaluate()
        else:
            forecast()
