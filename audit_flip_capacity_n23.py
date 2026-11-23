"""Bound attainable accuracy by decision coverage without assigning any test-passenger label."""
from decimal import Decimal
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_matched_model_changes_n23 import read_bank
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_three_corrections_replay_n23 import scores, PERSONAL
from export_three_corrections_n23 import NAME as THREE
from kaggle_workflow import sha256

REPORT=ROOT/'reports/nov23_flip_capacity'
CHAMPION='proposal_v131_party_consensus'


def required(n):
    return 85*n//100+1


def same_denominator(values, max_n, method):
    """Decimal digits could be truncated or rounded; include half ties on either side."""
    candidates=[]
    for n in range(1,max_n+1):
        for p in values:
            a,r=divmod(p*n,100000)
            offsets=[-r,100000-r] if a<n else [-r]
            if method=='truncated': ok=any(0<=v<n for v in offsets)
            elif method=='nearest': ok=any(2*abs(v)<=n for v in offsets)
            else: ok=any(-n<=2*v<2*n for v in offsets)
            if not ok:break
        else:candidates.append(n)
    return candidates


def correct_count(score,n):
    p=int(Decimal(score)*100000)
    valid=[k for k in range(n+1) if -n<=2*(100000*k-p*n)<2*n]
    assert len(valid)==1
    return valid[0]


def capture(q,base,y):
    err=(base!=y)
    return dict(rows=int(q.sum()),universal_max_net_gain=int(min(q.sum(),err.sum())),covered_errors=int((q&err).sum()),
                other_old_errors=int((~q&err).sum()),unreachable_correct_needed=max(0,int(required(len(y))-(~err).sum()-(q&err).sum())))


def dev():
    row,y,frame,groups,_=load()
    output={}
    for seed in [*map(str,SEEDS),'v131']:
        versions,base=scores(frame,groups,row,y,seed)
        before=int((base==y).sum())
        pred={'personal':versions[:,0]>=.5,'three':versions.mean(1)>=.5}
        changes={k:v!=base for k,v in pred.items()}
        union=changes['personal']|changes['three']
        result=dict(n=len(y),before=before,required=required(len(y)),initial_deficit=required(len(y))-before,
                    pool_two=capture(union,base,y),candidates={})
        for k,p in pred.items():
            diff=changes[k]
            rows=int(diff.sum())
            fixed=int((diff&(base!=y)).sum())
            net=2*fixed-rows
            assert before+net==(y==p).sum()
            result['candidates'][k]=dict(changed=rows,fixed=fixed,harmed=rows-fixed,net=net,actual_correct=before+net,
                                          unchecked_optimum=before+rows,true_label_oracle=before+fixed)
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            fold=z['fold']
            for k,v in [('row',row),('y',y),('ref',base)]:np.testing.assert_array_equal(z[k],v)
            b=read_bank(row,y,fold,seed).reshape(len(y),-1)
            extra=np.c_[*[z[k] for k in ['basket_conditional','basket_direct','sleep_conditional','sleep_direct']]]
            bank=np.column_stack([b,extra])>=.5
        any_dissent=(bank!=base[:,None]).any(1)
        rate=bank.mean(1)
        majority=np.where(rate==.5,base,rate>.5)
        fixed=(majority!=base)&(y!=base)
        harmed=(majority!=base)&(y==base)
        result['pool_16']=dict(**capture(any_dissent,base,y),error_coverage_outside_two=int((any_dissent&~union&(base!=y)).sum()),
                               majority_fixed=int(fixed.sum()),majority_harmed=int(harmed.sum()))
        output[seed]=result
    return output


def public():
    info=json.loads((REPORT/'primary_scores.json').read_text())
    official=info['public_scores']
    assert info['metadata'][0]['metric']=='Categorization Accuracy'
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    n=len(test)
    def load_file(name,digest):
        file=ART/name/'submission.csv'
        assert sha256(file)==digest
        df=pd.read_csv(file)
        pd.testing.assert_series_equal(df.PassengerId,test)
        assert df.Transported.dtype==bool
        return df.Transported.to_numpy()
    boxes={}
    for pkg in (ROOT/'reports').glob('*/online_attempts/*.json'):
        entry=json.loads(pkg.read_text())
        evaluation=entry.get('evaluation',{})
        key=str(entry.get('submission_ref'))
        if evaluation.get('Status')!='COMPLETE' or key not in official:continue
        assert Decimal(official[key])==Decimal(evaluation['Public Score'])
        assert key not in boxes
        name=pkg.parents[1].name
        boxes[key]=(name,load_file(name,entry['file_sha256']))
    assert set(boxes)==set(official)
    incumbent=boxes['58653601'][1]
    assert boxes['58653601'][0]==CHAMPION
    digits=sorted({int(Decimal(s)*100000) for s in official.values()})
    possible={k:same_denominator(digits,n,k) for k in ('nearest','truncated','either')}
    assert possible['either']==[n]
    before=correct_count(official['58653601'],n)
    snapshot=json.loads((ROOT/'reports/nov23_adaptive_manifest_splines/official.json').read_text())
    third=correct_count(str(snapshot['third']),n)
    need=required(n)-before
    history=[]
    for key,(name,v) in boxes.items():
        m=int((v!=incumbent).sum())
        actual=correct_count(official[key],n)
        assert (m+actual-before)%2==0 and abs(actual-before)<=m
        history.append(dict(name=name,ref=int(key),changed=m,public_score=official[key],net_vs_incumbent=actual-before))
    history.sort(key=lambda x:x['ref'])
    proposals={}
    arrays={}
    for name in (PERSONAL,THREE):
        manifest=json.loads((ROOT/'reports'/name/'manifest.json').read_text())
        vote=load_file(name,manifest['sha256'])
        changed=vote!=incumbent
        arrays[name]=changed
        m=int(changed.sum())
        proposals[name]=dict(changed=m,max_correct=before+min(m,n-before),max_accuracy=(before+min(m,n-before))/n,
                             minimum_extra_changes=max(0,need-m))
    union=np.logical_or.reduce(list(arrays.values()))
    m=int(union.sum())
    return dict(n=n,distinct_scores=len(digits),possible_denominators=possible,scored_files=len(boxes),before=before,
                required=required(n),needed_net=need,proposals=proposals,
                observed_third=dict(date=snapshot['observed_utc'],score=snapshot['third'],correct=third,net_to_tie=third-before),
                both_candidates_union=dict(changed=m,overlap=int(np.logical_and.reduce(list(arrays.values())).sum()),
                                          max_correct=before+m,max_accuracy=(before+m)/n,minimum_extra_changes=max(0,need-m)),
                historic_changing_enough=sum(x['changed']>=need for x in history),
                historic_changing_enough_best=max(float(x['public_score']) for x in history if x['changed']>=need),
                distinct_test_decisions_ever_changed=int(np.any([v!=incumbent for _,v in boxes.values()],axis=0).sum()),
                minimum_fixed_by_change_count={str(m):(m+need+1)//2 for m in [142,180,250,350,500]},
                history=history)


if __name__=='__main__':
    with threadpool_limits(1):
        result=dict(development=dev(),official=public())
        write(REPORT/'capacity.json',result)
        print({k:v for k,v in result['official'].items() if k!='history'})
        for k,v in result['development'].items():print(k,v)
