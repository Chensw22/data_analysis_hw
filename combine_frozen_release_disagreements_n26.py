"""Test whether the already frozen release decisions can be combined into a stronger submission."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_legacy_queue_reach_n26 import contemporary
from booking_mix_confidence_transfer_n26 import partition
from exhaustive_family_quorums_n26 import packed,summary
from forecast_frozen_queue_n26 import NAMES
from full_roster_booking_validation_n23 import ROOT,data
from held_expert_portfolios_n26 import sources
from monotone_countervote_envelope_n26 import monotone_rule,predict
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import RESULT,collect


def exact(x,y,q):
    key,inv=np.unique(x,return_inverse=True)
    benefit=np.bincount(inv,weights=np.where(y,1,-1))
    return np.isin(q,key[benefit>0])


def policies(x,wrong,q):
    roots,_=monotone_rule(x,wrong)
    return dict(exact=exact(x,wrong,q),monotone=predict(q,roots))


def run():
    raw,y,g,f=data()
    runs,truth,groups,fold,versions=collect()
    for a,b in [(truth,y),(groups,g),(fold,f)]:np.testing.assert_array_equal(a,b)
    valid=np.zeros(len(y),bool)
    last=valid.copy()
    for ids,same,base,edits in runs['full']:
        valid[ids]=True
        last[ids]=base^(same&edits['actual_and_scores'])
    assert (valid&(last==y)).sum()==7091
    hist=sources()[-1]['historical']
    ids=hist['ids']
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        old,_,cold=contemporary(ids,z['v131'],g[ids].astype(int))
    train=cold & hist['known']
    new=~np.isin(g,g[ids])
    labels=list(NAMES)
    query_ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    submitted=np.column_stack([test_decisions(name,query_ids)[0] for name in NAMES.values()])
    all_frozen=np.column_stack([versions,last])
    agrees=(all_frozen==last[:,None]).all(1)
    wrong=y!=last
    # These known older releases have identical historical/modern recipes.
    # The filtered and newest TabNet releases enter the unrestricted diagnostic only.
    selected=[k for k in old if k!='six']
    positions=[labels.index(k) for k in selected]
    previous=old['six']
    hx=packed(np.column_stack([old[k]!=previous for k in selected]))
    cx=packed(versions[:,positions]!=versions[:,[-1]])
    tx=packed(submitted[:,positions]!=submitted[:,[labels.index('six')]])
    changed=(versions[:,-1]!=last)&valid
    assert not (changed&(cx!=0)).any()
    assert not ((submitted[:,-1]!=submitted[:,labels.index('six')])&(tx!=0)).any()
    result={'release_names':labels,'test_disagreement':int((submitted!=submitted[:,[-1]]).any(1).sum())}
    take=valid&~agrees
    per_person=int((wrong&take).sum())
    result['individual_oracle']=dict(disagreeing_rows=int(take.sum()),max_net=per_person,
                                        maximum_accuracy=(7091+per_person)/valid.sum())
    signature=packed(all_frozen)
    edit=exact(signature[take],wrong[take],signature)
    result['one_outcome_per_disagreeing_signature']=summary(edit,wrong,take,f)
    cparts=[('same_rule',np.zeros(len(ids),int),np.zeros(len(y),int),np.zeros(len(tx),int)),
            ('direction',previous.astype(int),versions[:,-1].astype(int),submitted[:,labels.index('six')].astype(int))]
    half=partition(g[ids],'frozen-vote-combine:',2)
    for name,hc,cc,tc in cparts:
        actions={}
        for k in np.unique(hc):
            discovery=train&(hx!=0)&(hc==k)
            target=(cx!=0)&(cc==k)
            target_test=(tx!=0)&(tc==k)
            combined=np.r_[cx[target],tx[target_test]]
            hwrong=previous!=y[ids]
            all_fits=[policies(hx[s],hwrong[s],combined) for s in
                      [discovery,discovery&(half==0),discovery&(half==1)]]
            for mode,v in all_fits[0].items():
                for tag,p in [(mode,v),(mode+'_both_halves',all_fits[1][mode]&all_fits[2][mode])]:
                    a,b=actions.setdefault(tag,(np.zeros(len(y),bool),np.zeros(len(tx),bool)))
                    a[target],b[target_test]=p[:target.sum()],p[target.sum():]
        result[name]={}
        for method,(a,b) in actions.items():
            p=last^a
            test_prediction=submitted[:,-1]^b
            same_file=[k for j,k in enumerate(labels) if (test_prediction==submitted[:,j]).all()]
            result[name][method]=dict(versus_tabnet=summary(a,wrong,valid,f),versus_tabnet_new=summary(a,wrong,valid&new,f),
                versus_public_best=summary(p!=versions[:,0],y!=versions[:,0],valid,f),
                test_changes=int(b.sum()),test_identical_to=same_file)
        print(name,[(k,v['versus_tabnet']['net'],v['versus_tabnet_new']['net'],v['test_changes'])
                    for k,v in result[name].items()],flush=True)
    result['scope']='Observed outcomes provide hindsight bounds only. Learned adjustments to the six-way parent at disagreement rows use older whole-booking holds. The newer TabNet decisions on previously unanimous rows are retained. Fresh-order evaluation excludes every historical booking.'
    (RESULT.parent/'combine_frozen_release_disagreements.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
