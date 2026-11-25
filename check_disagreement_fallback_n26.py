"""Check the historical two-half veto in the actual release replay and with independently refitted vetoes."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from audit_legacy_queue_reach_n26 import contemporary
from booking_mix_confidence_transfer_n26 import partition
from evaluate_nested_release_n26 import DECK,OUT,REVIEW,panel_predictions,protocols,sources,stages
from exhaustive_family_quorums_n26 import packed,summary
from frozen_queue_shared_errors_n26 import RELEASES
from full_roster_booking_validation_n23 import ROOT,data
from monotone_countervote_envelope_n26 import monotone_rule,predict
from tabnet_attention_transfer_n26 import RESULT,collect
from target_mix_expert_routing_n26 import batches
from worst_booking_group_ensemble_n26 import first_stages

FIRST=['nine','fields','awake','fields_awake','band','all_three']
COLUMNS=[list(RELEASES).index(n) for n in FIRST]


def fit(old,labels,groups,valid):
    side=old[:,-1]
    codes=packed(old[:,:-1]!=side[:,None])
    half=partition(groups,'frozen-vote-combine:',2)
    rules={}
    for direction in (False,True):
        for j in (0,1):
            subset=valid&(codes!=0)&(side==direction)&(half==j)
            rules[direction,j]=monotone_rule(codes[subset],labels[subset]!=side[subset])[0] if subset.any() else np.array([],int)
    return rules


def apply(old,rules):
    side=old[:,-1]
    codes=packed(old[:,:-1]!=side[:,None])
    edit=np.zeros(len(side),bool)
    for direction in (False,True):
        mask=(codes!=0)&(side==direction)
        edit[mask]=predict(codes[mask],rules[direction,0])&predict(codes[mask],rules[direction,1])
    return edit


def historical_training(y,g,hist):
    h=hist['ids']
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        older,_,cold=contemporary(h,z['v131'],g[h].astype(int))
    matrix=np.column_stack([older[n] for n in [*FIRST,'review','fare','six']])
    return matrix,y[h],g[h],cold&hist['known']


def run():
    raw,y,g,f=data()
    runs,_,_,_,versions=collect()
    views=sources()[-1]
    hist=views['historical']
    frozen=fit(*historical_training(y,g,hist))
    first=first_stages(raw,views,versions)
    first['full']=versions[:,COLUMNS]
    pred={tag:panel_predictions(v,tag) for tag,v in views.items()}
    iters={tag:iter(rows) for tag,rows in runs.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    result={}
    for (key,src,dst,it,iq),b in zip(protocols(views,g,True),batches(True),strict=True):
        tag='full' if b.tag=='full' else 'historical'
        q,same,now,gate=next(iters[b.tag])
        t=b.train_ids
        np.testing.assert_array_equal(t,src['ids'][it])
        np.testing.assert_array_equal(q,dst['ids'][iq])
        assert not np.isin(g[t],g[q]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            rq=np.column_stack([z['identities_error'],z['identities_outcome']])
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],t)
            rt=z['risk']
        _,review,fare,six=stages(src,pred[tag],it,enabled[t],rt)
        train=np.column_stack([first[tag][it],review,fare,six])
        _,review,fare,six=stages(dst,pred['full'],iq,enabled[q],rq)
        query=np.column_stack([first['full'][iq],review,fare,six])
        np.testing.assert_array_equal(six,now)
        parent=now^(same&gate['actual_and_scores'])
        local=fit(train,y[t],g[t],np.ones(len(t),bool))
        chosen=apply(query,frozen)
        other=apply(query,local)
        assert not ((chosen|other)&(parent!=now)).any()
        result.setdefault(b.tag,[]).append((q,parent,chosen,other))
    report={}
    for tag,rows in result.items():
        q,parent,chosen,other=(np.concatenate([r[j] for r in rows]) for j in range(4))
        assert len(np.unique(q))==len(q)
        wh=parent!=y[q]
        nine=versions[q,0]
        report[tag]={}
        for name,v in [('global_historical',chosen),('nested_refit',other),('both',chosen&other)]:
            new=parent^v
            report[tag][name]=dict(versus_tabnet=summary(v,wh,np.ones(len(q),bool),f[q]),
                     versus_champion=summary(new!=nine,nine!=y[q],np.ones(len(q),bool),f[q]))
        print(tag,[(n,r['versus_tabnet']['net'],r['versus_champion']['net']) for n,r in report[tag].items()],flush=True)
    source=json.loads((RESULT.parent/'combine_frozen_release_disagreements.json').read_text())
    assert report['full']['global_historical']['versus_tabnet']['net']==source['direction']['monotone_both_halves']['versus_tabnet']['net']
    (RESULT.parent/'check_disagreement_fallback.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
