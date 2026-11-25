"""Repeat exact quorum search across different architectures; compare to their continuous probabilities."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import evaluate, inputs, packed, summary
from expanded_joint_error_witnesses_n26 import FAMILIES,load_scores
from full_roster_booking_validation_n23 import data
from held_expert_portfolios_n26 import sources
from historical_solo_expert_transfer_n26 import archived_outsiders
from outside_panel_coverage_n26 import archives
from tabnet_attention_reliability_n26 import signals
from tabnet_attention_transfer_n26 import RESULT,historical_outputs

ARCHIVED=['Mitra2_adapted','EXAONE_grouped','mask_net_joint','conditional_bill_flow',
          'latent_class_mixture','conditional_profile_mixture']


def scores(raw,y,g,f):
    views=sources()[-1]
    h=views['historical']
    ids=h['ids']
    np.testing.assert_array_equal(views['full']['ids'],np.arange(len(y)))
    old=archived_outsiders(raw,y,g,ids,h['fold'])
    current,_=archives(raw,y,g,f,True)
    extra=load_scores(raw,y,g,views)
    hn=historical_outputs(raw,y,g,h)[0][ids].mean(axis=1)
    cn=signals(raw,y,g,f)[0].mean(axis=1)
    hist=np.column_stack([*[old[n] for n in ARCHIVED],hn,extra['historical'][ids]])
    cur=np.column_stack([*[current[n] for n in ARCHIVED],cn,extra['full']])
    names=[*ARCHIVED,'three_tabnet_mean',*FAMILIES]
    assert len(names)==len(set(names))==18
    for p in (hist,cur):
        assert p.shape[1]==len(names) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    return names,hist,cur


def soft_transfer(hist,cur,hp,cp):
    out={}
    # The input direction is positive when a model opposes the current parent.
    h=logit(hp.clip(.002,.998))*np.where(hist['parent'],-1.,1.)[:,None]
    c=logit(cp.clip(.002,.998))*np.where(cur['parent'],-1.,1.)[:,None]
    for scope in ['common','known']:
        t,q=hist[scope],cur[scope]
        for scheme in ('global_rule','direction','direction_solo_paid'):
            train=hist['cells'][scheme]
            query=cur['cells'][scheme]
            actions={k:np.zeros(len(q),bool) for k in ('mean_odds','calibrated_mean','all_odds')}
            actions['mean_odds']=c.mean(1)>0
            for level in np.unique(train[t]):
                ht=t&(train==level)
                cq=q&(query==level)
                for name,x,z in [('calibrated_mean',h.mean(axis=1,keepdims=True),c.mean(axis=1,keepdims=True)),
                                 ('all_odds',h,c)]:
                    model=LogisticRegression(C=.1,max_iter=400).fit(x[ht],hist['wrong'][ht])
                    actions[name][cq]=model.predict_proba(z[cq])[:,1]>.5
            out[scope+'_'+scheme]={k:{j:summary(act,cur['wrong'],m,cur['fold']) for j,m in
                    [('all_including_historical',q),('new_orders',q&cur['new'])]} for k,act in actions.items()}
    return out


def run():
    _,cur,hist=inputs()
    names,hp,cp=scores(*data())
    hist['code']=packed((hp>=.5)!=hist['parent'][:,None])
    cur['code']=packed((cp>=.5)!=cur['parent'][:,None])
    result=evaluate(names,cur,hist)
    result['continuous']=soft_transfer(hist,cur,hp,cp)
    (RESULT.parent/'architecture_quorum_transfer.json').write_text(json.dumps(result,indent=2)+'\n')
    for key,entry in result['continuous'].items():
        print(key,[(k,v['new_orders']['net']) for k,v in entry.items()],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
