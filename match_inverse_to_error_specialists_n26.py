"""Test whether spending evidence can select valid counterexamples to the frozen combined candidate."""
import json
import numpy as np

from adjudicate_amount_dissent_n25 import load_fold
from disjoint_tree_fields_n25 import ART,DEST,stats
from evaluate_inverse_tabicl_n26 import OUT,ratio,odds
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from joint_candidate_coverage_n26 import branch_decisions
from matched_error_ranker_n25 import OUT as RANKER


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,archive,_,native,_,r=panels_and_reference()
    awake=raw.CryoSleep.eq(False).to_numpy()
    decisions=branch_decisions(np.arange(len(y)),f,archive,names,native,*r,awake,'full',891)
    total=(archive>=.5).sum(1)
    masks={'all':np.ones(len(y),bool),'new':~np.isin(g,g[rows]),'common':(total==0)|(total==68),
           'awake_solo_negative':(total==0)&awake&~raw.PassengerId.str[:4].duplicated(keep=False).to_numpy()}
    parts={}
    for k in range(8):
        q,t=np.flatnonzero(f==k),np.flatnonzero(f!=k)
        _,_,_,_,_,dispute,_=load_fold(k,q,t,raw,y)
        # Invert flags relative to the native sign to recover the actual model labels.
        generators=(native[q,None]>=.5)^dispute
        with np.load(RANKER/f'{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            pair=(native[q]>=.5)^(z['matched']>np.abs(odds(z['native'].clip(.002,.998))))
        with np.load(OUT/f'{k}_paid_identity.npz') as z:
            ids=z['q']
            sub=np.searchsorted(q,ids)
            np.testing.assert_array_equal(q[sub],ids)
            evidence={key:ratio(z[key]) for key in ('observed','shuffled')}
        for name,e in evidence.items():
            for branch,anchor in enumerate(decisions['all_three'][:,q]):
                contrary=np.where(anchor[sub],-e,e)>0
                changed=np.zeros(len(q),bool)
                changed[sub]=contrary
                old=generators!=anchor[:,None]
                adds={'any_precision':old.any(1),'direct_precision':old[:,0],
                      'majority_precision':old.sum(1)>=3,'matched_pairs':pair!=anchor,
                      'either_specialist':old.any(1)|(pair!=anchor),'both_specialists':old.any(1)&(pair!=anchor)}
                for variant,criterion in adds.items():
                    flip=changed&criterion
                    delta=((anchor^flip)==y[q]).astype(int)-(anchor==y[q])
                    parts.setdefault((name,variant,branch),[]).append((q,delta))
    result={}
    for source,variant,b in parts:
        if b:continue
        q,a=(np.concatenate([z[j] for z in parts[source,variant,0]]) for j in range(2))
        other,c=(np.concatenate([z[j] for z in parts[source,variant,1]]) for j in range(2))
        np.testing.assert_array_equal(q,other)
        known=r[0,q]>=0
        np.testing.assert_array_equal(a[known],c[known])
        out={}
        for name,scope in masks.items():
            m=scope[q]
            out[name]=stats(a,m&known,f[q],g[q])
            out[name]['whole_bounds']=[int(np.minimum(a,c)[m].sum()),int(np.maximum(a,c)[m].sum())]
        result.setdefault(source,{})[variant]=out
        print(source,variant,[(name,out[name]['fixed'],out[name]['harmed']) for name in masks])
    DEST.with_name('match_inverse_to_error_specialists.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    run()
