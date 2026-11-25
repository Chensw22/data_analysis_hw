"""Pool recurring subpatterns across distinct bookings and check agreement on three historical shards."""
import json
from itertools import combinations
from functools import reduce
from operator import and_

import numpy as np
from threadpoolctl import threadpool_limits

from architecture_quorum_transfer_n26 import scores
from booking_mix_confidence_transfer_n26 import partition
from exhaustive_family_quorums_n26 import inputs,packed,summary
from full_roster_booking_validation_n23 import data
from held_expert_portfolios_n26 import sources
from tabnet_attention_transfer_n26 import RESULT


def counts_above(hist,width):
    count=hist.copy()
    for bit in range(width):
        v=count.reshape(-1,2,1<<bit)
        v[:,0]+=v[:,1]
    return count


def counts(code,wrong,orders,width):
    n=1<<width
    occurrence=np.bincount(code,minlength=n)
    repaired=np.bincount(code[wrong],minlength=n)
    per_order=np.zeros(n,dtype=int)
    for rows in np.split(np.argsort(orders,kind='stable'),np.cumsum(np.unique(orders,return_counts=True)[1])[:-1]):
        if not len(rows):continue
        unique=np.unique(code[rows])
        # Keep only maximal patterns in this booking. Inclusion/exclusion of their
        # intersections counts the booking once for every subpattern it supports.
        patterns=[int(s) for s in unique if not np.any(((s&unique)==s)&(unique!=s))]
        for size in range(1,len(patterns)+1):
            for selected in combinations(patterns,size):
                per_order[reduce(and_,selected)]+=1 if size%2 else -1
    return tuple(counts_above(h,width) for h in (occurrence,repaired,per_order))


def closure(rules,width):
    result=rules.copy()
    for bit in range(width):
        view=result.reshape(-1,2,1<<bit)
        view[:,1]|=view[:,0]
    return result


def verify():
    rng=np.random.default_rng(249)
    width=6
    code=rng.integers(1<<width,size=98)
    wrong=rng.random(len(code))<.35
    groups=rng.integers(40,size=len(code))
    row,fixed,order=counts(code,wrong,groups,width)
    rules=np.zeros(1<<width,bool)
    rules[[3,5,53]]=True
    selected=closure(rules,width)
    for m in range(1<<width):
        a=(code&m)==m
        assert (row[m],fixed[m],order[m])==(a.sum(),wrong[a].sum(),len(np.unique(groups[a])))
        assert selected[m]==any((m&j)==j for j in np.flatnonzero(rules))


def run():
    verify()
    raw,y,g,f=data()
    _,cur,hist=inputs()
    names,hp,cp=scores(raw,y,g,f)
    width=len(names)
    ids=sources()[-1]['historical']['ids']
    h=packed((hp>=.5)!=hist['parent'][:,None])
    c=packed((cp>=.5)!=cur['parent'][:,None])
    w=hist['wrong']
    group=g[ids]
    shard=partition(group,'recurring-subcommittee:',3)
    size=np.bitwise_count(np.arange(1<<width))
    result={}
    for scheme in ['global_rule','direction','direction_solo_paid']:
        acts={name:np.zeros(len(y),bool) for name in ('two_shards','all_three','union_checked')}
        decisions={}
        ht,ct=hist['cells'][scheme],cur['cells'][scheme]
        for level in np.unique(ht[hist['common']]):
            t=hist['common']&(ht==level)
            q=cur['common']&(ct==level)
            passes=[]
            for j in range(3):
                part=t&(shard==j)
                total,fixed,bookings=counts(h[part],w[part],group[part],width)
                passes.append((bookings>=3)&(2*fixed>total)&(size>=1))
            good=np.stack(passes)
            two=closure(good.sum(0)>=2,width)
            all_three=closure(good.all(0),width)
            audits=[]
            for j in range(3):
                # Vote rules are chosen on two shards. Test the entire proposed
                # union (including overlaps) on the third, which contributed no rule.
                learned=closure(good[np.arange(3)!=j].all(0),width)
                val=t&(shard==j)
                flip=learned[h[val]]
                audits.append(int(np.where(w[val][flip],1,-1).sum()))
            choices={'two_shards':two[c[q]],'all_three':all_three[c[q]],
                     'union_checked':all_three[c[q]] if min(audits)>0 else np.zeros(q.sum(),bool)}
            for label,p in choices.items():acts[label][q]=p
            decisions[int(level)]={'subsets_passing_all_three':int(good.all(0).sum()),'held_shard_net':audits}
        metrics={name:{part:summary(act,cur['wrong'],scope,cur['fold']) for part,scope in
                       [('including_historical',cur['common']),('new_orders',cur['common']&cur['new'])]}
                 for name,act in acts.items()}
        result[scheme]=dict(source_cells=decisions,evaluation=metrics)
        print(scheme,[(n,v['new_orders']['fixed'],v['new_orders']['harmed']) for n,v in metrics.items()],flush=True)
    result['scope']='The 18 architecture opinions match architecture_quorum_transfer. Every candidate conjunction in a supporting shard occurs in at least three distinct bookings and has positive passenger net. New orders share no booking with historical discovery. Common agreement in history predates TabNet. The three-shard recurrence rule permits subsets never observed as exact complete signatures.'
    (RESULT.parent/'recurrent_vote_coalitions.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
