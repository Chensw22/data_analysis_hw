"""Exact search over every nonempty family subset and every threshold for opposing the release."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from audit_legacy_queue_reach_n26 import contemporary
from booking_mix_confidence_transfer_n26 import partition
from full_roster_booking_validation_n23 import ROOT,data
from held_expert_portfolios_n26 import sources
from price_implied_deck_n26 import FEES
from tabnet_attention_transfer_n26 import RESULT,collect


def packed(votes):
    return votes.astype(np.int32) @ (1 << np.arange(votes.shape[1],dtype=np.int32))


def all_gains(codes, wrong, width):
    # The polynomial at row M stores signed counts by |M intersect votes|.
    # Process one family at a time. Excluding it combines both vote states;
    # including it shifts the state which casts a countervote by one degree.
    n=1<<width
    table=np.zeros((n,width+1),np.int32)
    table[:,0]=np.bincount(codes,weights=np.where(wrong,1,-1),minlength=n).astype(np.int32)
    for bit in range(width):
        step=1<<bit
        view=table.reshape(-1,2,step,width+1)
        left=view[:,0,:,:bit+1].copy()
        right=view[:,1,:,:bit+1].copy()
        view[:,0,:,:bit+1]=left+right
        view[:,1,:,:bit+1]=left
        view[:,1,:,1:bit+2]+=right
    return np.cumsum(table[:,::-1],axis=1,dtype=np.int32)[:,::-1]


def action(code, rule):
    mask,k=rule
    if mask==0:
        return np.zeros(len(code),bool)
    return np.bitwise_count(code & mask)>=k


def choose(objective):
    best=objective[:,1:].max()
    if best<=0:
        return 0,1
    m,k=np.nonzero(objective[:,1:]==best)
    # Prefer simpler subsets, then larger thresholds, then a stable mask order.
    winner=np.lexsort((m,-k,np.bitwise_count(m.astype(np.uint32))))[0]
    return int(m[winner]),int(k[winner]+1)


def check_transform():
    rng=np.random.default_rng(745)
    for width in (1,4,7):
        code=rng.integers(0,2**width,80,dtype=np.int32)
        wrong=rng.random(len(code))<.3
        gains=all_gains(code,wrong,width)
        for m in range(1<<width):
            for k in range(1,width+1):
                a=action(code,(m,k))
                assert gains[m,k]==np.where(wrong[a],1,-1).sum()


def summary(selected,wrong,scope,fold):
    fixed=selected&scope&wrong
    harmed=selected&scope&~wrong
    delta=fixed.astype(int)-harmed.astype(int)
    return dict(rows=int(scope.sum()),errors=int((scope&wrong).sum()),fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                net=int(delta.sum()),net_by_fold=[int(delta[fold==k].sum()) for k in np.unique(fold)])


def fit(code,wrong,scope,cell,shard,width,robust):
    rules={}
    for level in np.unique(cell[scope]):
        m=scope&(cell==level)
        if robust:
            worst=None
            for j in range(3):
                t=m&(shard==j)
                if not t.any():
                    worst=np.zeros((1,2))
                    break
                score=all_gains(code[t],wrong[t],width).astype(np.float32)/t.sum()
                worst=score if worst is None else np.minimum(worst,score)
            rules[int(level)]=choose(worst)
        else:
            rules[int(level)]=choose(all_gains(code[m],wrong[m],width))
    return rules


def apply(rules,code,cell):
    out=np.zeros(len(code),bool)
    for level,rule in rules.items():
        mask=cell==level
        out[mask]=action(code[mask],rule)
    return out


def pretty(rules,names):
    return {str(level):dict(threshold=k,families=[n for i,n in enumerate(names) if m&(1<<i)])
            for level,(m,k) in rules.items()}


def inputs():
    raw,y,g,f=data()
    runs,truth,group,fold,_=collect()
    for a,b in [(truth,y),(group,g),(fold,f)]:
        np.testing.assert_array_equal(a,b)
    parent=np.zeros(len(y),bool)
    common=parent.copy()
    known=parent.copy()
    for ids,same,v,review in runs['full']:
        assert not known[ids].any()
        known[ids]=True
        change=same&review['actual_and_scores']
        parent[ids]=v^change
        common[ids]=same&~change
    assert known.sum()==8532 and common.sum()==7870 and (known&(parent==y)).sum()==7091
    views=sources()[-1]
    h,cur=views['historical'],views['full']
    ids=h['ids']
    np.testing.assert_array_equal(cur['ids'],np.arange(len(y)))
    names=list(cur['names'][:18])
    assert names==list(h['names'][:18]) and len(set(names))==18
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],ids)
        np.testing.assert_array_equal(z['y'],y[ids])
        old,_,cold=contemporary(ids,z['v131'],g[ids].astype(int))
    old_parent=old['six']
    old_common=(np.column_stack(list(old.values()))==old_parent[:,None]).all(1)
    new=~np.isin(g,g[ids])
    solo=raw.groupby(g).PassengerId.transform('size').eq(1).to_numpy()
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    def cells(parent,single,positive):
        return dict(global_rule=np.zeros(len(parent),int),direction=parent.astype(int),
                    direction_solo_paid=4*parent.astype(int)+2*single+positive)
    return names, dict(code=packed(cur['votes'][:,:18]!=parent[:,None]),wrong=parent!=y,fold=f,parent=parent,
                       known=known,common=common,new=new,cells=cells(parent,solo,paid)),dict(
        code=packed(h['votes'][:,:18]!=old_parent[:,None]),wrong=old_parent!=y[ids],parent=old_parent,
        known=cold&h['known'],common=cold&h['known']&old_common,cells=cells(old_parent,solo[ids],paid[ids]),
        shard=partition(g[ids],'quorum-source-shard:',3),fold=h['fold'])


def evaluate(names,cur,hist):
    width=len(names)
    result=dict(families=names,number_of_nonempty_subsets=(1<<width)-1,
                number_of_subset_thresholds=width*(1<<(width-1)),target_gain=162,experiments={})
    for label in ('common','known'):
        training=hist[label]
        target=cur[label]
        for scheme in cur['cells']:
            a=cur['cells'][scheme]
            b=hist['cells'][scheme]
            oracle=fit(cur['code'],cur['wrong'],target,a,None,width,False)
            flip=apply(oracle,cur['code'],a)
            upper=summary(flip,cur['wrong'],target,cur['fold'])
            upper['overall_known_accuracy']=(7091+upper['net'])/8532
            entry=dict(hindsight=upper,hindsight_rules=pretty(oracle,names),transfer={})
            for robust in (False,True):
                rules=fit(hist['code'],hist['wrong'],training,b,hist['shard'],width,robust)
                selected=apply(rules,cur['code'],a)
                tr=apply(rules,hist['code'],b)
                entry['transfer']['worst_historical_third' if robust else 'pooled_history']=dict(
                    rules=pretty(rules,names),hist=summary(tr,hist['wrong'],training,hist['fold']),
                    new_orders=summary(selected,cur['wrong'],target&cur['new'],cur['fold']),
                    full_including_historical=summary(selected,cur['wrong'],target,cur['fold']))
            result['experiments'][label+'_'+scheme]=entry
            print(label,scheme,upper['net'],[(n,t['new_orders']['net']) for n,t in entry['transfer'].items()],flush=True)
    result['interpretation']='Hindsight is an exact ceiling only for these predictors, scopes, partitions and common subset/quorum per cell. Historical selection excludes all new orders. The historical baseline predates the newest TabNet correction. Current base votes are individual order-held-out predictions. Full transfer includes historical discovery passengers.'
    return result


def run():
    check_transform()
    result=evaluate(*inputs())
    (RESULT.parent/'exhaustive_family_quorums.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
