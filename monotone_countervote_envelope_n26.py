"""Exact best monotone rule on countervote patterns, plus a historical out-of-order replay."""
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order,maximum_flow
from threadpoolctl import threadpool_limits

from architecture_quorum_transfer_n26 import scores
from exhaustive_family_quorums_n26 import inputs,packed,summary
from full_roster_booking_validation_n23 import data
from tabnet_attention_transfer_n26 import RESULT


def monotone_rule(code,wrong):
    patterns,inverse=np.unique(code,return_inverse=True)
    n=len(patterns)
    source,sink=n,n+1
    gains=np.bincount(inverse,weights=np.where(wrong,1,-1),minlength=n).astype(np.int64)
    positive=np.flatnonzero(gains>0)
    negative=np.flatnonzero(gains<0)
    rows=[np.full(len(positive),source),negative]
    cols=[positive,np.full(len(negative),sink)]
    capacities=[gains[positive],-gains[negative]]
    infinity=len(code)+1
    # A selected signature forces selection of each of its observed strict supersets.
    for offset in range(0,n,128):
        keys=patterns[offset:offset+128,None]
        i,j=np.nonzero(((keys & patterns)==keys) & (keys!=patterns))
        rows.append(i+offset)
        cols.append(j)
        capacities.append(np.full(len(i),infinity))
    zero=np.flatnonzero(patterns==0)
    rows.append(zero)
    cols.append(np.full(len(zero),sink))
    capacities.append(np.full(len(zero),infinity))
    matrix=coo_matrix((np.concatenate(capacities),(np.concatenate(rows),np.concatenate(cols))),
                      shape=(n+2,n+2),dtype=np.int64).tocsr()
    flow=maximum_flow(matrix,source,sink)
    residual=matrix-flow.flow
    residual.eliminate_zeros()
    reachable=breadth_first_order(residual,source,directed=True,return_predecessors=False)
    selected=np.zeros(n,bool)
    selected[reachable[reachable<n]]=True
    assert int(gains[selected].sum())==int(gains[positive].sum()-flow.flow_value)
    # Minimal winning sets imply all their supersets, including unseen ones.
    keys=patterns[selected]
    minimal=[]
    for key in sorted(keys,key=lambda v:int(v).bit_count()):
        if not any(key&m==m for m in minimal):
            minimal.append(int(key))
    roots=np.asarray(minimal,dtype=np.int32)
    np.testing.assert_array_equal(predict(patterns,roots),selected)
    return roots,int(gains[selected].sum())


def predict(code,roots):
    out=np.zeros(len(code),bool)
    for m in roots:
        out|=(code&m)==m
    return out


def verify():
    rng=np.random.default_rng(3278)
    for _ in range(8):
        x=rng.integers(0,8,42)
        y=rng.random(42)<.44
        _,gain=monotone_rule(x,y)
        best=0
        for selection in range(1<<7):
            roots=[j+1 for j in range(7) if selection&(1<<j)]
            active=predict(x,roots)
            best=max(best,int(np.where(y[active],1,-1).sum()))
        assert gain==best


def evaluate(cur,hist):
    results={}
    for scope in ('common','known'):
        for scheme in ('global_rule','direction','direction_solo_paid'):
            q,t=cur[scope],hist[scope]
            cq,ht=cur['cells'][scheme],hist['cells'][scheme]
            oracle=np.zeros(len(q),bool)
            learned={'history':oracle.copy(),'two_history_halves':oracle.copy(),'exact_pattern':oracle.copy()}
            oracle_details={}
            for l in np.unique(cq[q]):
                a=q&(cq==l)
                b=t&(ht==l)
                seeds,gain=monotone_rule(cur['code'][a],cur['wrong'][a])
                oracle[a]=predict(cur['code'][a],seeds)
                oracle_details[int(l)]=dict(gain=gain,rules=len(seeds),
                                           one_training_passenger=int(sum(np.count_nonzero(cur['code'][a]==v)==1 for v in seeds)))
                if b.any():
                    roots,_=monotone_rule(hist['code'][b],hist['wrong'][b])
                    learned['history'][a]=predict(cur['code'][a],roots)
                    y=hist['wrong'][b]
                    p,inv=np.unique(hist['code'][b],return_inverse=True)
                    profitable=np.bincount(inv,weights=np.where(y,1,-1))>0
                    learned['exact_pattern'][a]=np.isin(cur['code'][a],p[profitable & (p>0)])
                    halves=[]
                    for s in (False,True):
                        part=b & (hist['half']==s)
                        if part.any():
                            roots,_=monotone_rule(hist['code'][part],hist['wrong'][part])
                            halves.append(predict(cur['code'][a],roots))
                    if len(halves)==2:
                        learned['two_history_halves'][a]=halves[0]&halves[1]
            upper=summary(oracle,cur['wrong'],q,cur['fold'])
            upper['overall_known_accuracy']=(7091+upper['net'])/8532
            out={'oracle':upper,'oracle_cell_support':oracle_details}
            for name,act in learned.items():
                out[name]={key:summary(act,cur['wrong'],mask,cur['fold']) for key,mask in
                           [('all_including_history',q),('later_orders',q&cur['new'])]}
            results[scope+'_'+scheme]=out
            print(scope,scheme,upper['net'],[(n,out[n]['later_orders']['net']) for n in learned],flush=True)
    return results


def run():
    from booking_mix_confidence_transfer_n26 import partition
    from held_expert_portfolios_n26 import sources
    verify()
    _,cur,hist=inputs()
    raw,y,g,f=data()
    hist['half']=partition(g[sources()[-1]['historical']['ids']],'monotone-history-half:',2).astype(bool)
    result={'legacy_families':evaluate(cur,hist)}
    names,hp,cp=scores(raw,y,g,f)
    hist['code']=packed((hp>=.5)!=hist['parent'][:,None])
    cur['code']=packed((cp>=.5)!=cur['parent'][:,None])
    result['architectures']=evaluate(cur,hist)
    result['architecture_names']=names
    result['scope']='Rules may take any OR of positive countervote conjunctions within each predeclared cell. Adding a countervote cannot cancel an existing action. A rule never flips an all-agree signature. The exact data-dependent oracle is in-sample. Real transfer uses two distinct periods; unseen signatures inherit only historically selected subsets.'
    (RESULT.parent/'monotone_countervote_envelope.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
