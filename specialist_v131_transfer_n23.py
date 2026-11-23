"""Test corrections against v131 using all ten experts refit on the historical folds."""
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, write
from error_overlap_diagnostic import load_authenticated
from specialist_full_route_n23 import decision, family_tags, routing_partitions, use_tree
from specialist_v131_inputs_n23 import OUT

REPORT=ROOT/'reports/nov23_specialist_v131_transfer'


def policies(v80,p108,v131,experts,source_route,target_route,source_weights):
    locked=v131!=(p108>=.5)
    routed_source=decision(source_route,v80)
    routed_target=decision(target_route,p108)
    v80_vote=v80>=.5
    expert_dispute=routed_source!=v80_vote
    opportunity=expert_dispute & (routed_source!=v131) & ~locked
    # Hard posterior votes are on the artificial 0.25/0.75 scale.  Use the
    # same learned weights on the experts' native probabilities for the offset.
    continuous=(np.c_[v80,experts]*source_weights).sum(1)
    shifted=p108 + continuous-v80
    raw_shift=p108 + source_route-v80
    moves={
        'replace_unlocked':~locked & (routed_target!=v131),
        'transfer_disputes':opportunity,
        'transfer_both_anchors':opportunity & (routed_source==routed_target),
        'transfer_soft_delta':opportunity & ((shifted>=.5)==routed_source),
        'transfer_vote_delta':opportunity & ((raw_shift>=.5)==routed_source),
    }
    return {k:v131 ^ m for k,m in moves.items()}


def read():
    inp,_,fold,prob,_=load_authenticated()
    row,y,g=inp['dev'],inp['y'].astype(bool),np.asarray(inp['groups'])
    with np.load(ART/'nov23_incumbent_budget_transfer/oof.npz') as z:
        for k,v in [('row',row),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        p108,v131,cold=z['v108'],z['v131'],z['unseen']
    cols=[]
    for tag in family_tags():
        with np.load(OUT/f'{tag}.npz') as z:
            for k,v in [('row',row),('y',y),('fold',fold)]:
                np.testing.assert_array_equal(z[k],v)
            for f in np.unique(fold):
                q=row[fold==f]
                t=row[(fold!=f)&~np.isin(g,g[fold==f])]
                np.testing.assert_array_equal(z[f'train_{f}'],t)
                np.testing.assert_array_equal(z[f'query_{f}'],q)
            cols.append(z['p'])
    return row,y,g,fold,(prob['v44']+prob['rawMitra'])/2,p108,v131,cold,np.column_stack(cols)


def main():
    row,y,g,fold,v80,p108,v131,cold,experts=read()
    methods=['seed481','seed179','ordered']
    routed,weight={k:[] for k in methods},{}
    for label,anchor in [('v80',v80),('v108',p108)]:
        x=np.column_stack([anchor,experts])
        result={k:np.full(len(y),np.nan) for k in methods}
        mix={k:np.zeros_like(x) for k in methods}
        for f in np.unique(fold):
            ix=np.flatnonzero(fold==f)
            trees=routing_partitions(x[ix])
            randoms=[list(GroupKFold(5,shuffle=True,random_state=s).split(ix,groups=g[ix])) for s in (481,179)]
            ordered=[]
            for sel in np.array_split(np.unique(g[ix]),5):
                mask=np.isin(g[ix],sel)
                ordered.append((np.flatnonzero(~mask),np.flatnonzero(mask)))
            for method,splits in zip(methods,[*randoms,ordered]):
                for t,q in splits:
                    assert not np.intersect1d(g[ix[t]],g[ix[q]]).size
                    result[method][ix[q]],mix[method][ix[q]]=use_tree(trees,x[ix],t,q,y[ix[t]])
        for k in methods:
            assert np.isfinite(result[k]).all()
            routed[k].append(result[k])
            if label=='v80':
                weight[k]=mix[k]
    # Average over the two pre-existing random calibration seeds without
    # selecting the better-performing random assignment using these outcomes.
    routed['random_mean']=[np.mean([routed[k][i] for k in methods[:2]],0) for i in range(2)]
    weight['random_mean']=np.mean([weight[k] for k in methods[:2]],0)
    records={}
    saved={'row':row,'y':y,'fold':fold,'v131':v131,'cold':cold,'v80':v80,'v108':p108}
    inc=v131==y
    for cal,(source,target) in routed.items():
        preds=policies(v80,p108,v131,experts,source,target,weight[cal])
        records[cal]={}
        for k,v in preds.items():
            assert np.all(v[v131!=(p108>=.5)]==v131[v131!=(p108>=.5)])
            hit=v==y
            gain=hit.astype(float)-inc
            records[cal][k]=dict(vs_champion=paired(hit,inc),cold=paired(hit[cold],inc[cold]),
                         net_by_fold=[int(gain[fold==f].sum()) for f in np.unique(fold)],
                         cold_by_fold=[int(gain[(fold==f)&cold].sum()) for f in np.unique(fold)],
                         joint_ci_pp=interval(gain,g),cold_ci_pp=interval(gain[cold],g[cold]))
            saved[cal+'_'+k]=v
        print(cal,{k:(v['vs_champion']['fixed']-v['vs_champion']['harmed'],v['cold']['fixed']-v['cold']['harmed']) for k,v in records[cal].items()},flush=True)
    np.savez_compressed(OUT/'oof_comparison.npz',**saved)
    write(REPORT/'replay.json',dict(rows=len(y),reference_acc=float(inc.mean()),v80_acc=float(((v80>=.5)==y).mean()),
              cold_count=int(cold.sum()),locked_count=int((v131!=(p108>=.5)).sum()),methods=records))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
