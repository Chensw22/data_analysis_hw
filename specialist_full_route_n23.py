"""Replay the complete expert roster, inspect per-passenger posterior weights, and export test votes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import predictions
from broad_booking_policy_n23 import ACTIONS
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_specialist_tree_average_n23 import average, routing_partitions
from specialist_full_manifest_n23 import OUT, digest


MEMBERS = ['reference', *ACTIONS]
REPORT = ROOT/'reports/nov23_specialist_full_manifest'


def posterior_weights(levels, vote, t, q, labels):
    ll = np.log(np.where(labels[:,None], vote[t], 1-vote[t]))
    mass, mix = None, None
    for depth in reversed(range(len(levels))):
        cells = levels[depth]
        scores = np.zeros((2**depth, vote.shape[1]))
        np.add.at(scores,cells[t],ll)
        scores -= np.log(vote.shape[1])
        stop = logsumexp(scores,axis=1)
        local = np.exp(scores[cells[q]] - stop[cells[q],None])
        if mass is None:
            mix, mass = local, stop
        else:
            split = mass.reshape(-1,2).sum(1)
            both = np.logaddexp(stop,split)
            accept = np.exp(stop-both)[cells[q],None]
            mix = accept*local + (1-accept)*mix
            mass = both-np.log(2)
    return mix


def use_tree(trees, scores, t, q, labels):
    vote = .25+.5*(scores>=.5)
    weights = np.mean([posterior_weights(tree,vote,t,q,labels) for tree in trees],axis=0)
    np.testing.assert_allclose(weights.sum(1),1,atol=1e-11,rtol=0)
    routed = np.sum(weights*vote[q],axis=1)
    # Verify that reconstructing all eleven individual weights gives the
    # same probability as the existing implementation used in the search.
    original = np.mean([average(tree,vote,t,q,labels) for tree in trees],axis=0)
    np.testing.assert_allclose(routed, original, atol=1e-11, rtol=0)
    return original, weights


def decision(p, base):
    return np.where(np.isclose(p,.5,atol=1e-12,rtol=0),base>=.5,p>.5)


def family_tags():
    return ['projection_shapes' if k=='feature_projections' else k for k in ACTIONS]


def full_inputs():
    rawtest = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    ids = rawtest.PassengerId.to_numpy(str)
    scores, detail = [], {}
    for key in ['mitra','v44']:
        folder = ART/'proposal_crossfamily_v80'/key
        c = json.loads((folder/'complete.json').read_text())
        assert c['full_reload_verified'] and c['full_fit_rows']==8693
        path = folder/'probabilities.csv'
        assert c['probabilities_sha256']==digest(path)
        data = pd.read_csv(path)
        np.testing.assert_array_equal(ids,data.PassengerId)
        scores.append(data.probability.to_numpy())
    matrix = [np.mean(scores,0)]
    for member, tag in zip(ACTIONS,family_tags()):
        p = OUT/f'{tag}.npz'
        with np.load(p) as z:
            np.testing.assert_array_equal(ids,z['passenger'])
            matrix.append(z['prediction'])
            detail[member]=dict(sha256=digest(p),seconds=float(z['seconds']))
            for k in ['weight_sha256','vectors_sha256','epochs']:
                if k in z:
                    detail[member][k]=z[k].item()
            if 'support' in z:
                assert len(z['support'])==len(np.unique(z['support']))==1536
    scores = np.column_stack(matrix)
    assert scores.shape==(4277,11) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    return ids,scores,detail


def test_predictions():
    row,y,_,g,_=load()
    ids, test, detail=full_inputs()
    testgroups=np.array([i.split('_')[0] for i in ids])
    # Preserve the number of passenger records per routing tree: use the same
    # ~1,855 calibration cases and ~475 test cases in one tree as during validation.
    blocks=list(GroupKFold(9,shuffle=True,random_state=227).split(test,groups=testgroups))
    bank, weight_bank = [], []
    for seed in SEEDS:
        b, p, fold = predictions(row,y,seed,ACTIONS)
        dev=np.c_[b,p]
        for f in np.unique(fold):
            ix=np.flatnonzero(fold==f)
            calibration=list(GroupKFold(5,shuffle=True,random_state=481).split(ix,groups=g[ix]))
            forecast=np.full(len(test),np.nan)
            weights=np.zeros_like(test)
            for j,(_,q) in enumerate(blocks):
                t=ix[calibration[j%5][0]]
                matrix=np.r_[dev[t],test[q]]
                fit=np.arange(len(t))
                query=np.arange(len(t),len(matrix))
                forecast[q],weights[q] = use_tree(routing_partitions(matrix),matrix,fit,query,y[t])
            assert np.isfinite(forecast).all()
            bank.append(forecast)
            weight_bank.append(weights)
            print('test',seed,int(f),flush=True)
    final=np.mean(bank,0)
    weights=np.mean(weight_bank,0)
    np.testing.assert_allclose((weights*(.25+.5*(test>=.5))).sum(1),final,rtol=0,atol=1e-11)
    output = OUT/'submission.csv'
    vote=decision(final,test[:,0])
    pd.DataFrame(dict(PassengerId=ids,Transported=vote)).to_csv(output,index=False)
    reloaded=pd.read_csv(output)
    np.testing.assert_array_equal(reloaded.Transported,vote)
    np.testing.assert_array_equal(reloaded.PassengerId,ids)
    np.savez_compressed(OUT/'routed_test.npz',passenger=ids,expert_scores=test,expert_weights=weights,
                        component_probability=np.stack(bank),p=final)
    return dict(rows=len(test),schema=list(reloaded),positives=int(vote.sum()),changed_from_reference=int((vote!=(test[:,0]>=.5)).sum()),
                mean_weights=dict(zip(MEMBERS,weights.mean(0).tolist())),
                max_weights=dict(zip(MEMBERS,weights.max(0).tolist())),sha256=digest(output),
                path=str(output.relative_to(ROOT)),experts=detail)


def replay():
    row,y,_,g,_=load()
    report, refs, per_plan, allweights = {}, [], {}, []
    for seed in SEEDS:
        b,p,fold=predictions(row,y,seed,ACTIONS)
        scores=np.c_[b,p]
        outputs={k:np.empty(len(y)) for k in ['random','booking_order']}
        weights=np.zeros_like(scores)
        for f in np.unique(fold):
            ix=np.flatnonzero(fold==f)
            trees=routing_partitions(scores[ix])
            splits=list(GroupKFold(5,shuffle=True,random_state=481).split(ix,groups=g[ix]))
            ordered=np.array_split(np.unique(g[ix]),5)
            ordered_splits=[(np.flatnonzero(~np.isin(g[ix],u)),np.flatnonzero(np.isin(g[ix],u))) for u in ordered]
            for name,split in [('random',splits),('booking_order',ordered_splits)]:
                for t,q in split:
                    assert not np.intersect1d(g[ix[t]],g[ix[q]]).size
                    result,w=use_tree(trees,scores[ix],t,q,y[ix[t]])
                    outputs[name][ix[q]]=result
                    if name=='random':
                        weights[ix[q]]=w
        with np.load(ART/f'nov23_specialist_tree_average/{seed}.npz') as z:
            np.testing.assert_allclose(outputs['random'],z['vote/average'],rtol=0,atol=1e-11)
        report[str(seed)]={}
        for name,pr in outputs.items():
            h=decision(pr,b)==y
            stats=paired(h,(b>=.5)==y)
            report[str(seed)][name]=stats
            per_plan.setdefault(name,[]).append((pr,h))
        refs.append(b)
        allweights.append(weights)
    baseline=np.stack(refs)
    result={}
    for name,values in per_plan.items():
        combined=np.mean([v[0] for v in values],0)
        delta=np.stack([v[1] for v in values]).astype(float)-((baseline>=.5)==y)
        result[name]=dict(three_plan_mean=paired(decision(combined,baseline.mean(0))==y,(baseline.mean(0)>=.5)==y),
                          by_booking_gain_ci_pp=interval(delta.mean(0),g))
    return dict(folds=report,combined=result,mean_weights=dict(zip(MEMBERS,np.mean(allweights,axis=(0,1)).tolist())))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['replay','export'])
    args=parser.parse_args()
    with threadpool_limits(2):
        result=replay() if args.mode=='replay' else test_predictions()
    write(REPORT/(args.mode+'.json'),result)
