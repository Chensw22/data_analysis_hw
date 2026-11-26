"""Test domain-specific TabICL support against equally sized, label-matched random bookings."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit,logit

from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES, build_features

OUT=ROOT/'artifacts/nov26_matched_state_context'
REPORT=ROOT/'reports/nov23_matched_learning_curve/matched_state_context.json'
SIZE=1024


def inputs():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    frame=pd.concat([raw,test],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(g,groups[len(y):]).any()
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    masks={'awake_paid':frame.CryoSleep.eq(False).to_numpy()&paid,
           'sleep_unpaid':frame.CryoSleep.eq(True).to_numpy()&~paid}
    return frame,y,groups,f,masks


def one_per_order(ids,groups,rng):
    shuffled=rng.permutation(ids)
    _,positions=np.unique(groups[shuffled],return_index=True)
    return shuffled[np.sort(positions)]


def reference_pools(t,m,g,seed):
    rng=np.random.default_rng(seed)
    specific=one_per_order(t[m[t]],g,rng)[:SIZE]
    general=one_per_order(t,g,rng)
    return specific,general


def banks(t,m,g,y,seed):
    specific,general=reference_pools(t,m,g,seed)
    chosen=np.r_[general[y[general]][:y[specific].sum()],general[~y[general]][:(~y[specific]).sum()]]
    assert len(specific)==len(chosen)==SIZE and y[chosen].sum()==y[specific].sum()
    return {'specific':np.sort(specific),'matched':np.sort(chosen)}


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    frame,y,g,f,masks=inputs()
    x=build_features(frame,g)['fine']
    n=len(y)
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
        assert not np.isin(g[t],g[q]).any()
        for j,(name,mask) in enumerate(masks.items()):
            path=OUT/f'{name}_{k}.npz'
            query=q[mask[q]]
            seed=5061+17*k+j
            models=banks(t,mask,g,y,seed)
            if path.exists():
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['q'],query)
                    for tag,idx in models.items():
                        np.testing.assert_array_equal(z[tag+'_ids'],idx)
                        np.testing.assert_array_equal(z[tag+'_labels'],y[idx])
                        assert np.isfinite(z[tag]).all()
                continue
            packet=dict(q=query,passenger=frame.PassengerId.iloc[query].to_numpy(str),weight=DIGEST)
            for tag,idx in models.items():
                assert len(set(g[idx]))==SIZE and not np.isin(g[idx],g[query]).any()
                p=infer(x,y,idx,query,seed)
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                packet.update({tag:p,tag+'_ids':idx,tag+'_labels':y[idx]})
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**packet)
            tmp.replace(path)
            print(k,name,len(query),flush=True)


def report():
    from exhaustive_family_quorums_n26 import summary
    from latent_passenger_difficulty_n26 import OUT as PANEL
    frame,y,g,f,masks=inputs()
    n=len(y)
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        np.testing.assert_array_equal(z['y'],y)
        assert z['names'][-2]=='TabICL fare'
        global_p=z['scores'][:,-2]
        base,known,common,fresh=[z[k] for k in ('anchor','known','common','fresh')]
    eligible=np.logical_or.reduce(list(masks.values()))
    prob={name:np.full(len(g),np.nan) for name in ('specific','matched','matched_prior_restored')}
    for k in range(9):
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
        for j,(name,mask) in enumerate(masks.items()):
            with np.load(OUT/f'{name}_{k}.npz') as z:
                ix=q[mask[q]]
                np.testing.assert_array_equal(z['passenger'],frame.PassengerId.iloc[ix])
                np.testing.assert_array_equal(z['q'],ix)
                for tag in ('specific','matched'):
                    np.testing.assert_array_equal(z[tag+'_labels'],y[z[tag+'_ids']])
                    assert not np.isin(g[z[tag+'_ids']],g[ix]).any()
                    prob[tag][ix]=z[tag]
                t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
                spec,pool=reference_pools(t,mask,g,5061+17*k+j)
                np.testing.assert_array_equal(np.sort(spec),z['specific_ids'])
                assert not np.isin(g[pool],g[ix]).any()
                # Matching prevalence changes the global reference's sampling odds.
                shift=logit(y[pool].mean())-logit(z['matched_labels'].mean())
                prob['matched_prior_restored'][ix]=expit(logit(z['matched'].clip(1e-9,1-1e-9))+shift)
    assert all(np.array_equal(np.isfinite(p),eligible) for p in prob.values())
    votes={name:eligible&((p>=.5)!=base) for name,p in prob.items()}
    global_flip=(global_p>=.5)!=base
    specific=votes['specific']
    actions=dict(**votes, context_only=specific&~votes['matched']&~global_flip,
                 with_global=specific&global_flip, without_matched=specific&~votes['matched'])
    scopes=dict(all=known, common=common, unseen=known&fresh, awake_paid=known&masks['awake_paid'][:n],
                sleeping_no_positive=known&masks['sleep_unpaid'][:n], hard_awake=common&masks['awake_paid'][:n]&~base[:n],
                hard_sleep=common&masks['sleep_unpaid'][:n]&base[:n])
    metrics={name:{s:summary(a[:n],base[:n]!=y,mask,f) for s,mask in scopes.items()} for name,a in actions.items()}
    paired={}
    for name,mask in masks.items():
        m=mask[:n]
        paired[name]={tag:summary(votes['specific'][:n]!=votes[tag][:n],(prob[tag][:n]>=.5)!=y,m,f)
                      for tag in ('matched','matched_prior_restored')}
    result=dict(comparison='Same 1024 distinct training bookings per bank and same exact class count; full-fare features and query batch held constant.',
                against_161=metrics,specific_vs_matched=paired,test_changes={k:int(a[n:].sum()) for k,a in actions.items()})
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**prob)
    print(paired,[(k,v['all']['fixed'],v['all']['harmed']) for k,v in metrics.items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','report'])
    if parser.parse_args().action=='train': train()
    else: report()
