"""Scale the two fixed TabICL specialists to all available passengers in their training cohorts."""
import argparse
import hashlib
import json

import numpy as np

from matched_state_context_n26 import inputs,ROOT,OUT as SMALL
from price_implied_deck_n26 import build_features

OUT=ROOT/'artifacts/nov26_full_state_context'
REPORT=ROOT/'reports/nov23_matched_learning_curve/full_state_context.json'


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    frame,y,g,f,masks=inputs()
    x=build_features(frame,g)['fine']
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(len(y))
        q=np.flatnonzero(f==k) if k<8 else np.arange(len(y),len(frame))
        for j,(name,mask) in enumerate(masks.items()):
            ids,query=t[mask[t]],q[mask[q]]
            assert not np.isin(g[ids],g[query]).any()
            path=OUT/f'{name}_{k}.npz'
            if path.exists():
                with np.load(path) as z:
                    for key,v in [('t',ids),('q',query),('labels',y[ids]),('passenger',frame.PassengerId.iloc[query])]:
                        np.testing.assert_array_equal(z[key],v)
                    assert np.isfinite(z['p']).all()
                continue
            p=infer(x,y,ids,query,5061+17*k+j)
            assert np.isfinite(p).all()
            temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,t=ids,q=query,labels=y[ids],passenger=frame.PassengerId.iloc[query].to_numpy(str),p=p,weight=DIGEST)
            temp.replace(path)
            print(k,name,len(ids),len(query),flush=True)


def report():
    from latent_passenger_difficulty_n26 import OUT as PANEL
    from exhaustive_family_quorums_n26 import summary
    frame,y,g,f,masks=inputs()
    n=len(y)
    with np.load(PANEL/'panel.npz') as z, np.load(SMALL/'aligned.npz') as sm:
        for bank in [z,sm]:
            np.testing.assert_array_equal(bank['passenger'],frame.PassengerId)
            np.testing.assert_array_equal(bank['fold'],f)
        assert z['names'][-2]=='TabICL fare'
        global_p=z['scores'][:,-2]
        base,common,known,fresh=[z[k] for k in ('anchor','common','known','fresh')]
        small,matched=sm['specific'],sm['matched']
    p=np.full(len(g),np.nan)
    support={}
    for k in range(9):
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
        for j,(name,mask) in enumerate(masks.items()):
            ix=q[mask[q]]
            with np.load(OUT/f'{name}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],ix)
                np.testing.assert_array_equal(z['passenger'],frame.PassengerId.iloc[ix])
                np.testing.assert_array_equal(z['labels'],y[z['t']])
                assert not np.isin(g[ix],g[z['t']]).any()
                p[ix]=z['p']
                support.setdefault(name,[]).append(len(z['t']))
    valid=np.logical_or.reduce(list(masks.values()))
    assert np.array_equal(valid,np.isfinite(p))
    direction=lambda score:valid&((score>=.5)!=base)
    large=direction(p)
    small_flip=direction(small)
    global_flip=direction(global_p)
    agree=large&small_flip
    actions=dict(full=large, both_sizes=agree, stable_specific_only=agree&~direction(matched)&~global_flip,
                 large_only=large&~small_flip&~global_flip, both_full_models=large&global_flip)
    scopes=dict(all=known, common=common, unseen=known&fresh,awake=known&masks['awake_paid'][:n],sleep=known&masks['sleep_unpaid'][:n],
        hard_awake=common&masks['awake_paid'][:n]&~base[:n],hard_sleep=common&masks['sleep_unpaid'][:n]&base[:n])
    gains={name:{s:summary(a[:n],base[:n]!=y,m,f) for s,m in scopes.items()} for name,a in actions.items()}
    comparison={name:{old:summary((score[:n]>=.5)!=(p[:n]>=.5),(score[:n]>=.5)!=y,m[:n],f)
                      for old,score in [('small',small),('global',global_p)]} for name,m in masks.items()}
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,p=p)
    REPORT.write_text(json.dumps(dict(against_161=gains,cohort_comparison=comparison,support=support,
                     test_changes={k:int(v[n:].sum()) for k,v in actions.items()}),indent=2)+'\n')
    print(comparison,[(k,r['all']['net']) for k,r in gains.items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','report'])
    if parser.parse_args().action=='train': train()
    else: report()
