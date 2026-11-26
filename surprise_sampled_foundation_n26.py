"""Train TabICL on conditionally surprising responses, then undo the known sampling odds."""
import argparse
import hashlib
import json

import numpy as np
from scipy.special import expit,logit

from matched_state_context_n26 import inputs,ROOT
from price_implied_deck_n26 import build_features

OUT=ROOT/'artifacts/nov26_surprise_sampled_foundation'
REPORT=ROOT/'reports/nov23_matched_learning_curve/surprise_sampled_foundation.json'
KEEP=.15


def selection(p):
    return KEEP+(1-KEEP)*p, KEEP+(1-KEEP)*(1-p)


def train():
    import torch
    from booking_block_teacher_n26 import OUT as TEACHER,TEACHERS,design,split_teachers
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    from threadpoolctl import threadpool_limits
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    frame,y,g,f,_=inputs()
    x=build_features(frame,g)['fine']
    OUT.mkdir(exist_ok=True)
    n=len(y)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'fold{k}.npz'
        if path.exists():
            with np.load(path) as z:
                for key,value in [('t',t),('q',q),('labels',y[t]),('passenger',frame.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],value)
                assert np.isfinite(z['adjusted']).all()
            continue
        if k<8:
            with np.load(TEACHER/'nested'/f'fold{k}.npz') as z:
                for key,val in [('t',t),('q',q),('train_y',y[t]),('teacher_names',TEACHERS)]:
                    np.testing.assert_array_equal(z[key],val)
                soft,outer,inner=(z[key] for key in ('soft','outer_single','inner_fold'))
        else:
            raw,_=design(frame,g)
            with threadpool_limits(1):
                soft,outer,inner=split_teachers(raw,y,g,t,q,1384)
        for i in range(3):
            assert not np.isin(g[t[inner!=i]],np.r_[g[t[inner==i]],g[q]]).any()
        # A query uses a single pilot bank, as does each training passenger.
        partitions=np.array([int.from_bytes(hashlib.sha256(('surprise:'+s).encode()).digest()[:2],'big')%3 for s in g[q]])
        pilot=np.full(len(g),np.nan)
        pilot[t]=soft.mean(axis=1)
        pilot[q]=outer[partitions,np.arange(len(q))].mean(axis=1)
        sel0,sel1=selection(pilot)
        rng=np.random.default_rng(8312+k)
        keep=rng.random(len(t))<np.where(y[t],sel1[t],sel0[t])
        target=t[keep]
        control=np.sort(np.concatenate([rng.choice(t[y[t]==v],size=(y[target]==v).sum(),replace=False) for v in (False,True)]))
        assert np.array_equal(np.bincount(y[target].astype(int)),np.bincount(y[control].astype(int)))
        # Counts in the global control are matched by class, with no local sampling.
        fractions=[np.mean(y[control]==v)*len(control)/np.count_nonzero(y[t]==v) for v in (False,True)]
        design_frame=x.assign(pilot_probability=pilot)
        seed=3103+k
        sample_p=infer(design_frame,y,target,q,seed)
        uniform_p=infer(design_frame,y,control,q,seed)
        full_p=infer(design_frame,y,t,q,seed)
        adjusted=expit(logit(sample_p.clip(1e-8,1-1e-8))+np.log(sel0[q]/sel1[q]))
        uniform=expit(logit(uniform_p.clip(1e-8,1-1e-8))+np.log(fractions[0]/fractions[1]))
        truth_surprise=(pilot[t]>=.5)!=y[t]
        packet=dict(t=t,q=q,labels=y[t],passenger=frame.PassengerId.iloc[q].to_numpy(str),
                    selected=target,control=control,pilot=pilot[q],unadjusted=sample_p,adjusted=adjusted,
                    uniform=uniform,full=full_p,inner=inner,train_pilot=pilot[t],weight=DIGEST)
        for key in ('pilot','unadjusted','adjusted','uniform','full'):
            assert np.isfinite(packet[key]).all()
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**packet)
        tmp.replace(path)
        print(k,len(target),round(truth_surprise[keep].mean(),3),flush=True)


def report():
    from latent_passenger_difficulty_n26 import OUT as PANEL
    from exhaustive_family_quorums_n26 import summary
    frame,y,g,f,masks=inputs()
    n=len(y)
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        base,common,known,fresh=[z[k] for k in ('anchor','common','known','fresh')]
    scores={k:np.full(len(g),np.nan) for k in ('pilot','unadjusted','adjusted','uniform','full')}
    training=[]
    for k in range(9):
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
        with np.load(OUT/f'fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId.iloc[q])
            t=z['t']
            np.testing.assert_array_equal(z['labels'],y[t])
            assert not np.isin(g[t],g[q]).any()
            wrong=(z['train_pilot']>=.5)!=y[t]
            a,b=np.isin(t,z['selected']),np.isin(t,z['control'])
            training.append(dict(fold=k,rows=len(t),retained=int(a.sum()),source_errors=int(wrong.sum()),
                                 sampled_errors=int((wrong&a).sum()),uniform_errors=int((wrong&b).sum())))
            for name in scores:
                scores[name][q]=z[name]
    assert all(np.isfinite(v).all() for v in scores.values())
    vote={k:(s>=.5)!=base for k,s in scores.items()}
    actions=dict(**vote,new_only=vote['adjusted']&~vote['full']&~vote['uniform'],
                 both=vote['adjusted']&vote['full'],two_samples=vote['adjusted']&vote['uniform'])
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        group=z['eligible']
        blind=group&((z['pairs']>=.5)==base[:n,None,None]).all(axis=(1,2))&((z['original_panel']>=.5)==base[:n,None]).all(axis=1)
        assert np.count_nonzero(blind&(base[:n]!=y))==266
    scopes=dict(all=known,common=common,unseen=known&fresh,all31_agree=blind,
                hard_awake=common&masks['awake_paid'][:n]&~base[:n],hard_sleep=common&masks['sleep_unpaid'][:n]&base[:n])
    out={name:{s:summary(a[:n],base[:n]!=y,m,f) for s,m in scopes.items()} for name,a in actions.items()}
    output=dict(source='https://doi.org/10.1214/14-AOS1220',application='Three nested independent pilots per outer fold. Local acceptance probability is .15+.85 times the surprise probability of the observed label. TabICL log odds are restored to the unsampled population using the known log acceptance odds.',
                accuracy={k:float(((v[:n]>=.5)==y).mean()) for k,v in scores.items()},training=training,
                against_161=out,test_changes={k:int(v[n:].sum()) for k,v in actions.items()})
    REPORT.write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**scores)
    print(output['accuracy'],[(k,v['all']['net']) for k,v in out.items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','report'])
    if parser.parse_args().action=='train': train()
    else: report()
