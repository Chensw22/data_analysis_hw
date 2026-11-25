"""Replay and release the first six archive families in their original CPU environments."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov26_archive_release'
ARCH=ROOT/'artifacts/nov24_remaining_cases/archive_scores.npz'
REPORT=ROOT/'reports/nov23_matched_learning_curve/archive_release_part_b.json'
SPECS={
    'bill_regime_tree':('pooled','constant','registration'),
    'class_neighborhood_mass':('personal','density','shuffled'),
    'companion_transition':('personal','real','removed','swapped'),
    'error_partition_kernel':('rand','all_grad','matched_random'),
    'labelled_reference_pairs':('point','joint','near_sep','global_rel'),
    'permutation_selection':('all','selected','random','protected_select','protected_random')}
NEURAL={'bill_regime_tree','companion_transition','labelled_reference_pairs'}


def attributes(frame,family):
    if family=='bill_regime_tree':
        from full_roster_bill_regime_tree_n24 import inputs,spending_features
        c,b,_=inputs(frame)
        return c,spending_features(b)
    if family=='companion_transition':
        from full_roster_maxout_n24 import fields
        return (fields(frame),)
    if family=='permutation_selection':
        from full_roster_permutation_selection_n24 import feature_bank
        return feature_bank(frame)
    from full_roster_mistake_influence_n24 import design
    return (design(frame),)


def probability(frame, y, t, q, family, x, replay):
    g=frame.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(g[t],g[q]).any()
    if family=='bill_regime_tree':
        from full_roster_bill_regime_tree_n24 import fit
        return fit(*x,y,t,q,590)
    if family=='class_neighborhood_mass':
        from full_roster_class_neighborhood_mass_n24 import predict
        return predict(x[0],y,g,t,q,selected=(0,1,3))
    if family=='companion_transition':
        from full_roster_companion_transition_n24 import booking_index,swapped_index,fit
        codes,orders,peers=booking_index(g)
        alternate,_=swapped_index(q,codes,orders,peers,509)
        if replay:
            with np.load(ROOT/'artifacts/nov24_companion_transition/full_0.npz') as z:
                np.testing.assert_array_equal(alternate,z['alternate'])
        return fit(x[0],y,t,q,peers,alternate,632)
    if family=='error_partition_kernel':
        from full_roster_error_partition_kernel_n24 import tree
        z=((x[0]-x[0][t].mean(0))/x[0][t].std(0).clip(.2)).clip(-7,7)
        return np.column_stack([tree(z,y,g,t,q,mode,2,0) for mode in ('random','all_gradients','matched_random_gradients')])
    if family=='labelled_reference_pairs':
        from full_roster_labelled_reference_pairs_n24 import predict
        return predict(x[0],y,g,t,q,include_near_relation=False)
    from full_roster_permutation_selection_n24 import select,ensemble
    features,names,core=x
    picks=select(features,y,t,g,core,497)
    if replay:
        with np.load(ROOT/'artifacts/nov24_permutation_selection/full_0.npz') as z:
            np.testing.assert_array_equal(names,z['features'])
            np.testing.assert_array_equal(np.stack(picks[1:]),z['cols'])
    return np.column_stack([ensemble(features[:,cols],y,t,q,497) for cols in picks])


def build(family):
    raw,y,g,fold=data()
    t,q=np.flatnonzero(fold!=0),np.flatnonzero(fold==0)
    names=[family+'/'+suffix for suffix in SPECS[family]]
    with np.load(ARCH) as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        np.testing.assert_array_equal(z['y'],y)
        np.testing.assert_array_equal(z['fold'],fold)
        ix=[z['names'].tolist().index(n) for n in names]
        expected=z['full'][q][:,ix]
    original=attributes(raw,family)
    if family in NEURAL:
        import torch
        torch.set_num_threads(1)
    with threadpool_limits(1 if family in NEURAL else 2):
        matched=probability(raw,y,t,q,family,original,True)
    err=np.max(abs(matched-expected),axis=0)
    flips=((matched>=.5)!=(expected>=.5)).sum(0)
    print(family,'replay',err,flips,flush=True)
    np.testing.assert_array_equal(flips,0)
    np.testing.assert_allclose(matched,expected,atol=2e-6,rtol=0)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    joined=pd.concat([raw,test],ignore_index=True)
    assert set(raw.Cabin.dropna()).isdisjoint(test.Cabin.dropna())
    t=np.arange(len(y))
    q=np.arange(len(y),len(joined))
    x=attributes(joined,family)
    for a,b in zip(original,x,strict=True):
        if isinstance(a,np.ndarray):
            np.testing.assert_array_equal(a,b[t])
        else:
            assert a==b
    # Zero pads support source functions that use len(y) for the sentinel. All fits index training rows only.
    padded=np.r_[y,np.zeros(len(test),bool)]
    with threadpool_limits(1 if family in NEURAL else 2):
        p=probability(joined,padded,t,q,family,x,False)
    assert p.shape==(4277,len(names)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    OUT.mkdir(exist_ok=True)
    path=OUT/f'part_b_{family}.npz'
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,names=names,passenger=test.PassengerId.to_numpy(str),p=p,
                            train_passenger=raw.PassengerId.to_numpy(str),labels=y,fold0_error=err,fold0_flips=flips)
    tmp.replace(path)
    print(family,'released',flush=True)


def assemble():
    raw,y,_,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    check={}
    outputs=[]
    names=[]
    for family,suffixes in SPECS.items():
        path=OUT/f'part_b_{family}.npz'
        with np.load(path) as z:
            cols=[family+'/'+s for s in suffixes]
            for k,val in [('names',cols),('passenger',test.PassengerId),('train_passenger',raw.PassengerId),('labels',y)]:
                np.testing.assert_array_equal(z[k],val)
            np.testing.assert_array_equal(z['fold0_flips'],0)
            assert np.max(z['fold0_error'])<=2e-6
            check[family]=dict(columns=cols,maximum_replay_error=z['fold0_error'].tolist(),changed_replay_decisions=z['fold0_flips'].tolist())
            names.extend(cols)
            outputs.append(z['p'])
    p=np.column_stack(outputs)
    with np.load(ARCH) as z:
        np.testing.assert_array_equal(z['names'][:len(names)],names)
    target=OUT/'part_b.npz'
    tmp=target.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,names=names,p=p,passenger=test.PassengerId.to_numpy(str),
                            train_passenger=raw.PassengerId.to_numpy(str),labels=y)
    tmp.replace(target)
    with np.load(target) as z:
        np.testing.assert_array_equal(z['p'],p)
    REPORT.write_text(json.dumps(dict(training=len(y),test=len(test),fold=0,results=check),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['all','assemble',*SPECS],default='all',nargs='?')
    task=parser.parse_args().task
    if task in SPECS:
        build(task)
    else:
        if task=='all':
            for family in SPECS:
                env=os.environ.copy()
                if family in NEURAL:
                    interpreter=ROOT/'.venv-neural/bin/python'
                else:
                    interpreter=ROOT/'.venv/bin/python'
                    env['DYLD_FALLBACK_LIBRARY_PATH']=str(ROOT/'.venv/lib/python3.12/site-packages/sklearn/.dylibs')
                subprocess.run([str(interpreter),__file__,family],check=True,env=env)
        assemble()


if __name__=='__main__':
    main()
