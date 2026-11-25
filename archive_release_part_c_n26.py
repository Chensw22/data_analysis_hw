"""Replay one archived fold per model family and fit the same estimator on the full training roster."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data

DEST = ROOT/'artifacts/nov26_archive_release'
REPORT = ROOT/'reports/nov23_matched_learning_curve/archive_release_part_c.json'
COLUMNS = {
    'recursive_metric': ['isotropic','diag1','diag2','full2'],
    'service_order_mixture': ['empty'],
    'maxout': ['linear','relu','maxout'],
    'zero_truncated': ['independent','truncated','reported_sleep'],
    'honest_failure_partitions': ['natural','natural_honest','enriched','enriched_honest','placebo','placebo_honest'],
    'failure_sampling_offset': ['natural','enriched','remove_offset','in_fit_offset','placebo'],
}
NEURAL = {'service_order_mixture','maxout','zero_truncated'}


def features(family, raw, groups):
    if family == 'recursive_metric':
        from full_roster_recursive_metric_n24 import design
        return (design(raw),)
    if family == 'maxout':
        from full_roster_maxout_n24 import fields
        return (fields(raw),)
    if family in ('service_order_mixture','zero_truncated'):
        from full_roster_zero_truncated_bills_n24 import inputs
        return inputs(raw)
    from full_roster_two_view_reliability_n24 import design
    views = design(raw,groups)
    if family == 'honest_failure_partitions':
        return views
    from full_roster_permutation_selection_n24 import feature_bank
    return (feature_bank(raw)[0],*views)


def fit(family, x, y, groups, t, q):
    if family == 'recursive_metric':
        from full_roster_recursive_metric_n24 import score_fold
        return score_fold(x[0],y,groups,t,q)[:,[0,1,2,4]]
    if family == 'service_order_mixture':
        from full_roster_service_order_mixture_n24 import fit as train
        return train(x[0],x[1],y,t,q)[0][:,[1]]
    if family == 'maxout':
        from full_roster_maxout_n24 import fit as train
        return np.column_stack([np.column_stack([train(x[0],y,t,q,mode,seed)
            for seed in (403,1742)]).mean(1) for mode in COLUMNS[family]])
    if family == 'zero_truncated':
        from full_roster_zero_truncated_bills_n24 import fit as train
        # Original fold 0 starts at 1981. The full fit uses the same seed.
        return np.column_stack([train(*x,y,t,q,mode,1981) for mode in COLUMNS[family]])
    if family == 'honest_failure_partitions':
        from full_roster_honest_failure_partitions_n24 import predict
        return predict(x,y,groups,t,q)
    from full_roster_failure_sampling_offset_n24 import predict
    return predict((x[0],x[1:]),y,groups,t,q)


def original(family, names, q, t, y):
    if family in ('honest_failure_partitions','failure_sampling_offset'):
        from full_roster_solo_coverage_n24 import EXPERIMENTS
        assert list(EXPERIMENTS[family]) == COLUMNS[family]
        with np.load(ROOT/f'artifacts/nov24_{family}/full_0.npz') as z:
            for key,value in [('q',q),('t',t),('truth' if 'truth' in z else 'y',y[t])]:
                np.testing.assert_array_equal(z[key],value)
            return z['p']
    with np.load(ROOT/'artifacts/nov24_remaining_cases/archive_scores.npz') as z:
        np.testing.assert_array_equal(z['y'],y)
        np.testing.assert_array_equal(q,np.flatnonzero(z['fold']==0))
        cols = [z['names'].tolist().index(name) for name in names]
        return z['full'][q][:,cols]


def worker(family):
    raw,y,g,fold = data()
    t,q = np.flatnonzero(fold!=0),np.flatnonzero(fold==0)
    names = [f'{family}/{name}' for name in COLUMNS[family]]
    x = features(family,raw,g)
    replay = fit(family,x,y,g,t,q)
    expected = original(family,names,q,t,y)
    error = np.max(np.abs(replay-expected),axis=0)
    flips = np.sum((replay>=.5)!=(expected>=.5),axis=0)
    print('replay',family,error,flips,flush=True)
    assert np.max(error)<1e-4 and not flips.any()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    merged = pd.concat([raw,test],ignore_index=True)
    groups = merged.PassengerId.str.split('_').str[0].to_numpy()
    t,q = np.arange(len(y)),np.arange(len(y),len(merged))
    assert not np.isin(groups[t],groups[q]).any()
    full_x = features(family,merged,groups)
    for old,new in zip(x,full_x,strict=True):
        np.testing.assert_array_equal(old,new[t])
    p = fit(family,full_x,y,groups,t,q)
    assert p.shape==(4277,len(names)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST/f'part_c_{family}.npz',names=names,p=p,passenger=test.PassengerId.to_numpy(str),
                        train_passenger=raw.PassengerId.to_numpy(str),train_y=y,max_diff=error,flips=flips)
    print('full',family,p.shape,flush=True)


def assemble():
    train,y,_,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    blocks=[]
    names=[]
    info={}
    for family,cols in COLUMNS.items():
        with np.load(DEST/f'part_c_{family}.npz') as z:
            np.testing.assert_array_equal(z['passenger'],test.PassengerId)
            np.testing.assert_array_equal(z['train_passenger'],train.PassengerId)
            np.testing.assert_array_equal(z['train_y'],y)
            np.testing.assert_array_equal(z['names'],[family+'/'+c for c in cols])
            blocks.append(z['p'])
            names.extend(z['names'].tolist())
            info[family]=dict(fold=0,columns=cols,max_absolute_error=z['max_diff'].tolist(),
                              hard_label_differences=z['flips'].tolist())
    combined=np.column_stack(blocks)
    p=DEST/'part_c.npz'
    with p.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream,names=names,p=combined,passenger=test.PassengerId.to_numpy(str),
                            train_passenger=train.PassengerId.to_numpy(str),train_y=y)
    p.with_suffix('.tmp').replace(p)
    with np.load(p) as z:
        np.testing.assert_array_equal(z['p'],combined)
    REPORT.write_text(json.dumps(dict(train=len(y),query=len(test),replay=info),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('family',choices=[*COLUMNS,'all','assemble'])
    family=parser.parse_args().family
    if family=='all':
        for key in COLUMNS:
            env=os.environ.copy()
            use='.venv-neural' if key in NEURAL else '.venv'
            if key not in NEURAL:
                env['DYLD_FALLBACK_LIBRARY_PATH']=str(ROOT/'.venv/lib/python3.12/site-packages/sklearn/.dylibs')
            subprocess.run([ROOT/use/'bin/python',Path(__file__),key],env=env,check=True)
        assemble()
    elif family=='assemble':
        assemble()
    elif family=='service_order_mixture':
        import torch
        torch.set_num_threads(1)
        worker(family)
    else:
        if family in NEURAL:
            import torch
            torch.set_num_threads(2)
        with threadpool_limits(2):
            worker(family)
