"""Replay six archive families on fold zero, then refit their 24 configurations on all bookings."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data

# The project uses the base environment for tree dependencies and the neural environment for torch.
if not any(Path(entry, 'torch').is_dir() for entry in sys.path):
    sys.path.append(str(ROOT/f'.venv-neural/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'))
import torch

import broad_error_transport_projection_n24 as projection
import full_roster_fixed_bill_margins_n24 as margins
import full_roster_predicate_tables_n24 as predicate
import full_roster_response_channels_n24 as response
import full_roster_spending_shape_n24 as shape
import full_roster_two_view_reliability_n24 as reliability
from full_roster_bill_allocation_cycles_n24 import partitions
from full_roster_solo_coverage_n24 import EXPERIMENTS

OUT=ROOT/'artifacts/nov26_archive_release/part_a.npz'
MODES={'response_channels':response.MODES,'broad_error_transport_projection':projection.MODES,
       **{family:EXPERIMENTS[family] for family in ('predicate_tables','two_view_reliability','spending_shape','fixed_bill_margins')}}


def shape_predictions(raw, group, y, t, q):
    # The original protocol constructs and caches all folds inside one function.
    x, fees, width=shape.inputs(raw,group)
    direction,_=shape.directions(raw,fees,t,y)
    result=[]
    for rule in (np.zeros(10,int),np.tile(direction[0],2),direction[1:].ravel(),-direction[1:].ravel()):
        model=shape.LGBMClassifier(n_estimators=300,num_leaves=17,min_child_samples=30,learning_rate=.045,
            n_jobs=2,random_state=889,verbosity=-1,monotone_constraints=np.r_[np.zeros(width,int),rule].tolist(),
            monotone_constraints_method='advanced')
        model.fit(x[t],y[t])
        result.append(model.booster_.predict(x[q],num_threads=2))
    return np.column_stack(result)


def predict(family,raw,group,y,t,q):
    if family=='response_channels':
        return response.fit(response.field_blocks(raw,group),y,group,t,q)
    if family=='broad_error_transport_projection':
        return projection.predict(projection.design(raw),y,group,t,q)
    if family=='predicate_tables':
        return predicate.score(predicate.design(raw),y,group,t,q)
    if family=='two_view_reliability':
        return reliability.predict(reliability.design(raw,group),y,group,t,q)
    if family=='spending_shape':
        return shape_predictions(raw,group,y,t,q)
    if family=='fixed_bill_margins':
        counter,_,_=partitions(raw)
        return margins.predict(margins.matrices(raw,group,counter),y,group,t,q)
    raise ValueError(family)


def run():
    torch.set_num_threads(1)
    raw,y,group,fold=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    assert len(test)==4277 and 'Name' not in test and 'Transported' not in test
    roster=pd.concat([raw,test],ignore_index=True)
    groups=roster.PassengerId.str.split('_').str[0].to_numpy()
    t,q=np.flatnonzero(fold!=0),np.flatnonzero(fold==0)
    fit=np.arange(len(y))
    submit=np.arange(len(raw),len(roster))
    assert not np.isin(groups[fit],groups[submit]).any()
    predictions=[]
    names=[]
    maxdiff=[]
    for family,modes in MODES.items():
        with threadpool_limits(1 if family in ('response_channels','broad_error_transport_projection','predicate_tables') else 2):
            replay=predict(family,raw,group,y,t,q)
            with np.load(ROOT/f'artifacts/nov24_{family}/full_0.npz') as z:
                np.testing.assert_array_equal(z['t'],t)
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['truth' if 'truth' in z else 'y'],y[t])
                assert replay.shape==(len(q),len(modes))
                diff=np.max(np.abs(replay-z['p']),axis=0)
                np.testing.assert_array_equal(replay>=.5,z['p']>=.5)
                np.testing.assert_allclose(replay,z['p'],rtol=0,atol=1e-5)
            print('verified',family,diff.tolist(),flush=True)
            p=predict(family,roster,groups,y,fit,submit)
        assert p.shape==(len(test),len(modes)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        names.extend([f'{family}/{m}' for m in modes])
        predictions.append(p)
        maxdiff.extend(diff)
        print('refit',family,flush=True)
    OUT.parent.mkdir(exist_ok=True)
    tmp=OUT.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,p=np.column_stack(predictions),names=np.asarray(names),
            passenger=test.PassengerId.to_numpy(str),train_passenger=raw.PassengerId.to_numpy(str),
            train_y=y,replay_passenger=raw.PassengerId.iloc[q].to_numpy(str),replay_maxdiff=np.asarray(maxdiff))
    tmp.replace(OUT)
    with np.load(OUT) as z:
        np.testing.assert_array_equal(z['p'],np.column_stack(predictions))
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)


if __name__=='__main__':
    run()
