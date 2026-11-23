"""Refit the full-test expert recipes without access to labels in v131 validation bookings."""
import argparse
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

from specialist_full_manifest_n23 import ROOT, data, fit_member

OUT = ROOT/'artifacts/nov23_specialist_v131_transfer'


def run(name):
    raw,group,size,y,_,_=data()
    with np.load(ROOT/'artifacts/nov23_incumbent_budget_transfer/oof.npz') as z:
        row,fold,target=z['row'],z['fold'],z['y']
    np.testing.assert_array_equal(target,y[row])
    OUT.mkdir(exist_ok=True)
    p=np.full(len(y[row]),np.nan)
    info={}
    for f in np.unique(fold):
        t=row[(fold!=f)&~np.isin(group[row],group[row[fold==f]])]
        q=row[fold==f]
        assert not np.intersect1d(group[t],group[q]).size
        # Match the exact fixed configuration of the already exported official component.
        start=time.monotonic()
        pred,extra=fit_member(name,raw,group,size,y,t,q)
        info[f'train_{f}']=t
        info[f'query_{f}']=q
        info[f'seconds_{f}']=time.monotonic()-start
        for k,value in extra.items():
            info[f'{k}_{f}']=value
        p[fold==f]=pred
        print(name,int(f),len(t),flush=True)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.savez_compressed(OUT/f'{name}.npz',row=row,y=y[row],fold=fold,p=p,**info)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('member')
    with threadpool_limits(2):
        run(parser.parse_args().member)
