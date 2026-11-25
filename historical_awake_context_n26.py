"""Replay the awake TabICL candidate on historical folds with all held bookings excluded from context."""
import hashlib
import numpy as np
import pandas as pd
import torch

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data
from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer

OUT=ROOT/'artifacts/nov26_historical_awake'


def run():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    features=compact_features(pd.concat([raw,test],ignore_index=True))
    awake=raw.CryoSleep.eq(False).to_numpy()
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,fold=z['rows'],z['fold']
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows])
        np.testing.assert_array_equal(z['y'],y[rows])
    OUT.mkdir(exist_ok=True)
    for k in np.unique(fold):
        held=rows[fold==k]
        q=held[awake[held]]
        t=rows[(fold!=k)&awake[rows]&~np.isin(g[rows],g[held])]
        assert not np.isin(g[t],g[held]).any()
        for seed in (891+k,2991+k):
            path=OUT/f'{k}_{seed}.npz'
            if not path.exists():
                p=infer(features,y,t,q,int(seed))
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    np.savez_compressed(stream,q=q,t=t,train_y=y[t],seed=seed,checkpoint=DIGEST,
                                        passenger=raw.PassengerId.iloc[q].to_numpy(str),p=p)
                tmp.replace(path)
            with np.load(path) as z:
                for key,value in [('q',q),('t',t),('train_y',y[t]),('seed',seed),('checkpoint',DIGEST),
                                  ('passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],value)
                assert np.isfinite(z['p']).all() and ((z['p']>=0)&(z['p']<=1)).all()
            print('historical awake',k,int(seed),len(t),len(q),flush=True)


if __name__=='__main__':
    run()
