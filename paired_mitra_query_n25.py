"""Refit the two half-roster panels, confirming held-out scores before caching test predictions."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'

import numpy as np
import torch

from full_roster_booking_validation_n23 import ROOT,data
from full_roster_mitra_validation_n23 import fit
import mitra_counts_fixed15_online_first as counts


def run():
    if not torch.backends.mps.is_available():
        raise RuntimeError('Mitra training requires Metal in this environment.')
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    counts.verify_reference()
    roster=counts.load_roster()
    raw,y,g,f=data()
    np.testing.assert_array_equal(raw.PassengerId,roster.passenger_ids.iloc[:len(y)])
    test=np.arange(len(y),len(roster.passenger_ids))
    test_ids=roster.passenger_ids.iloc[test].to_numpy(str)
    target=ROOT/'artifacts/nov25_mitra_replica_queries'
    target.mkdir(exist_ok=True)
    for panel,fold,source in [('original',f%2,'nov23_half_roster_disagreement'),
                               ('rotated',(f//2)%2,'nov24_repartition_gate')]:
        for part in (0,1):
            t,q=np.flatnonzero(fold!=part),np.flatnonzero(fold==part)
            assert not np.isin(g[t],g[q]).any()
            with np.load(ROOT/'artifacts'/source/f'mitra_{part}.npz') as reference:
                for k,v in [('train',t),('query',q),('labels',y[t]),('checkpoint',counts.base.WEIGHT_SHA)]:
                    np.testing.assert_array_equal(reference[k],v)
                old=reference['p']
            path=target/f'{panel}_{part}.npz'
            if not path.exists():
                # Replay the original first complete inference chunk with unchanged query order.
                check=q[:2318]
                print('fit',panel,part,'support',len(t),'queries',len(test),flush=True)
                p=fit(roster,t,np.r_[check,test],y[t],counts)
                diff=np.abs(p[:len(check)]-old[:len(check)])
                print('replay',panel,part,'max',diff.max(),'changed',((p[:len(check)]>=.5)!=(old[:len(check)]>=.5)).sum(),flush=True)
                np.testing.assert_allclose(p[:len(check)],old[:len(check)],rtol=0,atol=1e-6)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=t,truth=y[t],q=check,replay=p[:len(check)],ids=test_ids,
                                        prediction=p[len(check):],checkpoint=counts.base.WEIGHT_SHA)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for k,v in [('t',t),('truth',y[t]),('ids',test_ids),('checkpoint',counts.base.WEIGHT_SHA),('q',q[:2318])]:
                    np.testing.assert_array_equal(z[k],v)
                np.testing.assert_allclose(z['replay'],old[:2318],atol=1e-6,rtol=0)
                assert z['prediction'].shape==(len(test),) and np.isfinite(z['prediction']).all()
            print('done',panel,part,flush=True)


if __name__=='__main__':
    run()
