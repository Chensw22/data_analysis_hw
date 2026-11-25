"""Recover service identity only among baskets with a unique positive maximum."""
import hashlib
import numpy as np
import torch

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import data
from inverse_tabicl_manifest_n26 import OUT,conditional_observation,masked_view
from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer


def run():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,f=data()
    x,attr=masked_view(compact_features(raw),'dominant_service')
    eligible=~np.isin(attr,['no_positive_charge','tied_services'])
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        t,q=np.flatnonzero((f!=k)&eligible),np.flatnonzero((f==k)&eligible)
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'{k}_paid_identity.npz'
        if not path.exists():
            prior=infer(x,y,t,q,891+k)
            values={}
            for sham,name in [(False,'observed'),(True,'shuffled')]:
                values[name],levels=conditional_observation(x,attr,y,t,q,891+k,sham)
            assert len(levels)==5 and np.isfinite(prior).all()
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream,q=q,t=t,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                    prior=prior,levels=levels,checkpoint=DIGEST,**values)
            tmp.replace(path)
        with np.load(path) as z:
            for key,expected in [('t',t),('q',q),('train_y',y[t]),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[key],expected)
        print('unique paid identity',k,len(q),flush=True)


if __name__=='__main__':
    run()
