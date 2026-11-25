"""Recalibrate the frozen release with three disjoint sets of labeled bookings."""
import hashlib
import numpy as np
import pandas as pd
from nested_release_review_n26 import OUT as NESTED, ROOT, matrices, protocols, sources

OUT=ROOT/'artifacts/nov26_replicate_release_calibration'


def run():
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST,WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,panels=sources()
    OUT.mkdir(exist_ok=True)
    for key,src,dst,t,q in protocols(panels,g,include_full=True):
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        with np.load(NESTED/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],a)
            np.testing.assert_array_equal(z['outer'],b)
            fold=z['fold']
        path=OUT/f'{key}.npz'
        if not path.exists():
            train=np.full((len(t),3,2),np.nan)
            query=np.full((len(q),3,2),np.nan)
            for k in range(3):
                take,held=t[fold==k],t[fold!=k]
                assert not np.isin(g[src['ids'][take]],g[src['ids'][held]]).any()
                xt,xc=matrices(raw,src,src,take,held)
                xt2,xq=matrices(raw,src,dst,take,q)
                pd.testing.assert_frame_equal(xt[1],xt2[1])
                ask=pd.concat([xc[1],xq[1]],ignore_index=True)
                anchor=np.r_[src['anchor'][held],dst['anchor'][q]]
                for j,labels in enumerate([src['wrong'][take],y[src['ids'][take]]]):
                    fit=model(7601).fit(xt[1],labels)
                    np.testing.assert_array_equal(fit.classes_,[False,True])
                    prob=fit.predict_proba(ask)[:,1]
                    risk=np.where(anchor,1-prob,prob) if j==1 else prob
                    train[fold!=k,k,j]=risk[:len(held)]
                    query[:,k,j]=risk[len(held):]
            assert np.isfinite(query).all()
            np.testing.assert_array_equal(np.isfinite(train[:,:,0]),fold[:,None]!=np.arange(3))
            np.testing.assert_array_equal(np.isfinite(train[:,:,1]),np.isfinite(train[:,:,0]))
            with path.with_suffix('.tmp').open('wb') as h:
                np.savez_compressed(h,ids=a,outer=b,train_y=y[a],fold=fold,train=train,query=query,checkpoint=DIGEST)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for name,val in [('ids',a),('outer',b),('train_y',y[a]),('fold',fold),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[name],val)
        print(key,flush=True)


if __name__=='__main__':
    run()
