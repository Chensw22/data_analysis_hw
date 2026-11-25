"""Obtain out-of-booking parent decisions inside each calibration set without using evaluation labels."""
import hashlib

import numpy as np
from sklearn.model_selection import GroupKFold
from tabicl_consensus_review_n26 import ROOT, OUT as REVIEW, matrices, sources, splits

OUT=ROOT/'artifacts/nov26_nested_release_review'
REPORT=ROOT/'reports/nov23_matched_learning_curve/nested_release_review.json'


def protocols(panels,g):
    for item in splits(panels,g):
        if item[0].startswith('bookingmix') or item[0]=='historical_to_new':
            yield item


def train():
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST,WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,panels=sources()
    OUT.mkdir(exist_ok=True)
    for key,src,dst,t,q in protocols(panels,g):
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        filename=OUT/f'{key}.npz'
        if not filename.exists():
            risk=np.full((len(t),2),np.nan)
            fold=np.full(len(t),-1)
            inner=GroupKFold(3,shuffle=True,random_state=8617)
            for k,(support,held) in enumerate(inner.split(a,groups=g[a])):
                assert not np.isin(g[a[support]],g[a[held]]).any()
                xt,xq=matrices(raw,src,src,t[support],t[held])
                for j,target in enumerate([src['wrong'][t[support]],y[a[support]]]):
                    fitted=model(7601).fit(xt[1],target)
                    np.testing.assert_array_equal(fitted.classes_,[False,True])
                    pr=fitted.predict_proba(xq[1])[:,1]
                    risk[held,j]=np.where(src['anchor'][t[held]],1-pr,pr) if j==1 else pr
                fold[held]=k
            assert np.isfinite(risk).all() and (fold>=0).all()
            with filename.with_suffix('.tmp').open('wb') as fp:
                np.savez_compressed(fp,ids=a,outer=b,fold=fold,risk=risk,checkpoint=DIGEST)
            filename.with_suffix('.tmp').replace(filename)
        with np.load(filename) as z:
            for name,value in [('ids',a),('outer',b),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[name],value)
        print(key,flush=True)


if __name__=='__main__':
    train()
