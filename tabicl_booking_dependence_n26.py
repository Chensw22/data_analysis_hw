"""Extract only the dependency of focal and companion outcomes learned by TabICL."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT,data
from joint_party_tabicl_n26 import OUT,features,joint,outcome_patterns


def train():
    import torch
    from tabicl_routed_context_n23 import WEIGHT,DIGEST
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,f=data()
    frame,size=features(raw,g)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(raw)),f)]:
        for k in np.unique(fold):
            with np.load(OUT/f'{tag}_{k}.npz') as z:
                t,q=z['t'],z['q']
                np.testing.assert_array_equal(q,ids[fold==k])
                np.testing.assert_array_equal(z['train_y'],y[t])
                marg={name:z[name] for name in ['joint','shuffled']}
            pool=np.flatnonzero(np.isin(g,g[q]))
            assert not np.isin(g[t],g[pool]).any()
            path=OUT/f'table_{tag}_{k}.npz'
            if not path.exists():
                seed=4605+int(k)
                pattern=outcome_patterns(y,t,g,size)
                shuffled=pattern.copy()
                rng=np.random.default_rng(seed)
                strata=2*size[t]+y[t]
                for level in np.unique(strata):
                    ix=np.flatnonzero(strata==level);shuffled[ix]=rng.permutation(pattern[ix])
                archive={}
                for name,labels in [('joint',pattern),('shuffled',shuffled)]:
                    table=joint(frame,2*labels+y[t],t,pool,size,seed,return_table=True)
                    np.testing.assert_allclose(table.sum((1,2)),1.,atol=2e-6)
                    lookup=np.searchsorted(pool,q)
                    np.testing.assert_array_equal(pool[lookup],q)
                    # Some historical predictions are now batched with additional
                    # group members. Check that this did not change their answers.
                    np.testing.assert_allclose(table[lookup,:,1].sum(1),marg[name],atol=2e-5,rtol=0)
                    archive[name]=table
                with path.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,pool=pool,train_y=y[t],checkpoint=DIGEST,**archive)
                path.with_suffix('.tmp').replace(path)
            print(tag,k,flush=True)


def score():
    from anchored_booking_distribution_n26 import update
    from disjoint_tree_fields_n25 import ART,DEST,stats
    from family_identity_transfer_n26 import panels_and_reference
    from outstanding_error_families_n26 import observed
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0]);ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    current=observed(np.arange(len(raw)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    result={}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(raw)),f,native,current)]:
        fresh=~np.isin(g[ids],g[rows]); preds={}
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k);q=ids[loc]
            if tag=='historical':
                fresh[loc]=~np.isin(g[q],g[ids[fold!=k]])
            with np.load(OUT/f'table_{tag}_{k}.npz') as z:
                pool=z['pool'];np.testing.assert_array_equal(z['q'],q)
                tables={name:z[name] for name in ('joint','shuffled')}
            lookup=np.full(len(raw),-1,int);lookup[q]=loc
            for ix in pd.Series(g[pool]).groupby(g[pool],sort=False).indices.values():
                if len(ix)<2:
                    continue
                n=len(ix); pos=lookup[pool[ix]]; has=pos>=0; target=pos[has]
                bits=(np.arange(1<<n)[:,None]>>np.arange(n))&1
                peer_count=bits.sum(1,keepdims=True)-bits
                status=np.where(peer_count==0,1,np.where(peer_count==n-1,2,3))
                for name,all_tables in tables.items():
                    table=all_tables[ix]
                    y_marg=table.sum(1)
                    neighbor_marg=table.sum(2)
                    independent=neighbor_marg[:,:,None]*y_marg[:,None,:]
                    log_ratio=np.log(table.clip(1e-9))-np.log(independent.clip(1e-9))
                    ratio=log_ratio[np.arange(n)[None,:],status,bits].mean(1)
                    prior=y_marg[:,1].copy()
                    prior[has]=base[target]
                    for strength in [.5,1.]:
                        key=f'{name}_{strength}'
                        probs=update(bits,prior,ratio,strength)
                        preds.setdefault(key,base.copy())[target]=probs[has]
        anchor=view['anchor']; wrong=anchor!=y[ids]
        family=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        metrics={}
        for name,pr in preds.items():
            flag=(anchor==(base>=.5))&((pr>=.5)!=anchor)
            delta=flag*np.where(wrong,1,-1)
            metrics[name]={n:stats(delta,m,fold,g[ids]) for n,m in [('all',view['known']),('new',view['known']&fresh),('families',family)]}
        result[tag]=metrics
        print(tag,[(key,v['all']['fixed'],v['all']['harmed']) for key,v in metrics.items()],flush=True)
    DEST.with_name('tabicl_booking_dependence.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','score'])
    {'train':train,'score':score}[parser.parse_args().action]()
