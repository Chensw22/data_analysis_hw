"""Fit a table foundation model to focal outcome and the remaining party's outcome pattern, then marginalize."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov26_joint_party_tabicl'
BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def features(raw,g):
    x=compact_features(raw)
    members=pd.Series(g)
    size=members.map(members.value_counts()).to_numpy()
    x['additional_members']=size-1
    for name,val in [('age',raw.Age),('sleep',raw.CryoSleep.map({True:1.,False:0.})),
                     *[(c,np.log1p(raw[c])) for c in BILLS]]:
        v=pd.Series(val).astype(float)
        count=v.notna().groupby(g).transform('sum')-v.notna().astype(int)
        other=v.fillna(0).groupby(g).transform('sum')-v.fillna(0)
        x['other_'+name]=other/count.replace(0,np.nan)
    return x,size


def outcome_patterns(y,t,g,size):
    visible=pd.Series(g[t]).groupby(g[t]).transform('size').to_numpy()
    assert np.array_equal(visible,size[t])
    rest=pd.Series(y[t].astype(int)).groupby(g[t]).transform('sum').to_numpy()-y[t]
    label=np.where(size[t]==1,0,np.where(rest==0,1,np.where(rest==size[t]-1,2,3)))
    return label


def model(seed):
    from tabicl import TabICLClassifier
    from tabicl_routed_context_n23 import WEIGHT
    return TabICLClassifier(n_estimators=2,model_path=WEIGHT,allow_auto_download=False,device='cpu',batch_size=1,
                            n_jobs=2,random_state=seed,kv_cache=False,use_amp=False,use_fa3=False)


def joint(x,target,t,q,size,seed,return_table=False):
    fit=model(seed).fit(x.iloc[t],target)
    proba=fit.predict_proba(x.iloc[q])
    labels=fit.classes_.astype(int)
    pattern=labels//2
    allowed=np.where(size[q,None]==1,pattern==0,(pattern>0)&((size[q,None]>2)|(pattern!=3)))
    weight=proba*allowed
    assert (weight.sum(1)>0).all()
    weight/=weight.sum(1,keepdims=True)
    if return_table:
        table=np.zeros((len(q),8))
        table[:,labels]=weight
        return table.reshape(len(q),4,2)
    return weight[:,labels%2==1].sum(1)


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,f=data()
    frame,size=features(raw,g)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    OUT.mkdir(exist_ok=True)
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            loc=fold==k; q=ids[loc]; other=ids[~loc]; tr=other[~np.isin(g[other],g[q])]
            present=pd.Series(g[tr]).groupby(g[tr]).transform('size').to_numpy()
            t=tr[present==size[tr]]
            assert not np.isin(g[t],g[q]).any()
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                seed=4605+int(k)
                pattern=outcome_patterns(y,t,g,size)
                fixed=pattern.copy()
                rng=np.random.default_rng(seed)
                strata=2*size[t]+y[t]
                for level in np.unique(strata):
                    sel=np.flatnonzero(strata==level)
                    fixed[sel]=rng.permutation(pattern[sel])
                direct=infer(frame,y,t,q,seed)
                before=joint(frame,2*fixed+y[t],t,q,size,seed)
                actual=joint(frame,2*pattern+y[t],t,q,size,seed)
                with path.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                        direct=direct,shuffled=before,joint=actual,checkpoint=DIGEST)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,val in [('t',t),('q',q),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q]),('checkpoint',DIGEST)]:
                    np.testing.assert_array_equal(z[name],val)
            print(tag,k,len(t),len(q),flush=True)


def score():
    from disjoint_tree_fields_n25 import ART,DEST,stats
    from family_identity_transfer_n26 import panels_and_reference
    from outstanding_error_families_n26 import observed
    raw,y,g,f=data()
    _,size=features(raw,g)
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    cur=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    result={}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,cur)]:
        clean=~np.isin(g[ids],g[rows]); votes={}
        for k in np.unique(fold):
            loc=fold==k;q=ids[loc]
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[ids[~loc]])
            with np.load(OUT/f'{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                for name in ('direct','shuffled','joint'):
                    votes.setdefault(name,np.zeros(len(ids)))[loc]=z[name]
        anchor=view['anchor']; known=view['known']; wrong=anchor!=y[ids]
        family=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        table={}
        for name,v in votes.items():
            label=v>=.5
            methods={'direct':label!=anchor,'induced':(label!=anchor)&(label!=(votes['direct']>=.5)),
                     'quarter':(anchor==(base>=.5))&((.75*base+.25*v>=.5)!=anchor)}
            for action,chg in methods.items():
                delta=chg*np.where(wrong,1,-1)
                table[name+'_'+action]={s:stats(delta,m,fold,g[ids]) for s,m in [('all',known),('solo',known&(size[ids]==1)),
                    ('parties',known&(size[ids]>1)),('new',clean&known),('families',family)]}
        result[tag]=table
        print(tag,[(n,s['all']['fixed'],s['all']['harmed']) for n,s in table.items()],flush=True)
    DEST.with_name('joint_party_tabicl.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','score'])
    {'train':train,'score':score}[parser.parse_args().action]()
