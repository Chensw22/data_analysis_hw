"""Measure what naturally occurring missing-field patterns do to the same fully observed people."""
import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits

from broad_adversarial_field_views_n23 import REPORT, objective
from broad_geographic_transfer_n23 import BILLS, CATEGORICAL, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART,SEEDS,write

FIELDS=[['HomePlanet'],['CryoSleep'],['Destination'],['Age'],['VIP'],*[[c] for c in BILLS],
        ['Deck','CabinNumber','Side']]


def rows(frame):
    blocks=[]
    miss=[]
    for fields in FIELDS:
        block=frame[fields].copy()
        for name in fields:
            if name in CATEGORICAL:
                block[name]=block[name].replace('__MISSING__',np.nan)
            if name in BILLS:
                block[name]=np.log1p(block[name])
        unavailable=block.isna().all(axis=1).to_numpy()
        observed=pd.get_dummies(block, columns=[c for c in fields if c in CATEGORICAL]).to_numpy(dtype=float,copy=True)
        observed[unavailable]=np.nan
        blocks.append(observed)
        miss.append(unavailable)
    return blocks,np.column_stack(miss)


def masked_rows(blocks, booking_size, missing, t, ids, seed):
    rng=np.random.default_rng(seed)
    donors=t[missing[t].any(1)]
    assert len(donors)
    samples=rng.choice(donors,size=(4,len(ids)))
    views=[]
    for pattern in [np.zeros_like(missing[ids]),*missing[samples]]:
        cols=[]
        for j,block in enumerate(blocks):
            values=block[ids].copy()
            values[pattern[:,j]]=np.nan
            cols.append(values)
        views.append(np.column_stack([*cols,booking_size[ids]]))
    return np.stack(views)


def fit(train,query,labels):
    params=dict(max_depth=4,eta=.06,reg_lambda=4.,min_child_weight=6,tree_method='hist',nthread=2,
                seed=913,base_score=.5,objective='binary:logistic')
    query_matrix=xgb.DMatrix(np.concatenate(query),nthread=2)
    observed=xgb.DMatrix(train[0],label=labels,nthread=2)
    augmented=xgb.DMatrix(np.concatenate(train),label=np.tile(labels,len(train)),nthread=2)
    scores={}
    for mode in ['clean','mean','worst']:
        if mode=='clean':
            learner=xgb.train(params,observed,num_boost_round=275)
        else:
            def update(raw,dm):
                g,h=objective(raw.reshape(len(train),len(labels)),labels,mode,None)
                return g.flatten(),h.clip(.00001).flatten()
            learner=xgb.train(params,augmented,num_boost_round=275,obj=update)
        scores[mode]=learner.predict(query_matrix).reshape(len(query),query.shape[1]).T
    return scores


def main():
    row,y,frame,groups,_=load()
    blocks,missing=rows(frame)
    incomplete=missing.any(1)
    size=frame.BookingSize.to_numpy()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        wrong=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    scopes={'complete':~incomplete,'incomplete':incomplete,'joint_wrong':wrong,'joint_right':right,
            'complete_joint_wrong':wrong&~incomplete,'incomplete_joint_wrong':wrong&incomplete}
    plans,hist={},{}
    out=ART/'nov23_observed_mask_shift'
    out.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        score={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            train=masked_rows(blocks,size,missing,t,t,275+int(f))
            query=masked_rows(blocks,size,missing,t,q,838+int(f))
            for k,p in fit(train,query,y[t]).items():
                score.setdefault(k,np.full((len(y),5),np.nan))[q]=p
        votes={'reference':ref}
        perturb={}
        for k,p in score.items():
            assert np.isfinite(p).all()
            votes[k]=p[:,0]>=.5
            votes['quarter_'+k]=np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*p[:,0]>=.5,ref)
            clean_hit=votes[k]==y
            corrupt=(p[:,1:]>=.5)==y[:,None]
            perturb[k]={k2:dict(people=int(m.sum()),mean_fixes=float((corrupt[m]&~clean_hit[m,None]).sum()/4),
                        mean_harms=float((~corrupt[m]&clean_hit[m,None]).sum()/4),
                        clean_accuracy=float(clean_hit[m].mean()),corrupt_accuracy=float(corrupt[m].mean()))
                        for k2,m in [('complete',~incomplete),('complete_joint_wrong',wrong&~incomplete)]}
        votes['by_missingness']=np.where(incomplete,votes['quarter_worst'],ref)
        h={k:v==y for k,v in votes.items()}
        if seed!='v131':
            for k,v in h.items():hist.setdefault(k,[]).append(v)
        plans[seed]=dict(changes={k:{name:paired(v[m],ref[m]==y[m]) for name,m in
                      {'all':np.ones(len(y),bool),**scopes}.items()} for k,v in h.items()},masking=perturb)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=op[:,0],ref=ref,**score)
        print(seed,plans[seed]['changes']['by_missingness']['all'],perturb['clean']['complete'],flush=True)
    h={k:np.stack(v) for k,v in hist.items()}
    stable={k:{name:int((v.all(0)&m).sum()) for name,m in scopes.items()} for k,v in h.items()}
    write(REPORT/'observed_masks.json',dict(fields=FIELDS,cohorts={k:int(m.sum()) for k,m in scopes.items()},
                                            plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):main()
