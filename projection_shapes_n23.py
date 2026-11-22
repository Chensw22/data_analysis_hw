"""Class-sensitive directions plus learned univariate response shapes on unseen bookings."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from awake_negative_recovery import BILLS, matrices
from error_overlap_diagnostic import load_authenticated
from exact_fold_error_selector import counts
from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"


def supervised_directions(x,y):
    # Rank directions by changes in within-class dispersion relative to pooled
    # covariance. Both ends matter: either outcome can be more dispersed.
    pos,neg=x[y],x[~y]
    c1,c0=np.cov(pos.T),np.cov(neg.T)
    regularizer=(c1+c0)/2+.3*np.eye(x.shape[1])
    value,vector=eigh(c1-c0,regularizer)
    axes=vector[:,np.argsort(-np.abs(value))[:6]]
    linear=np.linalg.solve(regularizer,pos.mean(0)-neg.mean(0))
    return np.column_stack([linear,axes])


def smooth_basis(t,q):
    # All knots and scales are estimated on the outer training fold.
    scale=StandardScaler().fit(t)
    t,q=[np.clip(scale.transform(part),-5,5) for part in (t,q)]
    index=np.ptp(t,axis=0)>0
    # Binary variables already have an exact linear representation.
    index &= np.sum(np.diff(np.sort(t,axis=0),axis=0)>1e-5,axis=0)>4
    knots=np.quantile(t[:,index],[.15,.35,.55,.75,.9],axis=0)
    def basis(x):
        h=np.maximum(x[:,None,index]-knots,0)
        return np.column_stack([x,h.reshape(len(x),-1)*.5])
    return basis(t),basis(q)


def fit(x,y,masks,t,q):
    scale=StandardScaler().fit(x[t])
    z=np.clip(scale.transform(x),-5,5)
    original_t,original_q=smooth_basis(z[t],z[q])
    blocks_t,blocks_q=[original_t],[original_q]
    for mask in masks:
        fit_rows=t[mask[t]]
        if min(y[fit_rows].sum(),(~y[fit_rows]).sum())<35:
            continue
        dirs=supervised_directions(z[fit_rows],y[fit_rows])
        train_view,query_view=smooth_basis(z[t]@dirs,z[q]@dirs)
        blocks_t.append(.55*train_view*mask[t,None])
        blocks_q.append(.55*query_view*mask[q,None])
    output={}
    for label,xt,xq in [('single_shapes',original_t,original_q),
                        ('projection_shapes',np.column_stack(blocks_t),np.column_stack(blocks_q))]:
        model=LogisticRegression(C=.18,max_iter=500,tol=2e-5)
        model.fit(xt,y[t])
        output[label]=model.predict_proba(xq)[:,1]
    return output


def run():
    inp,_,_,_,_=load_authenticated()
    dev,y,groups=inp['dev'],inp['y'].astype(bool),np.asarray(inp['groups'])
    x=inp['features'].iloc[dev]
    tickets=pd.concat([pd.read_csv(ROOT/f'data/raw/{p}.csv',usecols=['PassengerId']).PassengerId
                       for p in ['train','test']],ignore_index=True).str.split('_').str[0]
    size=tickets.map(tickets.value_counts()).to_numpy()[dev]
    bills=x[BILLS].fillna(0)
    total=bills.sum(axis=1).to_numpy()
    raw=np.column_stack([matrices(x)[0],x.CabinNumber.fillna(900)/1000,x.CabinNumber.isna(),np.log(size),
                         np.log1p(total)/6,bills.div(1+total,axis=0)])
    masks=[np.ones(len(y),bool),total>0,total==0]
    rows=pd.read_csv(ART/'nov23_global_error_inventory.csv')
    np.testing.assert_array_equal(rows.row,dev)
    np.testing.assert_array_equal(rows.target,y)
    stubborn=rows.v131_wrong.to_numpy() & rows.number_new_correct.eq(0).to_numpy()
    seeds=[20261112,20261115,20261116]
    for seed in seeds:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],dev)
            np.testing.assert_array_equal(z['y'],y)
            stubborn &= np.column_stack([(z[k]>=.5)!=y for k in ['base','fixed','learned_twice']]).all(1)
    assert stubborn.sum()==447
    out=ART/'nov23_projection_shapes'
    out.mkdir(exist_ok=True)
    results={}
    for seed in seeds:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as z:
            fold,base=z['fold'],z['base']
        preds={k:np.full(len(y),np.nan) for k in ['single_shapes','projection_shapes']}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(groups[t],groups[q]).size
            for name,p in fit(raw,y,masks,t,q).items():
                preds[name][q]=p
        preds['base']=base
        for name in ['single_shapes','projection_shapes']:
            preds[name+'_blend']=.75*base+.25*preds[name]
        results[str(seed)]={}
        for name,p in preds.items():
            assert np.isfinite(p).all()
            m=counts(p>=.5,base>=.5,y,fold,rows.historical_546.to_numpy(),stubborn)
            m['fixed_of_447']=m.pop('persistent_fixed')
            m['harmed_of_447']=m.pop('persistent_harmed')
            results[str(seed)][name]=m
        np.savez_compressed(out/f'{seed}.npz',row=dev,y=y,fold=fold,**preds)
        print(seed,results[str(seed)],flush=True)
    write(ROOT/'reports/nov23_projection_shapes/results.json',results)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
