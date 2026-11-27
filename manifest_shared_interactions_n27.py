"""Compare low-rank and explicit manifest interactions on entire held-out bookings."""
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from export_budget_quarter_n23 import source
from manifest_tree_mixture_n27 import discretize
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART,cohort,count
from unanimous_error_reach_n27 import ROOT,inputs


def indicators(fields):
    widths=fields.max(0)+1
    indices=(fields+np.r_[0,np.cumsum(widths)[:-1]]).ravel()
    return csr_matrix((np.ones(len(indices)),indices,np.arange(0,len(indices)+1,fields.shape[1])),
                      shape=(len(fields),sum(widths)))


def factor(x,y,q):
    # One vector per observed field level. A pair's log-odds contribution is
    # the dot product of its vectors; related contexts can share evidence.
    w=x.shape[1]; rank=5
    rng=np.random.default_rng(512)
    initial=np.r_[0.,np.zeros(w),rng.normal(0,.04,w*rank)]
    def loss(param):
        bias=param[0]; main=param[1:w+1]; v=param[w+1:].reshape(w,rank)
        proj=x@v
        z=bias+x@main+.5*((proj*proj).sum(1)-x@(v*v).sum(1))
        err=expit(z)-y
        l=np.logaddexp(0,z).sum()-z@y+1.5*(main@main)+4.*np.sum(v*v)
        col=x.T@err
        grad=np.r_[err.sum(),col+3*main,(x.T@(err[:,None]*proj)-col[:,None]*v+8*v).ravel()]
        return l,grad
    counts=np.asarray(x.sum(axis=0)).ravel()
    scale=np.r_[np.sqrt(len(y)),np.sqrt(counts+3),np.repeat(np.sqrt(.07*counts+8),rank)]
    def conditioned(w):
        value,gradient=loss(w/scale)
        return value,gradient/scale
    fit=minimize(conditioned,initial*scale,jac=True,method='L-BFGS-B',options={'maxiter':2200,'ftol':1e-6})
    if not fit.success:
        raise RuntimeError(fit.message)
    fit.x=fit.x/scale
    main=fit.x[1:w+1]; v=fit.x[w+1:].reshape(w,rank)
    return expit(fit.x[0]+q@main+.5*(((q@v)**2).sum(1)-q@(v*v).sum(1)))


def run():
    raw,frame,y,groups,fold,first,directional,_,opinions,_=inputs()
    n=len(y)
    _,_,known,fresh=cohort(raw,y,fold)
    anchor=first^directional
    _,official,receipt=source('proposal_n27_directional_foundation',frame.iloc[n:])
    np.testing.assert_array_equal(anchor[n:],official)
    archive=ART/'nov27_shared_interactions'
    archive.mkdir(exist_ok=True)
    f=discretize(frame)
    x=indicators(f)
    pair_fields=np.column_stack([f[:,i]*(int(f[:,j].max())+1)+f[:,j] for i,j in combinations(range(f.shape[1]),2)])
    from scipy.sparse import hstack
    pairs=hstack((x,indicators(pair_fields))).tocsr()
    g=frame.PassengerId.str.split('_').str[0].to_numpy()
    seq=np.empty(n,int)
    for i,b in enumerate(np.array_split(np.unique(groups),8)):seq[np.isin(groups,b)]=i
    common=((opinions>=.5)==anchor[:,None]).all(1)
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    awake=common&paid&frame.CryoSleep.eq(False).to_numpy()
    sleep=common&~paid&frame.CryoSleep.eq(True).to_numpy()
    scopes=dict(all=known,fresh=known&fresh,unanimous=known&common[:n],
                awake_paid=known&awake[:n],sleep_zero=known&sleep[:n])
    wrong=anchor[:n]!=y
    assert sum(scopes['awake_paid']&wrong)==476 and sum(scopes['sleep_zero']&wrong)==300
    tables={}; modes=['additive','explicit_pairs','shared_factors']
    signed=np.where(anchor,1,-1)*np.abs(logit(np.median(opinions[:,1:4],axis=1).clip(.003,.997)))
    report={'official_directional_hash':receipt}
    for key,folds in [('random',fold),('ordered',seq)]:
        scores=np.empty((len(frame),3))
        for i in range(9):
            t=np.flatnonzero(folds!=i); q=np.flatnonzero(folds==i) if i<8 else np.arange(n,len(frame))
            assert not np.isin(g[t],g[q]).any()
            if i==8 and 'random' in tables:
                scores[q]=tables['random'][q]
                continue
            for j,z in enumerate((x,pairs)):
                lr=LogisticRegression(C=(1/3,1/24)[j],max_iter=450,tol=1e-6)
                scores[q,j]=lr.fit(z[t],y[t]).predict_proba(z[q])[:,1]
            scores[q,2]=factor(x[t],y[t],x[q])
        assert np.isfinite(scores).all()
        tables[key]=scores
        rules=dict(shared=(scores[:,2]>=.5)!=anchor)
        rules['explicit']=(scores[:,1]>=.5)!=anchor
        rules['unique_to_shared']=rules['shared']&~rules['explicit']&((scores[:,0]>=.5)==anchor)
        delta=logit(scores[:,2].clip(.003,.997))-logit(scores[:,1].clip(.003,.997))
        rules['increment_over_pairs']=rules['shared']&((signed+delta>=0)!=anchor)
        # A deployable policy aims at both broad populations; it changes no other class.
        rules['both_populations_increment']=rules['increment_over_pairs']&(awake|sleep)
        results={}
        for mode,flag in rules.items():
            results[mode]={scope:count(flag[:n],rows,wrong) for scope,rows in scopes.items()}
            results[mode]['test']=int(flag[n:].sum())
            tables[key+'_'+mode]=flag
        report[key]=dict(scores={name:dict(accuracy=float(np.mean((scores[:n,j]>=.5)==y)),
                            log_loss=float(log_loss(y,scores[:n,j]))) for j,name in enumerate(modes)},
                         rules=results)
        print(key,report[key]['scores'],results['both_populations_increment'],flush=True)
    report['repeat']={mode:{scope:count((tables['random_'+mode]&tables['ordered_'+mode])[:n],rows,wrong)
                          for scope,rows in scopes.items()} for mode in rules}
    np.savez_compressed(archive/'predictions.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=fold,ordered_fold=seq,**tables)
    (ROOT/'reports/nov23_matched_learning_curve/manifest_shared_interactions.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):run()
