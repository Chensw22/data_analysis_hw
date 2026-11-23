"""Train against each passenger's most misleading field omissions and evaluate on intact manifests."""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit, logsumexp, softmax
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, CATEGORICAL, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT/'reports/nov23_field_robustness'
OUT = ART/'nov23_field_robustness'
BLOCKS = [BILLS, ['CabinNumber','Deck','Side'], ['CryoSleep'], ['Age','VIP','HomePlanet','Destination','BookingSize']]
TEMP = .35


def views(frame):
    numeric = pd.get_dummies(frame, columns=CATEGORICAL).astype(float)
    for field in BILLS:
        numeric[field] = np.log1p(numeric[field])
    variants = [numeric.to_numpy()]
    for block in BLOCKS:
        removed = [c for c in numeric if any(c==k or c.startswith(k+'_') for k in block)]
        assert removed
        masked = numeric.copy()
        masked[removed] = np.nan
        variants.append(masked.to_numpy())
    return np.stack(variants)


def objective(z, target, mode, permutation):
    p = expit(z)
    diff = p-target
    loss = np.logaddexp(0,z)-target*z
    alpha = softmax(loss[1:]/TEMP, axis=0)
    if mode=='mean':
        alpha.fill(1/len(alpha))
    elif mode=='shuffled':
        alpha = alpha[:,permutation]
    w = np.vstack([np.full_like(z[0], .5),.5*alpha])
    grad = w*diff
    hessian = w*p*(1-p)
    if mode=='worst':
        hessian[1:] += .5*alpha*(1-alpha)*diff[1:]**2/TEMP
    return grad, hessian


def verify_gradient():
    z = np.random.default_rng(524).normal(size=(5,9))
    y = np.arange(9)%2
    def scalar(a):
        loss = np.logaddexp(0,a)-y*a
        return .5*(loss[0].sum()+TEMP*logsumexp(loss[1:]/TEMP,axis=0).sum())
    jac, diag = objective(z,y,'worst',None)
    h = 1e-4
    for coord in [(0,0),(1,3),(3,1),(4,8)]:
        dx = np.zeros_like(z)
        dx[coord] = h
        np.testing.assert_allclose(jac[coord],(scalar(z+dx)-scalar(z-dx))/(2*h),atol=1e-7)
        np.testing.assert_allclose(diag[coord],(scalar(z+dx)+scalar(z-dx)-2*scalar(z))/h**2,atol=2e-6)


def fold_predictions(x, y, t, q):
    params = dict(max_depth=4, eta=.06, reg_lambda=4., min_child_weight=6,
                  tree_method='hist', nthread=2, seed=913, base_score=.5, objective='binary:logistic')
    original = xgb.DMatrix(x[0,t],label=y[t],nthread=2)
    aug = xgb.DMatrix(np.concatenate(x[:,t]),label=np.tile(y[t],len(x)),nthread=2)
    query = xgb.DMatrix(np.concatenate(x[:,q]),nthread=2)
    rng = np.random.default_rng(594)
    perm = np.arange(len(t))
    for label in [False, True]:
        cell = np.flatnonzero(y[t]==label)
        perm[cell] = rng.permutation(cell)
    scores = {}
    for mode in ['clean','mean','worst','shuffled']:
        if mode=='clean':
            fitted=xgb.train(params,original,num_boost_round=275)
        else:
            def update(raw, dm):
                z=raw.reshape(len(x),len(t))
                grad,hess=objective(z,y[t],mode,perm)
                return grad.flatten(),hess.clip(.00001).flatten()
            fitted=xgb.train(params,aug,num_boost_round=275,obj=update)
        scores[mode]=fitted.predict(query).reshape(len(x),len(q)).T
    return scores


def main():
    verify_gradient()
    row,y,frame,groups,_=load()
    x=views(frame)
    OUT.mkdir(exist_ok=True)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        release=z['votes']
    wrong=(release!=y[None,:,None]).all((0,2))
    right=(release==y[None,:,None]).all((0,2))
    assert wrong.sum()==983 and right.sum()==5343
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            for k,p in fold_predictions(x,y,t,q).items():
                scores.setdefault(k,np.full((len(y),len(x)),np.nan))[q]=p
        votes={'reference':ref}
        needs=ref==(op[:,0]>=.5)
        diagnostics={}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            intact=p[:,0]
            votes[k]=intact>=.5
            votes['quarter_'+k]=np.where(needs,.75*op[:,0]+.25*intact>=.5,ref)
            votes['offset_'+k]=np.where(needs,
                    logit(op[:,0].clip(.002,.998))+logit(intact.clip(.002,.998))
                    -logit(scores['clean'][:,0].clip(.002,.998))>=0,ref)
            gold=np.where(y[:,None],p,1-p)
            diagnostics[k]=dict(intact_logloss=float(-np.log(gold[:,0].clip(1e-8)).mean()),
                    worst_removed_logloss=float(-np.log(gold[:,1:].min(1).clip(1e-8)).mean()),
                    accuracy_by_view=((p>=.5)==y[:,None]).mean(0).tolist())
        hit={k:v==y for k,v in votes.items()}
        if seed!='v131':
            for k,v in hit.items():
                hist.setdefault(k,[]).append(v)
        plans[seed]=dict(models={k:dict(reference=paired(v,ref==y),clean=paired(v,hit['clean']))
                            for k,v in hit.items()},missing=diagnostics)
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],**scores)
        print(seed,plans[seed]['models']['worst'],plans[seed]['models']['quarter_worst'],flush=True)
    h={k:np.stack(v) for k,v in hist.items()}
    misses=(~h['reference']).sum(0)
    stats={k:dict(joint_wrong=int((v.all(0)&wrong).sum()),joint_right=int((v.all(0)&right).sum()),
                 stable_by_miss=[int((v.all(0)&(misses==i)).sum()) for i in range(4)],
                 gain_ci95_pp=interval((v.astype(float)-h['reference']).mean(0),groups)) for k,v in h.items()}
    write(REPORT/'masked.json',dict(blocks=BLOCKS,plans=plans,grouped=stats))


if __name__=='__main__':
    with threadpool_limits(2):main()
