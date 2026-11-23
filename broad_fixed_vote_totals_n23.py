"""Assess which experts dissent after conditioning on exactly how many dissent."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_matched_model_changes_n23 import read_bank
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT=ROOT/'reports/nov23_fixed_vote_totals'


def bank(seed,row,y):
    with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        fold,ref,base=z['fold'],z['ref'],z['base']
        extra=[z[k] for k in ('basket_direct','basket_conditional','sleep_direct','sleep_conditional')]
    models=read_bank(row,y,fold,seed).reshape(len(y),-1)
    with np.load(ART/f'nov23_opposing_passengers/{seed}.npz') as z:
        for k,v in [('fold',fold),('ref',ref),('base',base),('row',row),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        extra.extend([z['shuffled_conditional'],z['matched_conditional']])
    p=np.column_stack([models,*extra])
    assert p.shape==(len(y),18) and np.isfinite(p).all()
    return fold,ref,base,p


def coefficients(odds):
    # Coefficient r of product_j (1 + odds[j] * s) sums all r-element subset weights.
    result=np.array([1.])
    for v in odds:
        result=np.r_[result,0.]+np.r_[0.,v*result]
    return result


def estimate(subset):
    size=subset.shape[1]
    counts=np.bincount(subset.sum(1),minlength=size+1)
    used=subset.sum(0)
    def loss(w):
        odds=np.exp(w)
        z=coefficients(odds)
        # d log Z_r / d w_j = Pr(j belongs to an r-subset under this model).
        marginals=np.column_stack([np.r_[0.,odds[j]*coefficients(np.delete(odds,j))]/z for j in range(size)])
        return counts@np.log(z)-used@w+3*(w@w), counts@marginals-used+6*w
    fit=minimize(loss,np.zeros(size),jac=True,method='L-BFGS-B',bounds=[(-8,8)]*size)
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def conditional_ratio(vote,err,t,q):
    weights=[estimate(vote[t[err[t]==flag]]) for flag in (False,True)]
    count=vote[q].sum(1)
    a,b=weights
    return vote[q]@(b-a) - np.log(coefficients(np.exp(b))[count]/coefficients(np.exp(a))[count])


def paired_ratio(vote,err,t,q):
    split=vote.reshape(len(vote),-1,2)
    separate=split.sum(2)==1
    second=split[:,:,1]
    rates=[]
    for flag in (False,True):
        ids=t[err[t]==flag]
        n=separate[ids].sum(0)
        both=(second[ids]&separate[ids]).sum(0)
        rates.append((both+4)/(n+8))
    a,b=rates
    return (separate[q]*np.where(second[q],np.log(b/a),np.log((1-b)/(1-a)))).sum(1)


def baseline_risk(vote,base,ref,err,t,q,by_family):
    count=vote.sum(1)/vote.shape[1]
    off=-abs(logit(base.clip(.002,.998)))
    cols=[np.ones(len(base)),off/3,ref.astype(float),count,count*off, np.maximum(0,count-.5)]
    if by_family:
        cols.extend((vote.reshape(len(base),-1,2).sum(2)/2).T)
    x=np.column_stack(cols)
    def loss(w):
        margin=off[t]+x[t]@w
        return (np.logaddexp(0,margin)-margin*err[t]).sum()+2*(w@w),x[t].T@(expit(margin)-err[t])+4*w
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    if not fit.success:
        raise RuntimeError(fit.message)
    return off[q]+x[q]@fit.x


def forecast(vote,fold,ref,base,g,y):
    result={}
    err=y!=ref
    rng=np.random.default_rng(914)
    # Keep the count on each individual, shuffle only model identity.
    shuffled=np.take_along_axis(vote,np.argsort(rng.random(vote.shape),axis=1),axis=1)
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=351).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            total=baseline_risk(vote,base,ref,err,t,q,False)
            families=baseline_risk(vote,base,ref,err,t,q,True)
            like=conditional_ratio(vote,err,t,q)
            placebo=conditional_ratio(shuffled,err,t,q)
            pair=paired_ratio(vote,err,t,q)
            for k,margin in [('count',total),('fixed_total',total+like),('shuffled',total+placebo),
                             ('family_counts',families),('within_pair',families+pair)]:
                result.setdefault(k,np.full(len(y),np.nan))[q]=expit(margin)
    return result


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    out=ART/'nov23_fixed_vote_totals'
    out.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,base,p=bank(seed,row,y)
        votes=(p>=.5)!=ref[:,None]
        scores=forecast(votes,fold,ref,base,g,y)
        hits={'reference':ref==y}
        for k,prob in scores.items():
            assert np.isfinite(prob).all()
            pred=ref^((prob>.5)&(ref==(base>=.5))&votes.any(1))
            hits[k]=pred==y
        plans[seed]={k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,ref=ref,base=base,fold=fold,**scores)
        print(seed,{k:(v['fixed'],v['harmed']) for k,v in plans[seed].items()},flush=True)
    ref=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-ref).mean(0),g))
    write(REPORT/'fixed_count.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
