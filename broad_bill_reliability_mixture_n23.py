"""Estimate probabilistic service reliability from held-out billing states and routing losses."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import SEEDS,write
from broad_unexpected_bill_n23 import OUT,REPORT


def weights(x,known,theta):
    intercept, slope=theta
    # The intercept gives the prior odds of withholding any one measured field.
    num=known.sum(1).clip(1)
    logits=np.where(known,intercept+slope*x-np.log(num[:,None]),-np.inf)
    return softmax(np.column_stack([np.zeros(len(x)),logits]),axis=1)


def fit_route(x,known,expert,y,slope):
    observed=np.where(y[:,None],expert,1-expert)
    def obj(theta):
        w=weights(x,known,theta)
        likelihood=(w*observed).sum(1)
        post=w*observed/likelihood[:,None]
        diff=w[:,1:]-post[:,1:]
        prior=theta-np.array([-2.,0.])
        loss=-np.log(likelihood).sum()+.5*prior@prior
        return loss,np.array([diff.sum(),(diff*x).sum()])+prior
    lim=[(-8,4),(-3,3) if slope else (0,0)]
    fit=minimize(obj,[-2.,0.],jac=True,bounds=lim,method='L-BFGS-B')
    assert fit.success,fit.message
    return fit.x


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    history,plans={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(OUT/f'{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,ref,base,full,masked,p1,p2,known,state=[z[k] for k in ('fold','ref','base','full','masked','p1','p2','known','state')]
        odds=lambda p:logit(p.clip(.002,.998))
        expert=np.column_stack([base,expit(odds(base)[:,None]+odds(masked)-odds(full)[:,None])]).clip(.001,.999)
        scores={k:base.copy() for k in ('uniform','rare_state','conditional_only','likelihood_ratio','attribute_fit')}
        settings=[]
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=484).split(ids,groups=g[ids]):
                train,query=ids[a],ids[b]
                assert not np.intersect1d(g[train],g[query]).size
                train=train[ref[train]==(base[train]>=.5)]
                freq=np.stack([(1+(known[train] & (state[train]==s)).sum(0))/(3+known[train].sum(0)) for s in range(3)]).T
                marginal=freq[np.arange(5),state]
                logcond=.5*(np.log(p1.clip(.004,1.))+np.log(p2.clip(.004,1.)))
                ratio=np.where(known,np.log(marginal)-logcond,0.)
                designs={'uniform':np.zeros_like(ratio),'rare_state':-np.log(marginal),
                         'conditional_only':-logcond,'likelihood_ratio':ratio}
                for k,x in designs.items():
                    theta=fit_route(x[train],known[train],expert[train],y[train],k!='uniform')
                    w=weights(x[query],known[query],theta)
                    scores[k][query]=(w*expert[query]).sum(1)
                    if k=='likelihood_ratio':
                        settings.append(theta.tolist())
                # A one-corrupted-field likelihood surrogate, fitted from attributes alone.
                relative=np.where(known[train],np.exp(ratio[train]),0).sum(1)/known[train].sum(1).clip(1)
                delta=relative-1
                opt=minimize(lambda v:(-np.log1p(v[0]*delta).sum(),np.array([-(delta/(1+v[0]*delta)).sum()])),
                             [.1],jac=True,bounds=[(.001,.999)],method='L-BFGS-B')
                assert opt.success,opt.message
                w=weights(ratio[query],known[query],[logit(opt.x[0]),1.])
                scores['attribute_fit'][query]=(w*expert[query]).sum(1)
        hits={k:(np.where(ref==(base>=.5),v>=.5,ref)==y) for k,v in scores.items()}
        hits['reference']=ref==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(metrics=metrics,learned_ratio_parameters=settings)
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:m['fixed']-m['harmed'] for k,m in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       stable_harms=int((right&~h.any(0)).sum()),common_stable=int((wrong&h.all(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),g))
    write(REPORT/'reliability_mixture.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
