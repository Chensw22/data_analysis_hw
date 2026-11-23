"""Examine the order of dissent probabilities even when all hard votes agree."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_fixed_vote_totals_n23 import REPORT,bank
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def likelihood(order,w):
    # Each successive expert is chosen from the remaining experts with probability
    # proportional to its fitted weight. The last choice has probability one.
    ordered=w[order]
    weight=np.exp(ordered)
    denom=weight[:,::-1].cumsum(1)[:,::-1]
    return (ordered-np.log(denom)).sum(1)


def fit_order(order):
    d=order.shape[1]
    def cost(w):
        ordered=w[order]
        weights=np.exp(ordered)
        denom=weights[:,::-1].cumsum(1)[:,::-1]
        # All selected experts appear exactly once per record.
        contributions=weights*(1/denom).cumsum(1)-1
        grad=np.bincount(order.ravel(),weights=contributions.ravel(),minlength=d)
        return (np.log(denom)-ordered).sum()+6*(w@w),grad+12*w
    ans=minimize(cost,np.zeros(d),jac=True,method='L-BFGS-B',bounds=[(-8,8)]*d)
    if not ans.success:
        raise RuntimeError(ans.message)
    return ans.x


def ratio(order,y,t,q):
    negative,positive=[fit_order(order[t[y[t]==v]]) for v in (False,True)]
    return likelihood(order[q],positive)-likelihood(order[q],negative)


def fit_risk(x,y,offset):
    def loss(w):
        m=offset+x@w
        return (np.logaddexp(0,m)-m*y).sum()+3*(w@w),x.T@(expit(m)-y)+6*w
    fitted=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x


def forecast(fold,ref,base,p,g,y):
    dissent=np.where(ref[:,None],1-p,p)
    # An exact tie has no individual ordering; use the same seeded, label-independent
    # feature permutation in training and validation to break ties consistently.
    permutation=np.random.default_rng(946).permutation(p.shape[1])
    order=permutation[np.argsort(-dissent[:,permutation],axis=1,kind='stable')]
    npc=np.quantile(dissent,[.1,.25,.5,.75,.9],axis=1).T
    off=-np.abs(logit(base.clip(.002,.998)))
    shape=np.column_stack([off,ref.astype(float),(dissent>=.5).mean(1),npc,dissent.std(1)])
    hits={}
    err=y!=ref
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=351).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            training=np.full(len(t),np.nan)
            for j,k in GroupKFold(2,shuffle=True,random_state=75).split(t,groups=g[t]):
                assert not np.intersect1d(g[t[j]],g[t[k]]).size
                training[k]=ratio(order,err,t[j],t[k])
            assert np.isfinite(training).all()
            test=ratio(order,err,t,q)
            for key in ('summary','order_only','summary_and_order'):
                tx,qx=shape[t],shape[q]
                if key=='order_only':
                    tx,qx=tx[:,:2],qx[:,:2]
                if key!='summary':
                    tx,qx=np.c_[tx,training],np.c_[qx,test]
                mean,scale=tx.mean(0),tx.std(0).clip(.05)
                tx=np.column_stack([np.ones(len(t)),(tx-mean)/scale])
                qx=np.column_stack([np.ones(len(q)),(qx-mean)/scale])
                w=fit_risk(tx,err[t],off[t])
                risk=off[q]+qx@w
                hits.setdefault(key,ref.copy())[q]=ref[q]^((risk>0)&(ref[q]==(base[q]>=.5)))
    return hits


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,base,p=bank(seed,row,y)
        pred=forecast(fold,ref,base,p,g,y)
        pred['reference']=ref
        hits={k:v==y for k,v in pred.items()}
        plans[seed]={k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:(v['fixed'],v['harmed']) for k,v in plans[seed].items()},flush=True)
    r=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-r).mean(0),g))
    write(REPORT/'ranked.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
