"""Integrate over a single possible exchange in each booking using only outcomes of training bookings."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_link_authenticity_n23 import REPORT
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_within_booking_bill_matching_n23 import encode
from catboost import CatBoostClassifier


def packets(frame,g,size,person,bill,row,q,t,base,ref,y):
    # Each packet covers every member in a validation booking. Only scored passengers
    # have labels in the inner fit. Other passengers contribute attributes only.
    active=q[(size[row[q]]>1)]
    full=np.flatnonzero(np.isin(g,g[row[active]]))
    members=pd.Series(full).groupby(g[full]).agg(list)
    x=np.c_[person,bill,np.log(size)]
    model=CatBoostClassifier(iterations=300,depth=5,learning_rate=.06,l2_leaf_reg=4,thread_count=2,
                             verbose=False,allow_writing_files=False,random_seed=806).fit(x[row[t]],y[t])
    idx=np.full(len(frame),-1)
    idx[row[active]]=active
    results=[]
    costs=frame[BILLS].fillna(0).to_numpy()
    for n in sorted(set(map(len,members))):
        m=np.array([r for r in members if len(r)==n])
        a,b=np.array(list(combinations(range(n),2))).T
        own=np.broadcast_to(m[:,:,None],(len(m),n,n))
        donor=own.transpose(0,2,1)
        flat=own.ravel()
        query=np.c_[person[flat],bill[donor.ravel()],np.log(size[flat])]
        raw=logit(model.predict_proba(query)[:,1].clip(.002,.998)).reshape(own.shape)
        offset=raw-raw.diagonal(axis1=1,axis2=2)[:,:,None]
        scored=idx[m]
        valid=scored>=0
        safe=np.maximum(scored,0)
        adjustable=valid & (ref[safe]==(base[safe]>=.5))
        # Differences for unseen pairs are never used for scoring a passenger outside this fold.
        delta=np.zeros((len(m),1+len(a),n))
        s=np.arange(1,len(a)+1)
        delta[:,s,a]=offset[:,a,b]
        delta[:,s,b]=offset[:,b,a]
        p=expit(logit(base[safe].clip(.002,.998))[:,None,:]+delta)
        age=frame.Age.to_numpy()[m]
        same_sleep=frame.CryoSleep.to_numpy()[m[:,a]]==frame.CryoSleep.to_numpy()[m[:,b]]
        amt=np.log1p(costs.sum(1))[m]
        same_berth=(frame.Side.to_numpy()[m[:,a]]==frame.Side.to_numpy()[m[:,b]]) & (
                    frame.CabinNumber.to_numpy()[m[:,a]]==frame.CabinNumber.to_numpy()[m[:,b]]) & (
                    frame.Deck.to_numpy()[m[:,a]]==frame.Deck.to_numpy()[m[:,b]])
        design=np.zeros((len(m),1+len(a),5))
        design[:,1:,:]=np.stack([np.ones_like(same_sleep),same_sleep,same_berth,
            np.nan_to_num(abs(age[:,a]-age[:,b]),nan=15)/30,abs(amt[:,a]-amt[:,b])/6],axis=2)
        prior=np.r_[0.,np.full(len(a),-np.log(len(a))-1.)]
        observed=np.where(y[safe][:,None,:],p,1-p)
        ll=np.where(adjustable[:,None,:],np.log(observed),0.)
        results.append(dict(g=g[m[:,0]],x=design,prior=prior,p=p,ll=ll,idx=scored,valid=valid))
    return results


def fit(packets,train_groups,joint):
    use=[]
    for item in packets:
        t=np.isin(item['g'],train_groups)
        if t.any():use.append((item['x'][t],item['prior'],item['ll'][t]))
    def objective(w):
        value=4*(w@w)
        grad=8*w
        for x,prior,ll in use:
            logits=x@w+prior
            z=logsumexp(logits,axis=1)
            probability=softmax(logits,axis=1)
            if joint:
                obs=logits+ll.sum(2)
                value+=(z-logsumexp(obs,axis=1)).sum()
                residual=probability-softmax(obs,axis=1)
            else:
                obs=logits[:,:,None]+ll
                value+=(z[:,None]-logsumexp(obs,axis=1)).sum()
                residual=(probability[:,:,None]-softmax(obs,axis=1)).sum(2)
            grad+=np.einsum('ns,nsd->d',residual,x)
        return value,grad
    ans=minimize(objective,np.zeros(5),method='L-BFGS-B',jac=True)
    if not ans.success:raise RuntimeError(ans.message)
    return ans.x


def score(items,query_groups,coef,result):
    additional=[]
    for item in items:
        mask=np.isin(item['g'],query_groups)
        if not mask.any():continue
        weights=softmax(item['x'][mask]@coef+item['prior'],axis=1)
        p=np.einsum('ns,nsi->ni',weights,item['p'][mask])
        idx=item['idx'][mask]
        valid=item['valid'][mask]
        result[idx[valid]]=p[valid]
        additional.extend((1-weights[:,0]).tolist())
    return additional


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    frame,groups,size,_=load_roster()
    np.testing.assert_array_equal(groups[row],g)
    person,bill=encode(frame)
    many=size[row]>1
    plans,hist={},{}
    out=ART/'nov23_latent_bill_exchange'
    out.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        base=op[:,0]
        scores={k:base.copy() for k in ['fixed','individual','joint']}
        weights={k:[] for k in scores}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            pack=packets(frame,groups,size,person,bill,row,q,t,base,ref,y)
            ids=q[many[q]]
            for a,b in GroupKFold(3,shuffle=True,random_state=312).split(ids,groups=g[ids]):
                train,query=np.unique(g[ids[a]]),np.unique(g[ids[b]])
                assert not np.intersect1d(train,query).size
                for k in scores:
                    coef=np.zeros(5) if k=='fixed' else fit(pack,train,k=='joint')
                    weights[k]+=score(pack,query,coef,scores[k])
        hits={'reference':ref==y}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            np.testing.assert_array_equal(p[~many],base[~many])
            hits[k]=np.where(ref==(base>=.5),p>=.5,ref)==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,mean_exchange_weight={k:float(np.mean(v)) for k,v in weights.items()})
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},plans[seed]['mean_exchange_weight'],flush=True)
        if seed!='v131':
            for k,v in hits.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**scores)
    r=np.stack(hist['reference'])
    stats={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                  gain95_pp=interval((np.stack(v).astype(float)-r).mean(0),g)) for k,v in hist.items()}
    write(REPORT/'latent_exchange.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(1):main()
