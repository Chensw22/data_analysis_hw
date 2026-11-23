"""Learn class-conditional patterns of correlated expert votes on independent booking groups."""
import numpy as np
from scipy.special import expit, logit, logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from paid_conditional_tilt_n23 import REPORT
from released_correction_audit_n23 import development


def ballots(seed,row,y,fold,base):
    def get(directory,key):
        with np.load(ART/f'nov23_{directory}/{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('fold',fold),('base',base)]:np.testing.assert_array_equal(z[k],v)
            return z[key]
    # Use first-stage predictions fitted entirely outside the original query fold.
    p=np.column_stack([get('chained_tree_ensemble','chained'),expit(get('tabicl_prior_context','sides')[:,:2].mean(1)),
        expit(get('class_cost_path','logits')[:,1]),get('booking_coordinates','real'),
        expit(get('local_class_shapes','votes')),get('plausible_replacements','local')[:,-1],
        expit(get('booking_size_likelihood','own')),get('broad_neural_path','epoch_80'),
        get('broad_bill_exception','neutral'),get('broad_service_transport','learned')])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p


def state_loglik(train, query, modes):
    # Each passenger belongs to one latent voting regime. Conditioned on this regime,
    # the categorical ballot probabilities factor across experts.
    n,voters,levels=train.shape
    overall=(train.sum(0)+1)/(n+levels)
    order=np.argsort((train@np.arange(levels)).mean(1),kind='stable')
    resp=np.zeros((n,modes))
    for j,indices in enumerate(np.array_split(order,modes)):
        resp[indices,j]=1.
    for _ in range(65):
        count=resp.sum(0)
        prob=(np.einsum('nk,nvl->kvl',resp,train)+5*overall)/(count[:,None,None]+5)
        weight=(count+1)/(n+modes)
        params=np.log(prob).reshape(modes,-1).T
        log_weights=np.log(weight)
        logits=train.reshape(n,-1)@params+log_weights
        updated=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        if np.abs(resp-updated).max()<1.e-5:
            break
        resp=updated
    return logsumexp(query.reshape(len(query),-1)@params+log_weights,axis=1)


def score(p,y,fit,query,mode):
    # Three levels are based on a model's rank within the training bookings to permit differing calibrations.
    cuts=np.quantile(p[fit],[1/3,2/3],axis=0)
    codes=(p>cuts[0]).astype(int)+(p>cuts[1])
    rng=np.random.default_rng(531)
    terms=[]
    for flag in [False,True]:
        ids=fit[y[fit]==flag]
        train=codes[ids].copy()
        if mode=='shuffled':
            for col in range(train.shape[1]):
                train[:,col]=rng.permutation(train[:,col])
        terms.append(state_loglik(np.eye(3)[train],np.eye(3)[codes[query]],1 if mode=='independent' else 3))
    return terms[1]-terms[0]


def run():
    row,y,frame,g,_=load()
    ids,labels,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    _,release,_=development()
    state=strata(frame)
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    hist,refs,plans={},[],{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,release[131] if seed=='v131' else base>=.5)
        p=ballots(seed,row,y,fold,base)
        modes=['mean','independent','latent','shuffled']
        prediction={k:ref.copy() for k in modes}
        fixed=np.c_[logit(base.clip(.002,.998)),logit(p.clip(.002,.998)).mean(1)]
        for f in np.unique(fold):
            ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
            parts=[ix[b] for _,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix])]
            for i,q in enumerate(parts):
                prob={k:[] for k in modes}
                for j in [1,2]:
                    train,cal=parts[(i+j)%3],parts[(i+3-j)%3]
                    for a,b in [(train,cal),(train,q),(cal,q)]: assert not np.intersect1d(g[a],g[b]).size
                    joint=np.r_[cal,q]
                    for mode in modes:
                        x=fixed[joint]
                        if mode!='mean':
                            ratio=score(p,y,train,joint,mode)
                            x=np.c_[x,ratio]
                        prob[mode].append(calibrated(x[:len(cal)],x[len(cal):],y,cal))
                for mode,s in prob.items():prediction[mode][q]=np.mean(s,axis=0)>=.5
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        plans[seed]={}
        for k,v in prediction.items():
            hit=v==y
            plans[seed][k]=dict(accuracy=float(hit.mean()),fixed=int((hit&~anchor).sum()),harmed=int((~hit&anchor).sum()))
            if seed!='v131':hist.setdefault(k,[]).append(hit)
        print(seed,plans[seed]['latent'],flush=True)
    write(REPORT/'vote_patterns.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        run()
