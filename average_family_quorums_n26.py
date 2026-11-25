"""Average over all family voting rules using historical error rates on both sides of each split."""
import json
from math import comb

import numpy as np
from scipy.special import betaln, logsumexp
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import all_gains, inputs, summary
from tabnet_attention_transfer_n26 import RESULT


def expectation(weight,width):
    """Transpose of the subset/quorum transform: expected value for every possible vote pattern."""
    t=np.cumsum(weight,axis=1)
    for bit in reversed(range(width)):
        view=t.reshape(-1,2,1<<bit,width+1)
        low=view[:,0,:,:bit+1].copy()
        same=view[:,1,:,:bit+1].copy()
        shifted=view[:,1,:,1:bit+2].copy()
        view[:,0,:,:bit+1]=low+same
        view[:,1,:,:bit+1]=low+shifted
    return t[:,0].copy()


def average(code,wrong,width):
    support=all_gains(code,np.ones(len(code),bool),width)
    gains=all_gains(code,wrong,width)
    w=(gains+support)//2
    r=support-w
    wrong_out=wrong.sum()-w
    right_out=(~wrong).sum()-r
    sizes=np.bitwise_count(np.arange(1<<width,dtype=np.uint32))
    k=np.arange(width+1)
    valid=(k[None,:]>=1)&(k[None,:]<=sizes[:,None])
    # Prior mass 1/2 on homogeneous risk. In the other half, give equal mass
    # to each committee size, then to each subset and to each attainable quorum.
    prior=np.zeros(width+1)
    for j in range(1,width+1):
        prior[j]=-np.log(2*width)-np.log(comb(width,j))-np.log(j)
    prior=prior[sizes]
    nbad=int(wrong.sum())
    ngood=len(wrong)-nbad
    null=betaln(nbad+1,ngood+1)
    likelihood_ratio=betaln(w+1,r+1)+betaln(wrong_out+1,right_out+1)-null
    inside=(w+1)/(support+2)
    outside=(wrong_out+1)/(len(wrong)-support+2)
    results={}
    for name,power in [('ordinary',1.),('discounted',.25)]:
        logweights=power*likelihood_ratio+prior[:,None]
        logweights[~valid]=-np.inf
        norm=np.logaddexp(logsumexp(logweights),np.log(.5))
        weights=np.exp(logweights-norm)
        null_mass=.5*np.exp(-norm)
        assert abs(weights.sum()+null_mass-1)<1e-8
        base=null_mass*(nbad+1)/(len(wrong)+2)+np.sum(weights*outside)
        probability=base+expectation(weights*(inside-outside),width)
        assert np.min(probability)>=-1.e-8 and np.max(probability)<=1+1.e-8
        results[name]=(probability,dict(homogeneous_mass=float(null_mass),
                                       largest_model_mass=float(weights.max()),source_rows=len(code)))
    return results


def check_inverse():
    rng=np.random.default_rng(973)
    for width in (2,5):
        n=1<<width
        weights=rng.normal(size=(n,width+1))
        weights[:,0]=0
        a=expectation(weights,width)
        for q in range(n):
            exact=sum(weights[m,1:(m&q).bit_count()+1].sum() for m in range(n))
            np.testing.assert_allclose(a[q],exact,atol=1.e-10)
        codes=rng.integers(n,size=50)
        wrong=rng.integers(2,size=50).astype(bool)
        output=average(codes,wrong,width)
        for name,power in [('ordinary',1.),('discounted',.25)]:
            # Explicit beta-binomial posterior for one potential test pattern.
            q=3
            denominator=.5
            numerator=.5*(wrong.sum()+1)/(len(wrong)+2)
            for m in range(1,n):
                size=m.bit_count()
                for k in range(1,size+1):
                    inside=np.bitwise_count(codes&m)>=k
                    score=0.
                    for mask in [inside,~inside]:
                        err=wrong[mask].sum()
                        score+=betaln(err+1,mask.sum()-err+1)
                    log_ratio=score-betaln(wrong.sum()+1,(~wrong).sum()+1)
                    weight=.5*np.exp(power*log_ratio)/(width*comb(width,size)*size)
                    source=inside if (q&m).bit_count()>=k else ~inside
                    numerator+=weight*(wrong[source].sum()+1)/(source.sum()+2)
                    denominator+=weight
            np.testing.assert_allclose(output[name][0][q],numerator/denominator,atol=1.e-10)


def run():
    check_inverse()
    names,cur,hist=inputs()
    width=len(names)
    result={}
    for scope in ['common','known']:
        for scheme in ['global_rule','direction','direction_solo_paid']:
            train=hist[scope]
            q=cur[scope]
            ts,qs=hist['cells'][scheme],cur['cells'][scheme]
            models={key:np.zeros(len(q),bool) for key in ['ordinary','discounted']}
            weights={}
            for level in np.unique(ts[train]):
                at=train&(ts==level)
                target=q&(qs==level)
                out=average(hist['code'][at],hist['wrong'][at],width)
                weights[int(level)]={}
                for name,(p,record) in out.items():
                    models[name][target]=p[cur['code'][target]]>.5
                    weights[int(level)][name]=record
            entry=dict(posterior=weights)
            for name,edit in models.items():
                entry[name]={key:summary(edit,cur['wrong'],q&region,cur['fold']) for key,region in
                             [('all_including_training_orders',q),('later_orders',cur['new'])]}
            result[scope+'_'+scheme]=entry
            print(scope,scheme,[(n,entry[n]['later_orders']['fixed'],entry[n]['later_orders']['harmed']) for n in models],flush=True)
    result['interpretation']='The training cohort predates the latest TabNet correction and contains no later orders. Each possible split has two Beta(1,1) error rates; the null has one. Calibration treats training passengers as Bernoulli observations; quarter likelihood is a sensitivity analysis for correlated bookings and repeated predictors. The actual later-order scores are the evaluation.'
    (RESULT.parent/'average_family_quorums.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
