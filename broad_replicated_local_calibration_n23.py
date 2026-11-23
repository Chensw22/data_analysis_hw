"""Estimate neighborhood calibration variance from residuals in disjoint booking samples."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit, logit, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import REPORT, cohorts, coordinates
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, SEEDS, write


def local_offsets(distance, q, t, anchor, y):
    probabilities = expit(anchor[t])
    variance = probabilities*(1-probabilities)
    residual = y[t]-probabilities
    intercept = residual.sum()/(variance.sum()+2.)
    drift, information = [], []
    for start in range(0,len(q),128):
        gap = cdist(distance[q[start:start+128]],distance[t],'sqeuclidean')
        ix = np.argsort(gap,axis=1,kind='stable')[:,:64]
        d = np.take_along_axis(gap,ix,axis=1)
        d -= d[:,:1]
        w = softmax(-d/np.maximum(.015,d[:,[31]]),axis=1)
        fisher = (w*variance[ix]).sum(1)
        error = (w*residual[ix]).sum(1)
        # Variance of the weighted residual mean under the reference Bernoulli predictions.
        information.append(fisher**2/(w*w*variance[ix]).sum(1))
        drift.append(error/fisher - intercept)
    return intercept, np.concatenate(drift), np.concatenate(information)


def estimate(coords, train, query, groups, anchor, y):
    a,b = next(GroupKFold(2,shuffle=True,random_state=510).split(train,groups=groups[train]))
    assert not np.intersect1d(groups[train[a]],groups[train[b]]).size
    halves = [local_offsets(coords,query,train[ix],anchor,y)[1] for ix in (a,b)]
    first,second = halves
    # A query receives no booking outcomes in either reference half. This single
    # variance estimate per calibration split uses query attributes but no query outcomes.
    replicated = np.mean(first*second)
    naive = .5*np.mean(first**2+second**2)
    mu,local,info = local_offsets(coords,query,train,anchor,y)
    result = {'global':anchor[query]+mu,'unpooled':anchor[query]+mu+local}
    for k,tau in [('replicated',max(0,replicated)),('uncorrected',naive)]:
        w = tau*info/(1+tau*info)
        result[k] = anchor[query]+mu+w*local
    return result, [float(replicated),float(naive),float(np.mean(info))]


def main():
    row,y,frame,groups,_ = load()
    wrong,right,hard = cohorts(row,y)
    plans,bank={},{}
    output=ART/'nov23_replicated_local_calibration'
    output.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        anchor=logit(op[:,0].clip(.003,.997))
        totals,odds={},{}
        for f in np.unique(fold):
            eligible=(ref==(op[:,0]>=.5))
            ids=np.flatnonzero((fold==f)&eligible)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=groups[ids]):
                train,q=ids[a],ids[b]
                assert not np.intersect1d(groups[train],groups[q]).size
                x,_=coordinates(frame,train)
                risk=expit(anchor)[:,None]
                for name,z in [('risk',risk),('manifest',np.c_[x,2*risk]),('opinion',op)]:
                    result,stats=estimate(z,train,q,groups,anchor,y)
                    totals.setdefault(name,[]).append(stats)
                    for method,v in result.items():
                        odds.setdefault(name+'_'+method,np.copy(anchor))[q]=v
        hits={'reference':ref==y}
        for k,v in odds.items():
            assert np.isfinite(v).all()
            hits[k]=np.where(eligible,v>=0,ref)==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,variance_observed_vs_naive={k:np.mean(v,axis=0).tolist() for k,v in totals.items()})
        if seed!='v131':
            for k,v in hits.items():bank.setdefault(k,[]).append(v)
        np.savez_compressed(output/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,**odds)
        print(seed,{k:d['fixed']-d['harmed'] for k,d in metrics.items()},flush=True)
    ref=np.stack(bank['reference'])
    stats={k:dict(common_right=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                  gain95_pp=interval((np.stack(v).astype(float)-ref).mean(0),groups)) for k,v in bank.items()}
    write(REPORT/'local_calibration.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(1):main()
