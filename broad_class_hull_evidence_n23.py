"""Measure how well neighboring cases of each outcome can jointly reconstruct a query profile."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import REPORT,cohorts,coordinates
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,SEEDS,write


def closest_mixture(gram):
    """Frank-Wolfe with exact segment minimization for squared distance plus a weight penalty."""
    n,k,_=gram.shape
    idx=np.arange(n)
    matrix=gram+.12*np.eye(k)
    weight=np.full((n,k),1/k)
    moment=matrix.mean(2)
    initial=moment.mean(1)
    for _ in range(64):
        vertex=moment.argmin(1)
        surface=(weight*moment).sum(1)
        gap=surface-moment[idx,vertex]
        curvature=surface-2*moment[idx,vertex]+matrix[idx,vertex,vertex]
        step=np.clip(gap/np.maximum(curvature,1e-12),0,1)
        weight*=1-step[:,None]
        weight[idx,vertex]+=step
        moment=(1-step[:,None])*moment+step[:,None]*matrix[idx,vertex]
    value=(weight*moment).sum(1)
    assert (value<=initial+1e-9).all()
    np.testing.assert_allclose(weight.sum(1),1,atol=1e-10)
    return initial,value,value-moment.min(1)


def evidence(x,t,q,y):
    observations=[]
    for start in range(0,len(q),96):
        ids=q[start:start+96]
        distance=cdist(x[ids],x[t],'sqeuclidean')
        neighbours=[]
        for label in (False,True):
            pool=np.flatnonzero(y[t]==label)
            ix=np.argsort(distance[:,pool],axis=1,kind='stable')[:,:24]
            neighbours.append(t[pool[ix]])
        deltas=[x[ix]-x[ids,None] for ix in neighbours]
        scale=np.maximum(.15,np.mean([np.square(d).sum(2)[:,-1] for d in deltas],axis=0))
        sides=[]
        for delta in deltas:
            gram=(delta@delta.transpose(0,2,1))/scale[:,None,None]
            sq=gram.diagonal(axis1=1,axis2=2)
            between=np.maximum(sq[:,:,None]+sq[:,None,:]-2*gram,0.)
            rbf=np.exp(-between)-np.exp(-sq[:,:,None])-np.exp(-sq[:,None,:])+1
            terms=[np.log1p(sq[:,-1]),np.log1p(sq.mean(1))]
            for kernel in (gram,rbf):
                terms.extend(closest_mixture(kernel))
            sides.append(np.column_stack(terms))
        observations.append(np.stack(sides,axis=1))
    return np.vstack(observations)


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    out=ART/'nov23_class_hulls'
    out.mkdir(exist_ok=True)
    bank,plans={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        raw=np.full((len(y),2,8),np.nan)
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            x,_=coordinates(frame,t)
            raw[q]=evidence(x,t,q,y)
        assert np.isfinite(raw).all()
        features={k:raw[:,:,i].reshape(len(y),-1) for k,i in
                  {'radii':[0,1], 'centroids':[0,1,2,5], 'hulls':[0,1,2,3,5,6]}.items()}
        anchor=logit(op[:,0].clip(.003,.997))
        scores={}
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=478).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                for k,h in features.items():
                    scale=StandardScaler().fit(h[t])
                    x=scale.transform(h).clip(-8,8)
                    fit=LogisticRegression(C=.2,max_iter=250).fit(x[t],y[t])
                    scores.setdefault(k,np.full(len(y),np.nan))[q]=fit.predict_proba(x[q])[:,1]
        votes={'reference':ref}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            votes[k]=p>=.5
            votes['quarter_'+k]=np.where((anchor>=0)==ref,.75*op[:,0]+.25*p>=.5,ref)
        shift=logit(scores['hulls'].clip(.003,.997))-logit(scores['centroids'].clip(.003,.997))
        votes['increment']=np.where((anchor>=0)==ref,anchor+shift>=0,ref)
        hits={k:v==y for k,v in votes.items()}
        changes={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(models=changes, mean_duality_gap=raw[:,:,[4,7]].mean((0,1)).tolist(),
                         mean_uniform_distance=raw[:,:,[2,5]].mean((0,1)).tolist(),
                         mean_optimized_distance=raw[:,:,[3,6]].mean((0,1)).tolist())
        if seed!='v131':
            for k,v in hits.items():bank.setdefault(k,[]).append(v)
        print(seed,{k:d['fixed']-d['harmed'] for k,d in changes.items()},flush=True)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],raw=raw,**scores)
    reference=np.stack(bank['reference'])
    stats={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                  hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                  gain95_pp=interval((np.stack(v).astype(float)-reference).mean(0),g)) for k,v in bank.items()}
    write(REPORT/'class_hulls.json',dict(plans=plans,stable=stats))


if __name__=='__main__':
    with threadpool_limits(1):main()
