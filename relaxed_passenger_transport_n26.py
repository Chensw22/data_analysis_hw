"""Entropic passenger matching with soft donor capacities on exact sparse neighbour lists."""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from exaone_validation_data import compact_features,CATEGORICAL
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,OUT,data
from price_implied_deck_n26 import FEES


def coordinates(raw,g,t):
    frame=compact_features(raw)
    blocks=[]
    for col in CATEGORICAL:
        observed=frame[col].astype(str).to_numpy()
        levels=np.unique(observed[t])
        # Assign unseen values a separate coordinate. Equal categorical mismatch cost is one.
        codes=pd.Index(levels).get_indexer(observed)
        blocks.append(np.eye(len(levels)+1)[codes]/np.sqrt(2))
    bill=frame[FEES].to_numpy(float)
    size=raw.groupby(g).PassengerId.transform('size')
    for values in [frame.Age,frame.CabinNumber,np.log1p(size),np.log1p(np.nansum(bill,axis=1)),*np.log1p(bill).T]:
        a=np.array(values,float)
        known=np.isfinite(a)
        center=np.nanmedian(a[t])
        lo,hi=np.nanquantile(a[t],[.05,.95])
        scaled=np.clip((np.where(known,a,center)-center)/max(hi-lo,.5),-2,2)
        blocks.extend([scaled[:,None],(~known)[:,None]*.5])
    blocks.append((bill>0)*.65)
    return np.column_stack(blocks)


def nearest(x,q,t,k,g,exclude_orders=False):
    keep=[];cost=[]
    for block in np.array_split(q,max(1,(len(q)+127)//128)):
        d=cdist(x[block],x[t],'sqeuclidean')
        if exclude_orders:
            d[g[block,None]==g[t][None,:]]=np.inf
        idx=np.argsort(d,axis=1,kind='stable')[:,:k]
        keep.append(idx)
        cost.append(np.take_along_axis(d,idx,axis=1))
    return np.vstack(keep),np.vstack(cost)


def adjust(kernel,index,n,rho):
    # Row sums equal one. Column totals c have KL penalty rho relative to uniform m/n.
    # The unpenalized control is the same kernel normalized independently within each row.
    m=len(kernel)
    local=kernel/kernel.sum(1,keepdims=True)
    def marg(plan):
        return np.bincount(index.ravel(),weights=plan.ravel(),minlength=n)
    active=marg(local)>0
    b=m/n
    logv=np.zeros(n)
    exponent=rho/(rho+1.)
    for i in range(500):
        scale=np.exp(logv[index])
        u=1/(kernel*scale).sum(1)
        unscaled=marg(kernel*u[:,None])
        target=exponent*(np.log(b)-np.log(unscaled[active]))
        change=float(np.max(abs(logv[active]-target)))
        logv[active]=target
        if change<1e-9:
            break
    else:
        raise RuntimeError('Soft column scaling did not converge.')
    weighted=kernel*np.exp(logv[index])
    plan=weighted/weighted.sum(1,keepdims=True)
    col=marg(plan)
    kkt=float(np.max(abs(logv[active]+rho*np.log(col[active]/b))))
    assert kkt<1.e-7
    np.testing.assert_allclose(plan.sum(1),1.,atol=1.e-12)
    def diag(v):
        col=marg(v)/m
        ess=1./np.square(v).sum(1)
        return dict(donor_effective_n=float(1/(col@col)),donor_L1_from_uniform=float(abs(col-1/n).sum()),
                    min_median_row_effective_neighbors=[float(ess.min()),float(np.median(ess))])
    return local,plan,dict(iterations=i+1,stationarity_error=kkt,supported_donors=int(active.sum()),
                           training_donors=n,local=diag(local),relaxed=diag(plan))


def run():
    raw,y,g,f=data()
    outputs=np.zeros((len(y),2)); diagnostics=[]
    for k in np.unique(f):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        x=coordinates(raw,g,t)
        subset=np.random.default_rng(801+int(k)).choice(t,640,replace=False)
        _,near=nearest(x,subset,t,12,g,True)
        eps=max(.03,float(np.median(near[:,-1]))/2.)
        ids,cost=nearest(x,q,t,96,g)
        kernel=np.exp(-(cost-cost[:,:1])/eps)
        local,relaxed,diag=adjust(kernel,ids,len(t),2.)
        for j,plan in enumerate([local,relaxed]):
            outputs[q,j]=(plan*y[t[ids]]).sum(1)
        diag.update(fold=int(k),epsilon=eps,source_rate=float(y[t].mean()),
                    local_mean=float(outputs[q,0].mean()),relaxed_mean=float(outputs[q,1].mean()))
        diagnostics.append(diag)
        print('fold',int(k),'soft iterations',diag['iterations'],flush=True)
    assert np.isfinite(outputs).all() and ((outputs>=0)&(outputs<=1+1e-12)).all()
    outputs=outputs.clip(0,1)
    folder=ROOT/'artifacts/nov26_relaxed_passenger_transport'
    folder.mkdir(exist_ok=True)
    tmp=folder/'probabilities.tmp.npz';archive=folder/'eight_fold_oof.npz'
    np.savez_compressed(tmp,passenger=raw.PassengerId.to_numpy(str),row=np.arange(len(y)),fold=f,y=y,
                        local=outputs[:,0],relaxed=outputs[:,1])
    tmp.replace(archive)
    with np.load(OUT/'mitra96.npz') as z,np.load(archive) as saved:
        for key in ('y','fold','passenger'):
            np.testing.assert_array_equal(saved[key],z[key])
        for j,name in enumerate(('local','relaxed')):
            np.testing.assert_array_equal(saved[name],outputs[:,j])
        mitra=z['p'].clip(.003,.997)
    _,cur,_=inputs()
    parent,wrong=cur['parent'],cur['wrong']
    lo,rel=outputs.T>=.5
    lb,rb=(.75*logit(mitra[:,None])+.25*logit(outputs.clip(.003,.997))>=0).T
    actions=dict(local=lo!=parent,relaxed=rel!=parent,relaxed_only=(rel!=parent)&(lo==parent),
                 blend_local=lb!=parent,blend_relaxed=rb!=parent,
                 blend_unique=(rb!=parent)&(lb==parent)&((mitra>=.5)==parent))
    report=dict(neighbors=96,donor_kl_over_entropy=2,fold_diagnostics=diagnostics,
       whole_accuracy={name:float((p==y).mean()) for name,p in [('local',lo),('relaxed',rel),
         ('mitra',mitra>=.5),('blend_local',lb),('blend_relaxed',rb)]},scopes={})
    for tag,mask in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
        record={name:summary(c,wrong,mask,f) for name,c in actions.items()}
        for name,a,b in [('relaxed_vs_local',rel,lo),('blended_relaxed_vs_local',rb,lb),
                         ('blended_relaxed_vs_mitra',rb,mitra>=.5)]:
            record[name]=summary(a!=b,b!=y,mask,f)
        report['scopes'][tag]=record
    report['scope']='Same 96 closest individual donors and same costs for both variants. Bandwidth is half the training median squared distance to the 12th different-order neighbour among 640 training query rows (floor .03); each selected neighbour is from a different order than its query, several can share an order with each other. Capacity is per donor passenger. Uniform donor capacity is penalized softly, with no class total imposed. Zero-support columns contribute a constant KL cost. Trimmed support approximates dense transport. Existing booking_transport_classifier_n26 compressed by class to prototypes and enforced strict capacity, broad_passenger_transport_n23 used strict balancing on a smaller nested roster.'
    (ROOT/'reports/nov23_matched_learning_curve/relaxed_passenger_transport.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['whole_accuracy'], {k:(v['fixed'],v['harmed']) for k,v in report['scopes']['common'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
