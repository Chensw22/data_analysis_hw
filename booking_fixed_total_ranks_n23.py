"""Factor a booking into its unknown total and a conditional allocation with within-booking ranks."""
from itertools import product
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partial_booking_likelihood_n23 import contexts
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT=ROOT/'reports/nov23_booking_fixed_total_ranks'


def rank_design(frame, g):
    # Observed members get their centered midrank; unobserved members carry no rank evidence.
    fields=frame[['Age','CabinNumber',*BILLS]].copy()
    fields['sum']=frame[BILLS].sum(axis=1,min_count=len(BILLS))
    ranked=fields.groupby(g).rank(method='average',na_option='keep')
    n=fields.notna().groupby(g).transform('sum')
    return ((2*ranked-n-1)/(n-1).clip(lower=1)).fillna(0).to_numpy()


def blocks(ids,groups,prior,rank,observed):
    members=pd.Series(ids).groupby(groups[ids]).agg(list)
    out=[]
    for width in range(2,max(map(len,members))+1):
        subset=[v for v in members if len(v)==width]
        if not subset:
            continue
        ix=np.array(subset)
        states=np.array(list(product((0.,1.),repeat=width)))
        sums=states.sum(1).astype(int)
        base=prior[ix]@states.T
        alloc=np.einsum('si,nij->nsj',states,rank[ix])
        p=expit(prior[ix])
        k=np.arange(width+1)[None,:]
        centered=(k-p.sum(1)[:,None])/np.sqrt(width)
        extreme=(k==0)|(k==width)
        ctx=np.c_[np.ones(len(ix)),p.mean(1)-.5,p.std(1),np.full(len(ix),np.log(width))]
        count=np.concatenate([centered[:,:,None]*ctx[:,None,:],extreme[:,:,None]*ctx[:,None,:]],axis=2)
        mass=np.column_stack([logsumexp(base[:,sums==j],axis=1) for j in k[0]])
        out.append((ix,states,sums,base,mass,alloc,count,observed[ix]))
    return out


def fit(parts,kind):
    pack=[]
    for _,states,sums,base,mass,alloc,count,truth in parts:
        # Only use totals of completely labeled training bookings; partial bookings still
        # supply labels for the individual teacher and all their known roster features for inference.
        complete=(truth>=0).all(1)
        for k in np.unique(truth[complete].sum(1)):
            keep=complete&(truth.sum(1)==k)
            if kind=='count':
                pack.append((mass[keep],count[keep],count[keep,k]))
            elif 0<k<truth.shape[1]:
                allowed=sums==k
                true_state=(truth[keep,None,:]==states[None,:,:]).all(2)
                assert np.all(true_state.sum(1)==1)
                pack.append((base[keep][:,allowed],alloc[keep][:,allowed,:],alloc[keep][true_state]))
    def obj(w):
        loss=10.*(w@w)
        grad=20.*w
        for offset,x,positive in pack:
            z=offset+x@w
            loss+=np.sum(logsumexp(z,axis=1)-positive@w)
            grad+=np.einsum('ns,nsd->d',softmax(z,axis=1),x)-positive.sum(0)
        return loss,grad
    optim=minimize(obj,np.zeros(pack[0][1].shape[2]),jac=True,method='L-BFGS-B')
    assert optim.success,optim.message
    return optim.x


def predict(parts,prior,count_w,alloc_w):
    out={name:expit(prior.copy()) for name in ['independent','count','order','both','shuffled_order']}
    rng=np.random.default_rng(431)
    for ix,states,sums,base,mass,alloc,count,_ in parts:
        plain=softmax(mass,axis=1)
        learned=softmax(mass+count@count_w,axis=1)
        # Permute the fitted within-booking score among real companions at inference.
        single=np.array([np.flatnonzero((states[:,i]==1)&(sums==1))[0] for i in range(ix.shape[1])])
        score=alloc[:,single,:]@alloc_w
        shuffled=np.take_along_axis(score,np.argsort(rng.random(score.shape),axis=1),axis=1)
        for name,allocation,dist in [('independent',0.,plain),('count',0.,learned),('order',alloc@alloc_w,plain),
                                     ('both',alloc@alloc_w,learned),('shuffled_order',shuffled@states.T,learned)]:
            logits=base+allocation
            p=np.zeros(ix.shape)
            for k in range(ix.shape[1]+1):
                allowed=sums==k
                p+=dist[:,k,None]*(softmax(logits[:,allowed],axis=1)@states[allowed])
            np.testing.assert_allclose(p.sum(1),dist@np.arange(dist.shape[1]),atol=1e-10)
            out[name][ix]=p
    np.testing.assert_allclose(out['independent'],expit(prior),atol=1e-10,equal_nan=True)
    return out


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    frame,groups,size,test_info=load_roster()
    frame['BookingSize']=size
    total_train=len(frame)-test_info['test_rows']
    np.testing.assert_array_equal(groups[row],g)
    rank=rank_design(frame,groups)
    multi=size[row]>1
    fullbill=frame[BILLS].notna().all(axis=1).to_numpy()
    scope=dict(development=len(y),development_multi=int(multi.sum()),development_multi_complete_bills=int((multi&fullbill[row]).sum()),
               test_multi=int((size[total_train:]>1).sum()),test_multi_complete_bills=int(((size[total_train:]>1)&fullbill[total_train:]).sum()),
               hard_multi=int((hard&multi).sum()))
    plans,history={},{}
    directory=ART/'nov23_booking_fixed_total_ranks'
    directory.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['y'],y)
            np.testing.assert_array_equal(z['row'],row)
            fold,ref,anchor=z['fold'],z['ref'],z['base']
        bank,diagnostics={},[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            obs,it,zt,iq,zq=contexts(frame,groups,row,y,t,q)
            support=blocks(it,groups,zt,rank,obs)
            count_w=fit(support,'count')
            order_w=fit(support,'order')
            complete=sum((x[-1]>=0).all(1).sum() for x in support)
            diagnostics.append(dict(complete_multi_bookings=int(complete),allocation_weights=order_w.tolist()))
            queries=[predict(blocks(iq,groups,v,rank,obs),v,count_w,order_w) for v in zq]
            for name in queries[0]:
                bank.setdefault(name,np.full(len(y),np.nan))[q]=np.mean([v[name][row[q]] for v in queries],axis=0)
        results={}
        for name,p in bank.items():
            assert np.isfinite(p).all()
            locked=ref!=(anchor>=.5)
            votes={name:p>=.5,'quarter_'+name:np.where(locked,ref,.75*anchor+.25*p>=.5)}
            # Transfer only the difference between the structured and independent predictions.
            delta=logit(p.clip(.001,.999))-logit(bank['independent'].clip(.001,.999))
            votes['transfer_'+name]=np.where(locked,ref,logit(anchor.clip(.001,.999))+delta>=0)
            for label,v in votes.items():
                hit=v==y
                results[label]=dict(all=paired(hit,ref==y),multi_complete_bills=paired(hit[multi&fullbill[row]],(ref==y)[multi&fullbill[row]]),
                                    hard_fixed=int((hard&hit&~(ref==y)).sum()))
                if seed!='v131':
                    history.setdefault(label,[]).append(hit)
        if seed!='v131':
            history.setdefault('ref',[]).append(ref==y)
        plans[seed]=dict(results=results,folds=diagnostics)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,ref=ref,base=anchor,fold=fold,**bank)
        print(seed,{k:(v['all']['fixed'],v['all']['harmed']) for k,v in results.items() if k.startswith('transfer')},flush=True)
    ref=np.stack(history.pop('ref'))
    stable={}
    for name,v in history.items():
        h=np.stack(v)
        stable[name]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                          common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                          gain95_pp=interval((h.astype(float)-ref).mean(0),g))
    write(REPORT/'results.json',dict(scope=scope,plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
