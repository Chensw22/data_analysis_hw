"""Compare independent neighbors and population-balanced transport of training passenger labels."""
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from audit_disputed_passenger_separation_n23 import REPORT
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import coordinates,cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def graph(x,z):
    # All edges are between source (labeled) and target (unlabeled) passengers.
    k=48
    ds,js=cKDTree(z).query(x,k=k)
    dt,it=cKDTree(x).query(z,k=k)
    s=np.maximum(ds[:,-1],.08)
    t=np.maximum(dt[:,-1],.08)
    forward=np.exp(-ds**2/(s[:,None]*t[js]))
    reverse=np.exp(-dt**2/(t[:,None]*s[it]))
    shape=(len(x),len(z))
    a=csr_matrix((forward.ravel(),(np.repeat(np.arange(len(x)),k),js.ravel())),shape)
    b=csr_matrix((reverse.ravel(),(it.ravel(),np.repeat(np.arange(len(z)),k))),shape)
    return a.maximum(b).tocsr()


def matching(kernel,y,anchor):
    ns,nt=kernel.shape
    forward=kernel
    backward=kernel.T.tocsr()
    # The small complete-bipartite component allows rare unmatched profiles to receive mass.
    background=.18*kernel.sum()/ns/nt
    def incoming(source):
        return backward@source+background*source.sum()
    def outgoing(target):
        return forward@target+background*target.sum()
    native_pos=incoming(y.astype(float))
    native_neg=incoming((~y).astype(float))
    native=native_pos/(native_pos+native_neg)
    out={'independent':native}
    notes={}
    for name,initial,target_rate in [('balanced_geometry',native,y.mean()),('balanced_anchor',anchor,y.mean()),
                                     ('preserve_anchor_rate',anchor,anchor.mean())]:
        target_rows=(nt/ns)*np.where(y,target_rate/y.mean(),(1-target_rate)/(1-y.mean()))
        wp=initial/native
        wn=(1-initial)/(1-native)
        np.testing.assert_allclose(wp*native_pos/(wp*native_pos+wn*native_neg),initial)
        u=np.ones(ns)
        for step in range(500):
            pos=wp*incoming(u*y)
            neg=wn*incoming(u*~y)
            v=1/(pos+neg)
            rows=u*np.where(y,outgoing(wp*v),outgoing(wn*v))
            residual=np.max(abs(rows/target_rows-1))
            if residual<1e-6:
                break
            u*=target_rows/rows
        else:
            raise RuntimeError(f'Transport margin residual {residual}')
        prob=pos*v
        assert np.all((prob>=0)&(prob<=1))
        np.testing.assert_allclose(prob.mean(),target_rate,atol=2e-6)
        out[name]=prob
        notes[name]=dict(iterations=step+1,max_relative_marginal_error=float(residual))
    logodds=logit(anchor)
    intercept=brentq(lambda b:expit(logodds+b).mean()-y.mean(),-25,25)
    out['global_prior']=expit(logodds+intercept)
    return out,notes


def main():
    row,y,frame,groups,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    folder=ART/'nov23_balanced_passenger_matching'
    folder.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z['fold'],z['base'],z['ref']
        bank,notes={},[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[q],groups[t]).size
            x,_=coordinates(frame,t)
            values,info=matching(graph(x[t],x[q]),y[t],base[q].clip(.00001,.99999))
            for k,p in values.items():
                bank.setdefault(k,np.empty(len(y)))[q]=p
            notes.append(info)
        candidates={**bank,'quarter_anchor':.75*base+.25*bank['balanced_anchor'],
                    'quarter_geometry':.75*base+.25*bank['balanced_geometry'],
                    'quarter_preserve_rate':.75*base+.25*bank['preserve_anchor_rate'],
                    'quarter_independent':.75*base+.25*bank['independent'],
                    'quarter_global':.75*base+.25*bank['global_prior']}
        active=ref==(base>=.5)
        hits={k:np.where(active,p>=.5,ref)==y for k,p in candidates.items()}
        hits['reference']=ref==y
        metrics={k:paired(h,ref==y) for k,h in hits.items()}
        plans[seed]=dict(metrics=metrics,diagnostics=notes)
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        np.savez_compressed(folder/f'{seed}.npz',row=row,y=y,ref=ref,base=base,fold=fold,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),groups))
    write(REPORT/'balanced_matching.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
