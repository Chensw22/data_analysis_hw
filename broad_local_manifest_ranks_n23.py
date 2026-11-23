"""Give a frozen tabular foundation model environment-relative ranks from other training bookings."""
import hashlib
import numpy as np
import torch
from scipy.special import logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_pretrained_spending_views_n23 import predict
from tabicl_routed_context_n23 import DIGEST, WEIGHT

REPORT=ROOT/'reports/nov23_field_robustness'


def reference_cdf(ref, query):
    ref=np.sort(ref[np.isfinite(ref)])
    if not len(ref):
        return np.full(len(query),.5)
    return (.5+(.5*np.searchsorted(ref,query,side='left')+.5*np.searchsorted(ref,query,side='right')))/(1+len(ref))


def environments(region, group, train, random):
    edges=np.unique(np.quantile(group[train],[.25,.5,.75]))
    quarter=np.searchsorted(edges,group)
    zones=np.unique(region,return_inverse=True)[1]
    if random:
        # Shuffle environment pairs as whole bookings, preserving the original multiset across bookings.
        keys,first,inv=np.unique(group,return_index=True,return_inverse=True)
        rng=np.random.default_rng(194)
        perm=rng.permutation(len(keys))
        zones=zones[first][perm][inv]
        quarter=quarter[first][perm][inv]
    return zones,quarter


def neighborhood_ranks(frame,t,q,env):
    result=[]
    z,e=env
    query=np.r_[t,q]
    for col in [*BILLS,'Age','CabinNumber']:
        v=frame[col].to_numpy(float)
        keep=np.isfinite(v)&((v>0) if col in BILLS else True)
        source=t[keep[t]]
        u=reference_cdf(v[source],v[query])
        estimates=[]
        for group in [z,e,z*(e.max()+1)+e]:
            ans=u.copy()
            for k in np.unique(group[query]):
                ii=group[query]==k
                local=source[group[source]==k]
                w=len(local)/(len(local)+48.)
                if w:
                    ans[ii]=(1-w)*u[ii]+w*reference_cdf(v[local],v[query[ii]])
            estimates.append(ans)
        value=np.mean(estimates,axis=0)-u
        # A tie in the zero mass conveys no information about an amount among paying passengers.
        value[~keep[query]]=0
        result.append(value)
    out=frame.copy()
    for name,val in zip([*BILLS,'Age','CabinNumber'],result):
        a=np.full(len(frame),np.nan)
        a[query]=val
        out[name+'_environment_rank_change']=a
    return out


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,frame,g,region=load()
    frame=frame.drop(columns='BookingSize')
    root=ART/'nov23_local_manifest_ranks'
    root.mkdir(exist_ok=True)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        common=(z['votes']!=y[None,:,None]).all((0,2))
        correct=(z['votes']==y[None,:,None]).all((0,2))
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        with np.load(ART/f'nov23_crossfamily_unlabelled_distillation/{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('fold',fold),('ref',ref)]:np.testing.assert_array_equal(z[k],v)
            ordinary=z['teacher']
        scores={'original':ordinary}
        for f in np.unique(fold):
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[fold==f]))
            q=np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            for name,random in [('real',False),('shuffled',True)]:
                env=environments(region,g,t,random)
                table=neighborhood_ranks(frame,t,q,env)
                scores.setdefault(name,np.full(len(y),np.nan))[q]=predict(table,y,t,q,f)
        votes={'reference':ref}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            votes[k]=p>=.5
            votes['quarter_'+k]=np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*p>=.5,ref)
            votes['offset_'+k]=np.where(ref==(op[:,0]>=.5),logit(op[:,0].clip(.002,.998))
                                  +logit(p.clip(.002,.998))-logit(ordinary.clip(.002,.998))>=0,ref)
        h={k:v==y for k,v in votes.items()}
        plans[seed]={k:dict(reference=paired(v,ref==y),native=paired(v,h['original'])) for k,v in h.items()}
        if seed!='v131':
            for k,v in h.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(root/f'{seed}.npz',row=row,y=y,fold=fold,base=op[:,0],ref=ref,**scores)
        print(seed,plans[seed]['real'],plans[seed]['quarter_real'],flush=True)
    h={k:np.stack(v) for k,v in hist.items()}
    stats={k:dict(common=int((v.all(0)&common).sum()),common_right=int((v.all(0)&correct).sum()),
                  vs_reference_ci95_pp=interval((v.astype(float)-h['reference']).mean(0),g)) for k,v in h.items()}
    write(REPORT/'environment_ranks.json',dict(plans=plans,grouped=stats))


if __name__=='__main__':
    with threadpool_limits(2):main()
