"""Learn nonlinear representations by comparing opposite outcomes within each spending state."""
import numpy as np
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from broad_bill_exception_n23 import load, strata
from broad_booking_assignment_n23 import REPORT
from broad_error_vote_tables_n23 import bank
from broad_geographic_transfer_n23 import BILLS
from broad_matched_companions_n23 import data
from broad_partition_evidence_n23 import ART, SEEDS, evaluate, write
from broad_passenger_transport_n23 import manifest_coordinates
from released_correction_audit_n23 import development


def density_scores(z, ref, allowed):
    # With a normalized embedding, dot products differ from negative squared
    # distances by a constant. Average separately within each labeled pool.
    logits = torch.einsum("qd,ckd->qck",z,ref)*10
    count = allowed.sum(-1)
    assert torch.all(count>0)
    evidence = torch.logsumexp(logits.masked_fill(~allowed,-torch.inf),dim=-1)-torch.log(count)
    return evidence[:,1]-evidence[:,0]


def fit(x,y,state,groups,t,q,balanced,use_refs):
    torch.manual_seed(839)
    rng=np.random.default_rng(299)
    net=nn.Sequential(nn.Linear(x.shape[1],64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,24))
    head=nn.Linear(24,3)
    opt=torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=.002,weight_decay=.005)
    pools={(s,v):t[(state[t]==s)&(y[t]==v)] for s in range(3) for v in (0,1)}
    counts=np.array([[len(pools[s,v]) for v in (0,1)] for s in range(3)])
    assert counts.min()>5
    prior=torch.tensor(np.log((counts[:,1]+1)/(counts[:,0]+1)),dtype=torch.float32)
    for _ in range(180):
        losses=[]
        for s in range(3):
            # The two arms see the same focal rows and reference samples. Weight
            # focal losses by natural frequencies only in the population arm.
            focal=np.r_[rng.choice(pools[s,0],20),rng.choice(pools[s,1],20)]
            references=np.stack([rng.choice(pools[s,v],40) for v in (0,1)])
            query=net(x[focal])
            if use_refs:
                other=F.normalize(net(x[references]),dim=-1)
                allowed=torch.from_numpy(groups[focal,None,None]!=groups[references][None,:,:])
                score=density_scores(F.normalize(query,dim=-1),other,allowed)
            else:
                score=head(query)[:,s]
            labels=torch.from_numpy(y[focal].astype(np.float32))
            if balanced:
                loss=F.binary_cross_entropy_with_logits(score,labels)
            else:
                w=torch.tensor(counts[s]/counts.sum(),dtype=torch.float32)[labels.long()]
                loss=(w*F.binary_cross_entropy_with_logits(score+prior[s],labels,reduction="none")).mean()*6
            losses.append(loss)
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        nn.utils.clip_grad_norm_(net.parameters(),3.)
        opt.step()
    out=np.full(len(q),np.nan)
    net.eval()
    with torch.no_grad():
        for s in range(3):
            idx=np.flatnonzero(state[q]==s)
            if not len(idx):
                continue
            z=net(x[q[idx]])
            if use_refs:
                z=F.normalize(z,dim=-1)
                evidence=[]
                for v in (0,1):
                    part=pools[s,v]
                    assert not np.intersect1d(groups[q[idx]],groups[part]).size
                    ref=F.normalize(net(x[part]),dim=-1)
                    evidence.append(torch.logsumexp((z@ref.T)*10,dim=1)-np.log(len(part)))
                score=evidence[1]-evidence[0]
            else:
                score=head(z)[:,s]
            out[idx]=torch.sigmoid((score+prior[s]).double()).numpy().clip(1e-12,1-1e-12)
    assert np.isfinite(out).all()
    return out


def run():
    row,y,local,g,_=load()
    frame,groups,size,peers,_=data(row)
    np.testing.assert_array_equal(groups.astype(int),g)
    labels,deployed,masks=development()
    np.testing.assert_array_equal(labels,y)
    state=strata(local)
    miss=np.logical_and.reduce([p!=y for p in deployed.values()])
    core=miss&(((state==2)&y)|((state==0)&~y))
    masks={**masks,"release":miss,"core":core,"core_solo":core&(size==1),"core_party":core&(size>1)}
    assert [masks[k].sum() for k in ("release","core","core_solo","core_party")]==[941,696,435,261]
    own=manifest_coordinates(local)*np.sqrt(len(local.columns))
    amounts=np.c_[np.log1p(local[BILLS].fillna(0))/7,local[BILLS].gt(0)]
    x=torch.from_numpy(np.c_[own,amounts,peers].astype(np.float32))
    history,records={},{}
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,_,_=bank(name,row,y)
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for k,balanced,refs in (("pop_refs",False,True),("balanced_refs",True,True),("balanced_direct",True,False)):
                scores.setdefault(k,np.full(len(y),np.nan))[q]=fit(x,y,state,g,t,q,balanced,refs)
        updated={}
        for k,score in scores.items():
            z=np.c_[logit(base.clip(.001,.999)),logit(score.clip(.001,.999))]
            pred=np.full(len(y),np.nan)
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    pred[q]=LogisticRegression(C=.5,max_iter=500).fit(z[t],y[t]).predict_proba(z[q])[:,1]
            updated["update_"+k]=pred
        scores.update(updated)
        records[name],hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        if name!="v131":
            for k,(hit,_) in hits.items():
                history.setdefault(k,[]).append(hit)
        dest=ART/"nov23_spend_reference_network"
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/f"{name}.npz",row=row,y=y,fold=fold,ref=ref,base=base,**scores)
        print(name,[(k,v["fixed"],v["harmed"]) for k,v in records[name].items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&mask).sum()) for s,mask in masks.items()} for k,v in history.items()}
    write(REPORT/"reference_network.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
