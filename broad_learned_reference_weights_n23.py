"""Learn donor similarity through the log loss of out-of-booking label transfer."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_booking_assignment_n23 import REPORT
from broad_error_vote_tables_n23 import bank
from broad_geographic_transfer_n23 import BILLS
from broad_matched_companions_n23 import data
from broad_partition_evidence_n23 import ART, SEEDS, evaluate, write
from broad_passenger_transport_n23 import manifest_coordinates
from released_correction_audit_n23 import development


def shortlist(views, g, q, t):
    chunks=[]
    for batch in np.array_split(q,max(1,int(np.ceil(len(q)/240)))):
        selected=[]
        for k,x in zip((36,16,16),views):
            cost=cdist(x[batch],x[t],"sqeuclidean")
            cost[g[batch,None]==g[t]]=np.inf
            ix=np.argsort(cost,axis=1,kind="stable")[:,:k]
            assert np.isfinite(np.take_along_axis(cost,ix,axis=1)).all()
            selected.append(t[ix])
        chunks.append(np.sort(np.column_stack(selected),axis=1))
    idx=np.concatenate(chunks)
    return idx,np.c_[np.ones(len(q),bool),np.diff(idx,axis=1)!=0]


def distances(blocks,q,idx):
    return np.stack([np.square(x[q,None]-x[idx]).sum(2) for x in blocks],axis=2).astype(np.float32)


def transfer(d,valid,donor_labels,coef):
    cost = d @ coef.astype(np.float32)
    cost = np.where(valid,cost,np.inf)
    w = np.exp(np.min(cost,axis=1,keepdims=True)-cost)
    w = w / w.sum(1,keepdims=True)
    local = np.einsum("ij,ij->i",w,donor_labels)
    return .98*local+.01,w,local


def train(d,valid,donor_labels,y):
    def objective(coef):
        p,w,local = transfer(d,valid,donor_labels,coef)
        loss = -(y@np.log(p)+(1-y)@np.log1p(-p))/len(y)
        slope = -.98*(p-y)/(p*(1-p)*len(y))
        edge = slope[:,None]*w*(donor_labels-local[:,None])
        gradient = np.einsum("ij,ijk->k",edge,d,optimize=True)
        ridge = .0004
        return loss+ridge*np.sum((coef-1)**2),gradient+2*ridge*(coef-1)
    sol = minimize(objective,np.ones(d.shape[2]),method="L-BFGS-B",jac=True,
                   bounds=[(0,20)]*d.shape[2],options={"maxiter":120,"ftol":2e-8})
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x


def run():
    row,y,local,g,_=load()
    frame,group,size,peers,fields=data(row)
    np.testing.assert_array_equal(group.astype(int),g)
    labels,deployed,masks=development()
    np.testing.assert_array_equal(labels,y)
    missed=np.logical_and.reduce([p!=y for p in deployed.values()])
    state=strata(local)
    core=missed & (((state==2)&y)|((state==0)&~y))
    masks={**masks,"miss":missed,"core":core,"core_solo":core&(size==1),"core_party":core&(size>1)}
    assert [masks[k].sum() for k in ("miss","core","core_solo","core_party")]==[941,696,435,261]
    columns=list(local.columns)
    own=[manifest_coordinates(local[[k]]) for k in columns]
    own += [np.c_[np.log1p(local[k].fillna(0))/7,local[k].gt(0)] for k in BILLS]
    peer=[peers[:,ix]/np.sqrt(len(ix)) for ix in fields]
    # Reassign the full peer profile to a passenger of equal booking size. Keep
    # solos intact and prevent any passenger from receiving their own booking.
    rng=np.random.default_rng(772)
    donor=np.arange(len(y))
    for sz in np.unique(size[size>1]):
        idx=np.flatnonzero(size==sz)
        shuffled=idx[np.argsort(g[idx],kind="stable")]
        shift=sz+rng.integers(0,len(idx)-2*sz+1)
        donor[shuffled]=np.roll(shuffled,shift)
        assert np.all(g[donor[idx]]!=g[idx])
    sham=[x[donor] for x in peer]
    views=[np.column_stack(own),np.column_stack([own[i] for i,k in enumerate(columns) if k not in BILLS]),
           np.column_stack([own[i] for i,k in enumerate(columns) if k in BILLS]+own[len(columns):])]
    records,history,importance={},{},{}
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,_,_=bank(name,row,y)
        scores={};importance[name]=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            tr,train_valid=shortlist(views,g,t,t)
            qr,test_valid=shortlist(views,g,q,t)
            td=distances(own,t,tr)
            qd=distances(own,q,qr)
            targets=y[tr].astype(np.float32)
            truth=y[t].astype(float)
            weights={}
            for key,ctx in (("own",None),("peers",peer),("sham",sham)):
                train_d=td if ctx is None else np.concatenate([td,distances(ctx,t,tr)],axis=2)
                test_d=qd if ctx is None else np.concatenate([qd,distances(ctx,q,qr)],axis=2)
                coef=train(train_d,train_valid,targets,truth)
                for mode,w in [(key,coef)]+([("equal",np.ones_like(coef))] if key=="own" else []):
                    pred=transfer(test_d,test_valid,y[qr].astype(np.float32),w)[0]
                    scores.setdefault(mode,np.full(len(y),np.nan))[q]=pred
                weights[key]=coef.tolist()
            importance[name].append(weights)
        updated={}
        for key in scores:
            z=np.c_[logit(base.clip(.001,.999)),logit(scores[key])]
            pred=base.clip(.001,.999).copy()
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    pred[q]=LogisticRegression(C=.5,max_iter=500).fit(z[t],y[t]).predict_proba(z[q])[:,1]
            updated["update_"+key]=pred
        scores.update(updated)
        records[name],hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        if name!="v131":
            for key,(hit,_) in hits.items():
                history.setdefault(key,[]).append(hit)
        out=ART/"nov23_learned_reference_weights"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out/f"{name}.npz",row=row,y=y,base=base,ref=ref,fold=fold,**scores)
        print(name,[(k,r["fixed"],r["harmed"]) for k,r in records[name].items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"learned_weights.json",dict(plans=records,stable=stable,weights=importance,
          columns=columns+[n+"_amount" for n in BILLS],extra_peer_blocks=["route","sleep_vip","cabin","age","bills"]))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
