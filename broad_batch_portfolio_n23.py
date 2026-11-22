"""Use the observed disagreement mix of an unlabeled batch to choose an accuracy portfolio."""
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load,historical,strata
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import SEEDS
from broad_release_transfer_n23 import REPORT,matched,write


def rates(v,y,w,prior,strength):
    pos=v.T*w
    n=pos@(1-v)
    total=(pos*(2*y.astype(float)-1))@(1-v)
    return (total+strength*prior)/(n+strength)


def projected_skill(parts,rates):
    # For every pair, accumulate expected signed wins only where its two
    # members disagree. Row means solve the complete pairwise least squares fit.
    n=sum(len(v) for _,v in parts)
    directed=sum(((v.T@(1-v))*rates[c]) for c,v in parts)
    pair=(directed-directed.T)/n
    return pair.mean(1)


def policy(v,y,group,t,q,leaf,region,seed=503,reps=32):
    allcells=np.unique(leaf[q])
    unit,idx=np.unique(group[t],return_inverse=True)
    regs=np.unique(region[q])
    pres={}
    for r in regs:
        ids=q[region[q]==r]
        pres[r]=(ids,[(int(c),v[ids[leaf[ids]==c]]) for c in np.unique(leaf[ids])])
    out={k:np.zeros(len(q)) for k in ("direct","direction","roster")}
    # Preserve caller row order as q can be an arbitrary subset.
    where={k:i for i,k in enumerate(q)}
    rng=np.random.default_rng(seed)
    for rep in range(reps):
        w=rng.poisson(1,len(unit))[idx].astype(float)
        root=rates(v[t],y[t],w,0.,12.)
        values={}
        for c in allcells:
            ix=leaf[t]==c
            values[c]=rates(v[t][ix],y[t][ix],w[ix],root,16.)
        for r,(ids,parts) in pres.items():
            keep=region[t]==r
            # Direct selection uses training accuracy in this region, with
            # 16 passengers' worth of global training accuracy for shrinkage.
            hit=(v[t]==y[t,None]).astype(float)
            direct=(w[keep]@hit[keep]+16*(w@hit/w.sum()))/(w[keep].sum()+16)
            scores=(direct,projected_skill(parts,{c:root for c,_ in parts}),projected_skill(parts,values))
            at=[where[i] for i in ids]
            for k,val in zip(out,scores):
                winners=np.isclose(val,val.max(),atol=1e-12,rtol=0)
                out[k][at]+=v[ids][:,winners].mean(1)/reps
    return out


def trial(name,row,y,frame,g,oldvotes=None):
    fold,base,ref,dissent,names=bank(name,row,y)
    if oldvotes is None:
        votes=np.column_stack([ref,dissent^ref[:,None]]).astype(float)
    else:
        votes=np.column_stack([ref,oldvotes]).astype(float)
    paid=strata(frame)==2
    planet=np.asarray(frame.HomePlanet.astype("category").cat.codes)
    context=(planet*2+paid)*3+np.minimum(frame.BookingSize.to_numpy(),3)-1
    # The choice can differ across four broad prospective groups, each of which
    # has its own mixture of model disagreements.
    region=ref.astype(int)*2+paid
    eligible=ref==(base>=.5)
    maps={}
    for f in np.unique(fold):
        ids=np.flatnonzero((fold==f)&eligible)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            for batching,cells in (("all",np.zeros(len(y),int)),("regional",region)):
                scores=policy(votes,y,g,t,q,context,cells)
                for k,p in scores.items():
                    choice=np.where(np.isclose(p,.5,rtol=0,atol=1e-8),ref[q],p>.5)
                    maps.setdefault(batching+"_"+k,ref.copy())[q]=choice
    return ref,maps


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig);np.testing.assert_array_equal(y,labels)
    dev,labels,early=matched()
    np.testing.assert_array_equal(row,dev);np.testing.assert_array_equal(y,labels)
    oldvotes=np.column_stack([v["prob"]>=.5 for v in early])
    scopes=dict(**masks,repeat_paid=masks["repeat"]&(strata(frame)==2)&y,
                repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy())
    records,history={},{}
    for panel in [*map(str,SEEDS),"v131","v131_early"]:
        name=panel.split("_")[0]
        ref,maps=trial(name,row,y,frame,g,oldvotes if panel=="v131_early" else None)
        records[panel]={}
        for k,pred in maps.items():
            hit=pred==y
            gain=hit.astype(int)-(ref==y)
            records[panel][k]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                                   recovered={s:int(((gain>0)&m).sum()) for s,m in scopes.items()})
            if not panel.startswith("v131"):
                history.setdefault(k,[]).append(hit)
        print(panel,[(k,v["fixed"],v["harmed"]) for k,v in records[panel].items()],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in history.items()}
    write(REPORT/"batch_selection.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
