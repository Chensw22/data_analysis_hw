"""Marginalize shared, discrete changes in cabin-side risk across the manifest."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit,logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_partition_evidence_n23 import ART,ROOT,SEEDS,historical,evaluate,write
from broad_geographic_transfer_n23 import load,BILLS

BINS=48
REPORT=ROOT/"reports/nov23_synchronized_cabin_segments"


def geometry(row):
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{p}.csv",usecols=["PassengerId","Cabin"]) for p in ("train","test")],ignore_index=True)
    group=raw.PassengerId.str.split("_").str[0].astype(int)
    berth=raw.Cabin.str.split("/",expand=True)
    channel=(berth[0]+"-"+berth[2]).fillna("unknown")
    known=pd.to_numeric(berth[1])
    # Use each observed berth once per deck-side irrespective of its occupancy.
    progress=group.rank(method="dense",pct=True).to_numpy()
    adjusted=progress.copy()
    for _,ids in channel.groupby(channel).groups.items():
        ids=np.asarray(ids)
        v=known.iloc[ids].to_numpy(float)
        levels=np.unique(v[np.isfinite(v)])
        if len(levels):
            f=np.isfinite(v)
            adjusted[ids[f]]=(np.searchsorted(levels,v[f])+.5)/len(levels)
    ordinal=np.minimum((BINS*progress).astype(int),BINS-1)
    aligned=np.minimum((BINS*adjusted).astype(int),BINS-1)
    displaced=aligned.copy()
    rng=np.random.default_rng(128)
    for _,ids in channel.groupby(channel).groups.items():
        displaced[ids]=(aligned[ids]+rng.integers(1,BINS))%BINS
    return {name:arr[row] for name,arr in (("ordinal",ordinal),("relative",aligned),("rotated",displaced))}


def states(frame):
    side=frame.Side.map({"P":-1,"S":1}).fillna(0).to_numpy()
    paid=frame[BILLS].gt(0).any(axis=1).astype(float).to_numpy()*2-1
    deck=frame.Deck.isin(["E","F","G"]).astype(float).to_numpy()*2-1
    solo=(frame.BookingSize==1).astype(float).to_numpy()*2-1
    kinds=np.column_stack([np.ones(len(side)),side,deck,deck*side,paid,side*paid,solo])
    return np.column_stack([np.zeros(len(side)),.85*kinds,-.85*kinds])


def posterior(grid,shift,base,y,t,q,hazard):
    prior=np.r_[.5,np.repeat(.5/(shift.shape[1]-1),shift.shape[1]-1)]
    z=logit(base.clip(.001,.999))
    ratios=shift[t]*y[t,None]+np.logaddexp(0,z[t,None])-np.logaddexp(0,z[t,None]+shift[t])
    bins=np.zeros((BINS,shift.shape[1]))
    np.add.at(bins,grid[t],ratios)
    sums=np.vstack([np.zeros(bins.shape[1]),bins.cumsum(axis=0)])
    # Independent chance of a boundary at every interior bin; a segment of
    # length L survives L-1 interior gaps, then ends or reaches the final bin.
    i,j=np.triu_indices(BINS)
    length=j-i+1
    evidence=sums[j+1]-sums[i]+np.log(prior)
    norm=logsumexp(evidence,axis=1)
    conditional=np.exp(evidence-norm[:,None])
    weight=np.full((BINS,BINS+1),-np.inf)
    weight[i,j+1]=norm+(length-1)*np.log1p(-hazard)+(j<BINS-1)*np.log(hazard)
    prefix=np.zeros(BINS+1)
    tail=np.zeros_like(prefix)
    for end in range(1,BINS+1):
        prefix[end]=logsumexp(prefix[:end]+weight[:end,end])
    for start in range(BINS-1,-1,-1):
        tail[start]=logsumexp(weight[start,start+1:]+tail[start+1:])
    np.testing.assert_allclose(tail[0],prefix[-1],atol=1e-8)
    prob=np.exp(prefix[i]+weight[i,j+1]+tail[j+1]-tail[0])
    diff=np.zeros_like(sums)
    np.add.at(diff,i,prob[:,None]*conditional)
    np.add.at(diff,j+1,-prob[:,None]*conditional)
    marginal=diff.cumsum(axis=0)[:-1]
    np.testing.assert_allclose(marginal.sum(axis=1),1,atol=1e-8)
    odds=expit(z[q,None]+shift[q])
    global_post=np.exp(sums[-1]+np.log(prior)-logsumexp(sums[-1]+np.log(prior)))
    return (marginal[grid[q]]*odds).sum(1),odds@global_post,float(prob.sum()-1)


def main():
    row,y,frame,g,_=load()
    original,target,masks=historical()
    np.testing.assert_array_equal(row,original)
    np.testing.assert_array_equal(y,target)
    solo=(frame.BookingSize==1).to_numpy()
    masks=dict(**masks,solo_repeat=solo&masks["repeat"],solo_hard=solo&masks["unresolved"])
    grids=geometry(row)
    shift=states(frame)
    report,history={},{}
    diagnostic={}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            fold,ref,base=z["fold"],z["ref"],z["base"]
        if not diagnostic:
            # Inject a shared side-dependent risk shift into synthetic outcomes;
            # these labels are never passed to a real-data fit.
            planted=np.where((grids["relative"]//8)%2,1.,-1.)*1.5*shift[:,2]/.85
            rng=np.random.default_rng(827)
            synth=rng.random(len(y))<expit(logit(base)+planted)
            simulation={k:base.copy() for k in grids}
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=324).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    for k,grid in grids.items():
                        simulation[k][q]=posterior(grid,shift,base,synth,t,q,.125)[0]
            for k,p in dict(base=base,**simulation).items():
                diagnostic[k]=dict(accuracy=float(((p>=.5)==synth).mean()),
                                   nll=float(-np.log(np.where(synth,p,1-p)).mean()))
        estimates={k:base.copy() for k in [*grids,"global"]}
        cuts={k:[] for k in grids}
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=324).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                for k,grid in grids.items():
                    p,pooled,n=posterior(grid,shift,base,y,t,q,.125)
                    estimates[k][q]=p
                    estimates["global"][q]=pooled
                    cuts[k].append(n)
        scores={"update_"+k:v for k,v in estimates.items()}
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks=masks)
        for k,(hit,gain) in hits.items():
            result[k]["solo_fixed_harmed"]=[int((solo&(gain==d)).sum()) for d in (1,-1)]
            if name!="v131":
                history.setdefault(k,[]).append(hit)
        report[name]=dict(models=result,boundary_count_mean={k:float(np.mean(v)) for k,v in cuts.items()})
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"results.json",dict(plans=report,stable=stable,synthetic=diagnostic,sizes={s:int(m.sum()) for s,m in masks.items()}))

if __name__=="__main__":
    with threadpool_limits(2):
        main()
