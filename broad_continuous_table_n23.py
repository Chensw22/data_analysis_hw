"""Fit registration-dependent changes in the onset of numeric passenger effects."""
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS, load
from broad_manifest_vector_ranks_n23 import calibrate
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes

REPORT = ROOT / "reports/nov23_continuous_table"


def design(frame, train):
    context = frame[["HomePlanet", "CryoSleep", "Destination", "Deck", "Side", "VIP"]].copy()
    context["size"] = frame.BookingSize.clip(upper=5).astype(str)
    for first, second in (("HomePlanet", "CryoSleep"), ("Deck", "HomePlanet"), ("Deck", "Side")):
        context[first + second] = context[first] + "|" + context[second]
    gate = pd.get_dummies(context, dtype=float).to_numpy()
    values = frame[[*BILLS, "Age", "CabinNumber"]].to_numpy(float)
    observed = np.isfinite(values)
    applicable = observed.copy()
    applicable[:, :len(BILLS)] &= values[:, :len(BILLS)] > 0
    ranked = np.zeros_like(values)
    for j, v in enumerate(values.T):
        ref = np.sort(v[train][applicable[train, j]])
        left = np.searchsorted(ref, v, side="left")
        right = np.searchsorted(ref, v, side="right")
        ranked[:, j] = np.where(applicable[:, j], (left + right + 1)/(2*len(ref)+2), .5)
    main = np.c_[np.ones(len(frame)), gate, ~observed, applicable, ranked*applicable]
    return main, gate, ranked, applicable


KNOTS = np.linspace(.05, .95, 7)
WIDTH = .055


def forward(inputs, idx, par, amplitude, location, gradient=None):
    h,g,loc,x,obs= [v[idx] for v in inputs]
    main,coef,amp,delta=par
    a=np.tanh(g@amp) if amplitude else np.zeros_like(x)
    d=np.tanh(loc@delta) if location else np.zeros_like(x)
    height=1+.8*a
    knots=expit((x[:,:,None]-.35*d[:,:,None]-KNOTS)/WIDTH)
    curve=np.sum(knots*coef[None,:,:],axis=2)
    z=h@main + np.sum(obs*height*curve,axis=1)
    if gradient is None:
        return z
    residual=(expit(z)-gradient)/len(idx)
    dz=residual[:,None]*obs
    gmain=h.T@residual
    gcoef=np.einsum("ij,ijk->jk",dz*height,knots)
    gamp=g.T@(dz*curve*.8*(1-a*a)) if amplitude else np.zeros_like(amp)
    slope=(knots*(1-knots)*coef[None,:,:]).sum(axis=2)
    gdelta=loc.T@(dz*height*slope*(-.35/WIDTH)*(1-d*d)) if location else np.zeros_like(delta)
    return [gmain,gcoef,gamp,gdelta]


def train(inputs,y,t,q,mode):
    h,g,_,x,_=inputs
    par=[np.zeros(h.shape[1]),np.zeros((x.shape[1],len(KNOTS))),
         np.zeros((g.shape[1],x.shape[1])),np.zeros((g.shape[1],x.shape[1]))]
    m=[np.zeros_like(v) for v in par]
    var=[np.zeros_like(v) for v in par]
    rng=np.random.default_rng(814)
    step=0
    for epoch in range(36):
        for batch in np.array_split(rng.permutation(t),12):
            step+=1
            deriv=forward(inputs,batch,par,mode!="fixed",mode in ("location","shuffled"),y[batch])
            for j,(p,d) in enumerate(zip(par,deriv)):
                d+=([.0003,.0006,.002,.004][j])*p
                m[j]=.9*m[j]+.1*d
                var[j]=.999*var[j]+.001*d*d
                p-=.018*(m[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1.e-8)
    return expit(forward(inputs,q,par,mode!="fixed",mode in ("location","shuffled")))


def main():
    row,y,frame,groups,_=load()
    masks=scopes(y,frame)
    dest=ART/"nov23_continuous_table"
    dest.mkdir(exist_ok=True)
    histories,refs={},[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as cache:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(cache[k],v)
            fold,base=cache["fold"],cache["base"]
        scores={}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t],groups[q]).size
            h,g,x,observed=design(frame,t)
            sham=g.copy()
            rng=np.random.default_rng(572)
            for ix in (t,q):
                sham[ix]=g[rng.permutation(ix)]
            for mode in ("fixed","amplitude","location","shuffled"):
                inputs=(h,g,sham if mode=="shuffled" else g,x,observed)
                scores.setdefault(mode,np.full(len(y),np.nan))[q]=train(inputs,y,t,q,mode)
        for k in list(scores):
            assert np.isfinite(scores[k]).all()
            scores["combined_"+k]=calibrate(base,scores[k],y,groups,fold)
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in scores.items():histories.setdefault(k,[]).append((p>=.5)==y)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**scores)
        print(seed,[(k,int((~ref&v[-1]).sum()),int((ref&~v[-1]).sum())) for k,v in histories.items()],flush=True)
    write(REPORT/"results.json",{k:summarize(np.stack(v),np.stack(refs),masks) for k,v in histories.items()})


if __name__=="__main__":
    with threadpool_limits(2):
        main()
