"""Bound a common threshold as well as common nonnegative mixture weights."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from convex_portfolio_bound_n23 import ART,ROOT,conflicts,read_scores
from relational_error_corrector import write


def pair_bound(margins, incumbent):
    impossible=margins.max(1)<-1e-7
    variable=(margins.min(1)<=1.e-7)&~impossible
    left=np.flatnonzero(variable&~incumbent)
    right=np.flatnonzero(variable&incumbent)
    matching=maximum_bipartite_matching(csr_matrix(conflicts(margins[left],margins[right])),perm_type="column")
    take=np.flatnonzero(matching>=0)
    u,v=left[take],right[matching[take]]
    remaining=left[matching<0]
    graph=conflicts(margins[remaining],margins[remaining])
    extra=[]
    while graph.any():
        degree=graph.sum(1)
        i=np.flatnonzero(degree==degree[degree>0].min())[0]
        choices=np.flatnonzero(graph[i])
        j=choices[degree[choices].argmin()]
        extra.append((remaining[i],remaining[j]))
        graph[[i,j]]=False
        graph[:,[i,j]]=False
    if extra:
        a,b=np.asarray(extra).T
        u,v=np.r_[u,a],np.r_[v,b]
    # Verify a strictly negative separation certificate for every pair, including
    # the additional pairs in which the incumbent missed both passengers.
    a,b=margins[u],margins[v]
    delta=a-b
    roots=np.divide(-1.e-7-b,delta,out=np.zeros_like(b),where=abs(delta)>1.e-14)
    lower=np.maximum(0,np.max(np.where(delta < -1.e-14,roots,-np.inf),axis=1))
    upper=np.minimum(1,np.min(np.where(delta > 1.e-14,roots,np.inf),axis=1))
    lam=(lower+upper)/2
    assert np.all(np.max(lam[:,None]*a+(1-lam[:,None])*b,axis=1)<-5.e-8)
    assert len(np.unique(np.r_[u,v,np.flatnonzero(impossible)]))==len(u)*2+int(impossible.sum())
    return int(impossible.sum()),u,v,lam


def run():
    row,y,p=read_scores()
    # A nondecreasing decision function must classify at least one endpoint
    # incorrectly when every member ranks a true negative above a true positive.
    pos,neg=np.flatnonzero(y),np.flatnonzero(~y)
    dominance=np.stack([np.all(v+1e-7<p[neg],axis=1) for v in p[pos]])
    assign=maximum_bipartite_matching(csr_matrix(dominance),perm_type="column")
    at=np.flatnonzero(assign>=0)
    required=len(y)-int(np.floor(.85*len(y)))
    assert np.all(p[pos[at]]+1.e-7<p[neg[assign[at]]])
    folder=ART/"nov23_portfolio_threshold"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/"monotone.npz",positive=row[pos[at]],negative=row[neg[assign[at]]])
    bands=[]
    def visit(lo,hi,depth):
        # If the actual threshold t is in [lo,hi], the relaxed inequalities
        # here are necessary: positive p>=lo, negative p<=hi.
        margin=np.where(y[:,None],p-lo,hi-p)
        singles=int((margin.max(1)<-1e-7).sum())
        pairs=0
        if singles<required:
            singles,a,b,lam=pair_bound(margin,((p[:,0]>=(lo+hi)/2)==y))
            pairs=len(a)
        if singles+pairs<required and depth<3:
            mid=(lo+hi)/2
            visit(lo,mid,depth+1)
            visit(mid,hi,depth+1)
        else:
            key=f"interval_{len(bands):03d}"
            if pairs:
                np.savez_compressed(folder/f"{key}.npz",left=row[a],right=row[b],lambda_left=lam)
            bands.append(dict(lower=lo,upper=hi,unanimous_errors=singles,disjoint_pairs=pairs,
                              forced_mistakes=singles+pairs,key=key))
    edges=np.linspace(0,1,21)
    for lo,hi in zip(edges[:-1],edges[1:]):
        visit(float(lo),float(hi),0)
        print("thresholds through",round(hi,2),"certified segments",len(bands),flush=True)
    assert bands[0]["lower"]==0 and bands[-1]["upper"]==1
    assert all(a["upper"]==b["lower"] for a,b in zip(bands,bands[1:]))
    forced=min(v["forced_mistakes"] for v in bands)
    write(ROOT/"reports/nov23_portfolio_threshold/result.json",dict(rows=len(y),threshold_intervals=bands,
          minimum_mistakes_all_thresholds=forced,upper_accuracy_all_thresholds=1-forced/len(y),
          monotone_disjoint_dominated_pairs=len(at),upper_accuracy_any_monotone=1-len(at)/len(y),
          definition="Fixed combination of the ten repeated-OOF member means in convex_portfolio_bound_n23. Each threshold interval is certified including both endpoints. Weights and threshold are common across people."))
    print(forced,1-forced/len(y),len(at),flush=True)


if __name__=="__main__":
    run()
