"""Certify how well global convex mixtures and arbitrary monotone combinations can fit these scores."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,historical,interval,write

OUT=ART/"nov23_broad_ordered_ensemble"
REPORT=ROOT/"reports/nov23_broad_ordered_ensemble"


def dominated(left,right):
    result=np.ones((len(left),len(right)),bool)
    for a,b in zip(left.T,right.T):
        result &= a[:,None]<=b
    return result


def monotone_rule(x,y):
    pos,neg=x[y],x[~y]
    graph=csr_matrix(dominated(pos,neg))
    mates=maximum_bipartite_matching(graph,perm_type="column")
    matched=np.flatnonzero(mates>=0)
    inv=np.full(len(neg),-1)
    inv[mates[matched]]=matched
    left_seen=mates<0
    right_seen=np.zeros(len(neg),bool)
    stack=np.flatnonzero(left_seen).tolist()
    # Alternating paths from unmatched positives give a minimum vertex cover.
    while stack:
        i=stack.pop()
        choices=graph.indices[graph.indptr[i]:graph.indptr[i+1]]
        choices=choices[(choices!=mates[i])&~right_seen[choices]]
        right_seen[choices]=True
        partners=inv[choices]
        assert np.all(partners>=0)
        new=partners[~left_seen[partners]]
        left_seen[new]=True
        stack.extend(new)
    assert (~left_seen).sum()+right_seen.sum()==len(matched)
    return pos[left_seen],neg[~right_seen],len(matched)


def apply_rule(good,bad,query):
    force_pos=dominated(good,query).any(0)
    force_neg=dominated(query,bad).any(1)
    assert not (force_pos&force_neg).any()
    return force_pos|(~force_neg&(query[:,0]>=.5))


def threshold(score,y):
    # Evaluate only realizable thresholds, including an all-negative vote.
    values,inv=np.unique(score,return_inverse=True)
    changes=np.bincount(inv,weights=2*y.astype(int)-1)
    ncorrect=np.r_[(~y).sum()+np.cumsum(changes[::-1])[::-1],(~y).sum()]
    values=np.r_[values,np.nextafter(values[-1],np.inf)]
    best=np.flatnonzero(ncorrect==ncorrect.max())
    j=best[np.argmin(abs(values[best]-.5))]
    return int(round(ncorrect[j])),values[j]


def grid_fit(x,y):
    best=-1
    for indices in combinations_with_replacement(range(x.shape[1]),6):
        w=np.bincount(indices,minlength=x.shape[1])/6
        correct,cut=threshold(x@w,y)
        if correct>best:
            best=correct
            choice=(w,cut)
    return choice


def convex_bound(x,y,locked_errors,total):
    # Any weights inside a simplex give each row a score between the values
    # at its vertices. Sweep all thresholds over these per-row intervals.
    target=int(np.floor(.85*total))+1
    budget=total-target
    leaves=[]
    def visit(vertices,projections,path):
        high=projections[y].max(1)+1e-11
        low=projections[~y].min(1)-1e-11
        events=np.r_[np.ones(len(high),int),-np.ones(len(low),int)]
        order=np.lexsort((events,np.r_[high,low]))
        cost=locked_errors+len(low)+min(0,int(np.cumsum(events[order]).min()))
        if cost>budget or len(path)>=18:
            leaves.append((path,cost))
            return
        a,b=np.unravel_index(np.argmax(((vertices[:,None]-vertices)**2).sum(2)),(len(vertices),len(vertices)))
        mid=(vertices[a]+vertices[b])/2
        col=(projections[:,a]+projections[:,b])/2
        for j,node in enumerate((a,b)):
            child=vertices.copy()
            scores=projections.copy()
            child[node]=mid
            scores[:,node]=col
            visit(child,scores,path+str(j))
    visit(np.eye(x.shape[1]),x,"")
    assert abs(sum(2.**-len(path) for path,_ in leaves)-1)<1e-10
    minimum=min(c for _,c in leaves)
    return dict(upper_accuracy=1-minimum/total,certified_below_target=minimum>budget,
                minimum_errors=minimum,leaves=len(leaves),maximum_depth=max(len(s) for s,_ in leaves),
                certificates=leaves)


def run():
    row,y,_,groups,_=load()
    orig,target,masks=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(target,y)
    records,history={},{}
    OUT.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        files=[ART/f"nov23_broad_passenger_differences/balanced/{name}.npz",ART/f"nov23_broad_count_supervision/{name}.npz",
               ART/f"nov23_broad_latent_scale/{name}.npz"]
        sources=[]
        for f,keys in zip(files,[("native_pair","update_pair"),("native_similar","update_similar"),("native_real","update_real")]):
            with np.load(f) as z:
                for k,v in (("row",row),("y",y)):
                    np.testing.assert_array_equal(z[k],v)
                if sources:
                    for k,v in (("fold",fold),("base",base),("ref",ref)):
                        np.testing.assert_array_equal(z[k],v)
                fold,base,ref=z["fold"],z["base"],z["ref"]
                sources.append([z[k] for k in keys])
        locked=ref!=(base>=.5)
        lost=int((locked&(ref!=y)).sum())
        records[name]={}
        bank={}
        for j,kind in enumerate(("native","adjusted")):
            x=np.column_stack([base,*[v[j] for v in sources]])
            assert np.isfinite(x).all()
            # The oracle may inspect all labels for a mathematical upper bound.
            # Evaluation below refits each actual rule without the query group.
            a,b,n=monotone_rule(x[~locked],y[~locked])
            exact=apply_rule(a,b,x[~locked])
            assert int((exact!=y[~locked]).sum())==n
            bound=convex_bound(x[~locked],y[~locked],lost,len(y))
            w,c=grid_fit(x[~locked],y[~locked])
            achievable=int(((x[~locked]@w>=c)!=y[~locked]).sum())+lost
            assert max(n+lost,bound["minimum_errors"])<=achievable
            scored={k:ref.copy() for k in ("monotone","convex")}
            for f in np.unique(fold):
                outer=np.flatnonzero((fold==f)&~locked)
                for t,q in GroupKFold(3,shuffle=True,random_state=823).split(outer,groups=groups[outer]):
                    t,q=outer[t],outer[q]
                    assert not np.intersect1d(groups[t],groups[q]).size
                    pos,neg,_=monotone_rule(x[t],y[t])
                    scored["monotone"][q]=apply_rule(pos,neg,x[q])
                    weight,cut=grid_fit(x[t],y[t])
                    scored["convex"][q]=x[q]@weight>=cut
            result={}
            for k,pred in scored.items():
                hit=pred==y
                gain=hit.astype(int)-(ref==y)
                bank[kind+"_"+k]=pred
                result[k]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                               net_by_fold=[int(gain[fold==f].sum()) for f in np.unique(fold)])
                if name!="v131":
                    history.setdefault(kind+"_"+k,[]).append((hit,gain))
            records[name][kind]=dict(monotone_oracle_accuracy=1-(n+lost)/len(y),minimum_monotone_errors=n+lost,
                    convex_certificate=bound,attainable_convex_oracle_accuracy=1-achievable/len(y),heldout=result)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        write(REPORT/"results.json",dict(plans=records))
        print(name,[(k,v["convex_certificate"]["upper_accuracy"],v["monotone_oracle_accuracy"]) for k,v in records[name].items()],flush=True)
    stable={}
    for k,values in history.items():
        ok=np.logical_and.reduce([h for h,_ in values])
        stable[k]={s:int((m&ok).sum()) for s,m in masks.items()}
        stable[k]["paired_gain_ci95_pp"]=interval(np.mean([d for _,d in values],0),groups)
    write(REPORT/"results.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
