"""Certify a lower bound on mistakes for all fixed convex weightings of ten grouped OOF models."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,FAMILIES,ROOT,SEEDS,read_panel
from relational_error_corrector import write


def read_scores():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,_,_=read_panel()
    matrices=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        members=[base]
        for source,key in [*FAMILIES.values(),("simplex_likelihood","budget_prior")]:
            with np.load(ART/f"nov23_{source}/{seed}.npz") as z:
                verify(z,row,y,fold)
                members.append(z[key])
        matrices.append(np.column_stack(members))
    return row,y,np.mean(matrices,axis=0)


def conflicts(a,b):
    # If a convex combination of these two rows' signed margin vectors is
    # strictly negative in EVERY coordinate, no shared portfolio can classify
    # both correctly. Solve for the feasible scalar mixture on [0,1].
    tolerance=1e-7
    graph=np.zeros((len(a),len(b)),dtype=bool)
    for i,margin in enumerate(a):
        diff=margin-b
        active=abs(diff)>1e-14
        cross=np.divide(-tolerance-b,diff,out=np.zeros_like(b),where=active)
        low=np.maximum(0,np.max(np.where(diff < -1e-14,cross,-np.inf),axis=1))
        high=np.minimum(1,np.min(np.where(diff > 1e-14,cross,np.inf),axis=1))
        mix=(low+high)/2
        possible=(low<high)&np.all(active|(b < -tolerance),axis=1)
        # Independently certify every edge using direct convex arithmetic.
        ids=np.flatnonzero(possible)
        graph[i,ids]=np.max(mix[ids,None]*margin+(1-mix[ids,None])*b[ids],axis=1)<-tolerance/2
    return graph


def run():
    row,y,p=read_scores()
    assert np.isfinite(p).all()
    sign=np.where(y,1.,-1.)[:,None]
    margins=sign*(p-.5)
    impossible=margins.max(1)<-1e-7
    variable=(margins.min(1)<=1e-7)&~impossible
    correct=(p[:,0]>=.5)==y
    lhs=np.flatnonzero(variable&~correct)
    rhs=np.flatnonzero(variable&correct)
    graph=conflicts(margins[lhs],margins[rhs])
    matching=maximum_bipartite_matching(csr_matrix(graph),perm_type="column")
    paired=np.flatnonzero(matching>=0)
    assert len(np.unique(matching[paired]))==len(paired)
    left=lhs[paired];right=rhs[matching[paired]]
    opposite_pairs=len(left)
    unused=np.setdiff1d(lhs,left)
    remaining=conflicts(margins[unused],margins[unused])
    more=[]
    while remaining.any():
        degree=remaining.sum(1)
        a=np.flatnonzero(degree==degree[degree>0].min())[0]
        options=np.flatnonzero(remaining[a])
        b=options[degree[options].argmin()]
        more.append([unused[a],unused[b]])
        remaining[[a,b],:]=False
        remaining[:,[a,b]]=False
    if more:
        plus=np.asarray(more)
        left=np.r_[left,plus[:,0]]
        right=np.r_[right,plus[:,1]]
    # Explicit separating coefficients and worst margin are saved for every
    # disjoint pair so the numerical certificate needs no optimization to verify.
    a,b=margins[left],margins[right]
    diff=a-b
    roots=np.divide(-1e-7-b,diff,out=np.zeros_like(b),where=abs(diff)>1e-14)
    lo=np.maximum(0,np.max(np.where(diff < -1.e-14,roots,-np.inf),axis=1))
    hi=np.minimum(1,np.min(np.where(diff > 1.e-14,roots,np.inf),axis=1))
    lam=(lo+hi)/2
    largest=np.max(lam[:,None]*a+(1-lam[:,None])*b,axis=1)
    assert np.all(largest<-5.e-8)
    assert np.unique(np.r_[left,right,np.flatnonzero(impossible)]).size==2*len(left)+int(impossible.sum())
    out=ART/"nov23_convex_portfolio_bound"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/"certificate.npz",left=row[left],right=row[right],lambda_left=lam,
                        worst_margin=largest,unanimous_wrong=row[impossible])
    unavoidable=int(impossible.sum())+len(left)
    bound=len(y)-unavoidable
    summary=dict(rows=len(y),families=["base",*FAMILIES,"budget_prior"],
                 all_member_predictions_wrong=int(impossible.sum()),disputed_baseline_misses=len(lhs),
                 disputed_baseline_correct=len(rhs),incompatible_opposite_pairs=int(graph.sum()),opposite_pairs_used=opposite_pairs,
                 additional_disjoint_pairs=len(more),
                 disjoint_incompatible_pairs=len(left),forced_mistakes=unavoidable,
                 best_possible_correct_at_most=bound,upper_accuracy=bound/len(y),
                 base_correct=int(correct.sum()),needed_correct_for_target=int(np.floor(.85*len(y)))+1,
                 certificate_definition="All group repetitions averaged by member. A shared nonnegative weight vector sums to one. Each disjoint pair has a convex mixture of its label-signed member margins strictly negative for every member.")
    write(ROOT/"reports/nov23_convex_portfolio_bound/result.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    run()
