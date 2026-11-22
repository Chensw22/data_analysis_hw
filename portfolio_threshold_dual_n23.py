"""Validate upper accuracy bounds for unrestricted global thresholds via LP dual witnesses."""
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack, vstack

from convex_portfolio_bound_n23 import ART,ROOT,read_scores
from relational_error_corrector import write


def certify(row,y,p,lo,hi,key):
    sign=np.where(y,1.,-1.)
    relaxed=np.where(y[:,None],p-lo,hi-p)
    worst=np.where(y[:,None],p-hi,lo-p)
    forced=relaxed.max(1)<-1.e-7
    # A row can always be included unless it is already charged as a forced mistake.
    variable=(worst.min(1)<0)&~forced
    ids=np.flatnonzero(variable)
    n=len(ids)
    with np.load(ART/"nov23_portfolio_threshold"/(key+".npz")) as witness:
        position = {passenger: i for i, passenger in enumerate(row)}
        a, b = (np.array([position[v] for v in witness[k]]) for k in ("left", "right"))
        lam = witness["lambda_left"][:,None]
        assert np.all((lam >= 0) & (lam <= 1))
        assert np.all(lam*relaxed[a] + (1-lam)*relaxed[b] < 0)
    idx=np.full(len(y),-1)
    idx[ids]=np.arange(n)
    assert np.all(idx[a]>=0) and np.all(idx[b]>=0)
    k=p.shape[1]
    M=np.maximum(0,-worst[ids].min(1))
    matrix=hstack([csr_matrix(sign[ids,None]*p[ids]),csr_matrix(-sign[ids,None]),diags(M)],format="csr")
    paired=coo_matrix((np.ones(2*len(a)),(np.tile(np.arange(len(a)),2),np.r_[idx[a],idx[b]])),
                      shape=(len(a),n)).tocsr()
    all_constraints=vstack([matrix,hstack([csr_matrix((len(a),k+1)),paired])],format="csr")
    rhs=np.r_[np.zeros(n),np.ones(len(a))]
    c=np.r_[np.zeros(k+1),np.ones(n)]
    eq=np.r_[np.ones(k),np.zeros(n+1)][None,:]
    # scipy's dual values are used only to propose nonnegative multipliers.
    solution=linprog(c,A_ub=-all_constraints,b_ub=-rhs,A_eq=eq,b_eq=[1.],
                     bounds=[(0,1)]*k+[(lo,hi)]+[(0,1)]*n,method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    multiplier=np.maximum(0.,-solution.ineqlin.marginals)
    residual=c-all_constraints.T@multiplier
    lb=(int(forced.sum())+rhs@multiplier+residual[:k].min()
        +min(lo*residual[k],hi*residual[k])+np.minimum(0,residual[k+1:]).sum())
    # The Lagrangian expression lower-bounds all integer classifiers for any
    # nonnegative multipliers, regardless of LP solver feasibility or optimality.
    # The margin below is deliberately far larger than floating-point roundoff.
    integer=int(np.ceil(lb-1e-5))
    assert lb<=int(forced.sum())+solution.fun+1e-5
    return dict(singles=int(forced.sum()),pairs=len(a),continuous_dual_lower=float(lb),integer_lower=integer),multiplier


def run():
    row,y,p=read_scores()
    old=json.loads((ROOT/"reports/nov23_portfolio_threshold/result.json").read_text())
    minimum=len(y)-int(np.floor(.85*len(y)))
    intervals=[]
    cache=ART/"nov23_portfolio_dual"
    cache.mkdir(exist_ok=True)
    for r in old["threshold_intervals"]:
        lo,hi=r["lower"],r["upper"]
        lower=r["forced_mistakes"]
        check={}
        if lower<minimum:
            check,dual=certify(row,y,p,lo,hi,r["key"])
            np.savez_compressed(cache/(r["key"]+".npz"),multipliers=dual)
            lower=max(lower,check["integer_lower"])
        intervals.append(dict(lower_threshold=lo,upper_threshold=hi,errors_at_least=lower,dual=check))
    n=min(r["errors_at_least"] for r in intervals)
    result=dict(rows=len(y),minimum_mistakes_all_thresholds=n,upper_accuracy=1-n/len(y),intervals=intervals)
    write(ROOT/"reports/nov23_portfolio_dual/result.json",result)
    print(n,result["upper_accuracy"],[(r["lower_threshold"],r["errors_at_least"]) for r in intervals if r["dual"]],flush=True)


if __name__=="__main__":
    run()
