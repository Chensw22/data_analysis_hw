"""Penalize inconsistent outcome probabilities across two re-fits on identical calibration passengers."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import proposal_design
from full_roster_repartition_gate_n24 import ROOT, panel_data


def fit(x, y, differences, weight):
    def objective(w):
        z=x @ w
        disagreement=differences @ w
        value=np.logaddexp(0,z).sum()-y @ z + 5*(w[1:] @ w[1:]) + weight*.5*(disagreement@disagreement)
        grad=x.T @ (expit(z)-y)+weight*differences.T @ disagreement
        grad[1:]+=10*w[1:]
        return value,grad
    opt=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'maxiter':500})
    assert opt.success, opt.message
    return opt.x


def run():
    raw,y,g,fold,names,original,full,ref,base,cols,rotated=panel_data()
    sources=[original[:,cols],rotated]
    panels=[fold%2,(fold//2)%2]
    fd,fx=proposal_design(full[:,:5],full[:,cols[5:]])
    options={k:base.copy() for k in ['plain','matched','exchanged']}
    audits=[]
    for f in np.unique(fold):
        q=np.flatnonzero((fold==f)&fd)
        contexts=[]
        for bank,panel in zip(sources,panels,strict=True):
            dis,design=proposal_design(bank[:,:5],bank[:,5:])
            t=np.flatnonzero((fold!=f)&(panel==panel[q[0]])&dis)
            assert not np.isin(g[t],g[q]).any()
            contexts.append((t,design,bank[:,4]>=.5))
        same=np.intersect1d(contexts[0][0],contexts[1][0])
        x=np.vstack([design[t] for t,design,direction in contexts])
        labels=np.concatenate([y[t]!=direction[t] for t,design,direction in contexts])
        scaler=StandardScaler().fit(x)
        def to_design(xx):
            return np.column_stack([np.ones(len(xx)),scaler.transform(xx)])
        x=to_design(x)
        # Difference between outcome logits, accounting for a changed decision direction.
        paired=[to_design(design[same])*np.where(direction[same],-1,1)[:,None]
                for _,design,direction in contexts]
        exchange=paired[1].copy()
        rng=np.random.default_rng(754+int(f))
        # Match by each estimator's original direction before exchanging passenger identity.
        state=contexts[0][2][same].astype(int)*2+contexts[1][2][same]
        for value in np.unique(state):
            bucket=np.flatnonzero(state==value)
            for pos in bucket:
                other=bucket[g[same[bucket]]!=g[same[pos]]]
                if len(other):
                    exchange[pos]=paired[1][rng.choice(other)]
        audits.append(dict(fold=int(f),same_passenger_pairs=len(same),
                           opposite_base_direction=int((contexts[0][2][same]!=contexts[1][2][same]).sum())))
        for k,d,w in [('plain',paired[0]-paired[1],0.),('matched',paired[0]-paired[1],1.),
                      ('exchanged',paired[0]-exchange,1.)]:
            coef=fit(x,labels,d,w)
            options[k][q]^=(to_design(fx[q])@coef)>=0
    results={}
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    for k,pred in options.items():
        delta=(pred==y).astype(int)-(ref==y)
        results[k]=dict(accuracy=float((pred==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                        additional_net=int(delta.sum()-delta[dev].sum()),fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
                        ci_pp=interval(delta,g))
    baseline=json.loads((ROOT/'reports/nov23_matched_learning_curve/repartition_gate.json').read_text())
    np.testing.assert_array_equal(results['plain']['fold_net'],baseline['results']['pooled']['fold_net'])
    (ROOT/'reports/nov23_matched_learning_curve/panel_consistency.json').write_text(
        json.dumps(dict(matches=audits,results=results),indent=2)+'\n')
    print(results)


if __name__=='__main__':
    run()
