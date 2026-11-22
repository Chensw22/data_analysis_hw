"""Fit sparse rule corrections to the observed errors of frozen out-of-booking predictors."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import CATEGORICAL,compact_features
from grouped_five_way_coverage_n23 import verify
from grouped_sparse_rules_n23 import remaining,rule_basis
from relational_error_corrector import write


def residual(design,offset,y,t,q):
    a=design[t]
    width=a.shape[1]
    def objective(v):
        weight=v[:width]-v[width:]
        score=offset[t]+a@weight
        grad=a.T@(expit(score)-y[t])
        # Convex L1-penalized offset likelihood, with small quadratic stabilization.
        lam=5.
        loss=np.logaddexp(0,score).sum()-y[t]@score+lam*v.sum()+.5*weight@weight
        grad=grad+weight
        return loss,np.r_[grad+lam,-grad+lam]
    fit=minimize(objective,np.zeros(2*width),jac=True,method="L-BFGS-B",bounds=[(0,None)]*(2*width),
                 options={"maxiter":500,"ftol":1e-10})
    if not fit.success:raise RuntimeError(fit.message)
    weights=fit.x[:width]-fit.x[width:]
    return expit(offset[q]+design[q]@weights),int((np.abs(weights)>1e-6).sum())


def run():
    catalog=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=catalog.row.to_numpy(),catalog.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{p}.csv",usecols=lambda c:c not in ["Name","Transported"])
                   for p in ["train","test"]],ignore_index=True)
    party=raw.PassengerId.str.split("_").str[0]
    groups=party.iloc[row].to_numpy()
    frame=compact_features(raw).iloc[row].copy()
    frame["BookingSize"]=party.map(party.value_counts()).iloc[row].to_numpy()
    x=pd.get_dummies(frame,columns=CATEGORICAL).fillna(-1).to_numpy(np.float32)
    hard=remaining(row,y)
    out=ART/"nov23_rule_offsets"
    out.mkdir(exist_ok=True)
    summary={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_sparse_rules/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
            mixture=.75*base+.25*z["rules"]
            old_tree=z["tree"]
        offset=logit(np.clip(mixture,.001,.999))
        preds={key:np.full(len(y),np.nan) for key in ["global_calibration","rule_offsets"]}
        active=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not set(groups[t])&set(groups[q])
            tree=GradientBoostingClassifier(n_estimators=120,learning_rate=.045,max_depth=3,min_samples_leaf=20,
                                            max_features=.8,subsample=.78,random_state=312)
            tree.fit(x[t],y[t])
            np.testing.assert_allclose(tree.predict_proba(x[q])[:,1],old_tree[q])
            _,rules=rule_basis(tree,x[q])
            group_q=groups[q]
            for a,b in GroupKFold(3,shuffle=True,random_state=899).split(q,groups=group_q):
                assert not set(group_q[a])&set(group_q[b])
                # Only attendees of the same original validation fold train this
                # decision layer. No underlying tree or foundation saw query labels.
                p,n=residual(rules,offset[q],y[q].astype(float),a,b)
                preds["rule_offsets"][q[b]]=p
                active.append(n)
                intercept=minimize(lambda w:np.logaddexp(0,offset[q[a]]+w[0]).sum()-y[q[a]].astype(float)@(offset[q[a]]+w[0])+2*w[0]**2,[0.])
                preds["global_calibration"][q[b]]=expit(offset[q[b]]+intercept.x[0])
        results={}
        for key,p in {"fixed_rule_blend":mixture,**preds}.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            wrong=(base>=.5)!=y
            wrong_mix=(mixture>=.5)!=y
            results[key]=dict(accuracy=float(ok.mean()),fixed=int((ok&wrong).sum()),harmed=int((~ok&~wrong).sum()),
                              fixed_vs_mixture=int((ok&wrong_mix).sum()),harmed_vs_mixture=int((~ok&~wrong_mix).sum()),
                              hard356_correct=int(ok[hard].sum()))
        summary[str(seed)]=dict(results=results,nonzero_rules=active)
        print(seed,results,flush=True)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,mixture=mixture,**preds)
    write(ROOT/"reports/nov23_rule_offsets/result.json",summary)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
