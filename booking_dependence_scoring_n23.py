"""Separate the value of booking dependence for joint outcomes from its value for individual accuracy."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exact_booking_field_n23 import ART,ROOT,SEEDS,compact_features,fit,read_panel,structure,verify,write


def marginal_fit(selected):
    def objective(w):
        loss=6*w@w
        grad=12*w.copy()
        for spin,base,x,label in selected:
            score=base+x@w
            prob=softmax(score,axis=1)
            pred=(prob@spin).clip(1e-10,1-1e-10)
            valid=label>=0
            loss-=np.where(valid,np.where(label==1,np.log(pred),np.log1p(-pred)),0).sum()
            error=np.where(valid,(pred-label)/(pred*(1-pred)),0.)
            # Derivative of each marginal wrt a joint-state logit.
            influence=prob*((error@spin.T)-(error*pred).sum(axis=1)[:,None])
            grad+=np.einsum("bs,bsf->f",influence,x)
        return loss,grad
    sol=minimize(objective,np.zeros(5),jac=True,method="L-BFGS-B",options=dict(ftol=1e-12,maxiter=150))
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x


def run():
    frame=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row=frame.row.to_numpy()
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"])
    fields=compact_features(raw)
    g=raw.PassengerId.str.split("_").str[0]
    group=g.iloc[row].to_numpy()
    target=np.full(len(raw),-1)
    target[row]=y
    records=json.loads((ROOT/"reports/nov23_exact_booking_field/result.json").read_text())["results"]
    results={}
    output=ART/"nov23_booking_scoring"
    output.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        names=["independent","fit_joint","fit_marginal"]
        predictions={name:np.full(len(raw),np.nan) for name in names}
        for p in predictions.values():
            p[row]=base
        log_scores={name:np.zeros(2) for name in names}
        known_in_groups=0
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            groups=np.unique(group[q])
            assert not np.intersect1d(group[t],groups).size
            pool=np.flatnonzero(g.isin(groups))
            hidden=np.setdiff1d(pool,row[q])
            scores=np.full(len(raw),np.nan)
            if len(hidden):
                assert (target[hidden]<0).all()
                scores[hidden]=fit(fields,target,row[t],155).predict_proba(fields.iloc[hidden])[:,1]
            scores[row[q]]=base[q]
            blocks=structure(raw,g,pool,scores,target)
            weights=[np.array(s["params"]) for s in records[str(seed)]["calibrations"] if s["fold"]==k and s["method"]=="joint"]
            parts=GroupKFold(3,shuffle=True,random_state=117).split(q,groups=group[q])
            for j,(a,b) in enumerate(parts):
                learn,test=np.unique(group[q[a]]),np.unique(group[q[b]])
                assert not np.intersect1d(learn,test).size
                selected=[]
                for ids,spin,offset,x,_,keys in blocks:
                    sel=np.isin(keys,learn)
                    if sel.any():
                        selected.append((spin,offset[sel],x[sel],target[ids[sel]]))
                w=marginal_fit(selected)
                for ids,spin,offset,x,compatible,keys in blocks:
                    sel=np.isin(keys,test)
                    if not sel.any():
                        continue
                    ix=ids[sel]
                    labels=target[ix]
                    ok=labels>=0
                    known_in_groups+=int(ok.sum())
                    for name,beta in [("independent",np.zeros(5)),("fit_joint",weights[j]),("fit_marginal",w)]:
                        l=offset[sel]+x[sel]@beta
                        p=softmax(l,axis=1)@spin
                        predictions[name][ix]=p
                        joint=(logsumexp(l,axis=1)-logsumexp(np.where(compatible[sel],l,-np.inf),axis=1)).sum()
                        marginal=-np.log(np.where(labels==1,p,1-p)[ok].clip(1e-12,1)).sum()
                        log_scores[name]+=[joint,marginal]
        with np.load(ART/f"nov23_exact_booking_field/{seed}.npz") as z:
            np.testing.assert_allclose(predictions["fit_joint"][row],z["joint"],atol=1e-10)
        old=(base>=.5)==y
        params={}
        for name,p in predictions.items():
            score=p[row]
            ok=(score>=.5)==y
            a,b=log_scores[name]/known_in_groups
            params[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                              persistent1070=int(ok[~hits.any(0)].sum()),unresolved466=int(ok[hard].sum()),
                              joint_nll_per_known_member=float(a),marginal_nll_per_known_member=float(b))
        results[seed]=dict(multi_member_passengers=known_in_groups,metrics=params)
        np.savez_compressed(output/f"{seed}.npz",row=row,fold=fold,y=y,base=base,**{k:v[row] for k,v in predictions.items()})
        print(seed,params,flush=True)
    write(ROOT/"reports/nov23_booking_scoring/result.json",dict(results=results,
          interpretation="Held-out joint and marginal log losses are per observed member of non-singleton bookings; accuracy includes every development passenger."))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
