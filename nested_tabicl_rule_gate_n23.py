"""Train a rule-vs-outcome gate with group-disjoint cross-fits inside each outer train split."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from tabicl_routed_context_n23 import ART,ROOT,WEIGHT,DIGEST,CAT,hard_masks,infer
from tabicl_induced_rules_n23 import indicators

SEED=20261112


def predict_trio(x,y,g,t,q,seed):
    assert not np.intersect1d(g[t],g[q]).size
    cb=CatBoostClassifier(iterations=520,depth=5,learning_rate=.05,random_seed=311,
                          thread_count=2,allow_writing_files=False,verbose=False)
    cb.fit(x.iloc[t],y[t],cat_features=CAT)
    tree=cb.predict_proba(x.iloc[q])[:,1]
    full=infer(x,y,t,q,seed)
    rules=np.zeros(len(q))
    for a,b in GroupKFold(2,shuffle=True,random_state=483).split(t,groups=g[t]):
        discover,support=t[a],t[b]
        assert not np.intersect1d(g[discover],g[support]).size
        local=x.copy()
        for col in CAT:
            # Rule-selection design width and vocabulary depend on discovery bookings only.
            levels=pd.unique(x.iloc[discover][col])
            local[col]=pd.Categorical(local[col].where(local[col].isin(levels)),categories=levels)
        v=indicators(local,y,discover)
        augmented=x.join(pd.DataFrame(v,columns=[f"induced_{j}" for j in range(v.shape[1])]))
        rules+=.5*infer(augmented,y,support,q,seed+28)
    return np.column_stack([tree,full,rules])


def gate_design(pred):
    cb,full,rules=pred.T
    base=(cb+full)/2
    # Sign the two family margins so that a positive margin supports the proposed change.
    direction=np.where(base>=.5,-1.,1.)
    odds=logit(np.clip(base,.005,.995))*direction
    x=np.column_stack([np.ones(len(pred)),rules*direction-.5*direction,
                       (full-cb)*direction,np.abs(cb-full),base>=.5])
    dispute=(rules>=.5)!=(base>=.5)
    return base,odds,x,dispute


def policy(train,y,query):
    base,offset,design,eligible=gate_design(train)
    x=design[eligible]
    target=((base>=.5)!=y).astype(float)[eligible]
    bias=offset[eligible]
    def fun(w):
        z=bias+x@w
        return (np.logaddexp(0,z).sum()-target@z+4*w@w,
                x.T@(expit(z)-target)+8*w)
    fit=minimize(fun,np.zeros(x.shape[1]),jac=True,method="BFGS",tol=1e-6)
    if not np.isfinite(fit.x).all() or np.linalg.norm(fit.jac)>1e-4:
        raise ValueError(fit.message)
    b,prior,z,d=gate_design(query)
    error=expit(prior[d]+z[d]@fit.x)
    ans=b.copy()
    ans[d]=np.where(b[d]>=.5,1-error,error)
    return ans,dict(calibration_rows=len(y),disputes=int(eligible.sum()),viable=int(target.sum()),
                    query_disputes=int(d.sum()),weights=fit.x.tolist())


def run():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    with np.load(ART/f"nov23_group_kernel/{SEED}.npz") as z:
        row,fold,y,prior=(z[k] for k in ["row","fold","y","base"])
    roster=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[row].reset_index(drop=True)
    g=roster.PassengerId.str.split("_").str[0].to_numpy()
    x=roster.drop(columns=["Cabin","PassengerId"]).copy()
    x[["Deck","CabinNumber","Side"]]=roster.Cabin.str.split("/",expand=True)
    x.CabinNumber=pd.to_numeric(x.CabinNumber)
    x[CAT]=x[CAT].fillna("__MISSING__").astype(str)
    out=ART/"nov23_nested_rule_gate"
    out.mkdir(exist_ok=True)
    p=np.full((len(row),3),np.nan)
    learned=np.full(len(row),np.nan)
    evidence=[]
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.intersect1d(g[t],g[q]).size
        inner=np.full((len(t),3),np.nan)
        for j,(a,b) in enumerate(GroupKFold(3,shuffle=True,random_state=851).split(t,groups=g[t])):
            inner[b]=predict_trio(x,y,g,t[a],t[b],43+int(k))
            print("outer",int(k),"inner",j,flush=True)
        assert np.isfinite(inner).all()
        p[q]=predict_trio(x,y,g,t,q,43+int(k))
        learned[q],record=policy(inner,y[t],p[q])
        evidence.append(record)
        np.savez_compressed(out/f"fold{int(k)}.npz",calibration_row=row[t],calibration_predictions=inner,
                            query_row=row[q],query_predictions=p[q],gate=learned[q])
        print("outer",int(k),"complete",flush=True)
    ref=p[:,:2].mean(1)
    outputs=dict(reference=ref,rules=p[:,2],quarter_rules=.75*ref+.25*p[:,2],gate=learned)
    hard,unresolved=hard_masks(row,y)
    summary={}
    for name,prob in outputs.items():
        assert np.isfinite(prob).all()
        ok=(prob>=.5)==y
        wrong=(ref>=.5)!=y
        summary[name]=dict(accuracy=float(ok.mean()),fixed=int((wrong&ok).sum()),harmed=int((~wrong&~ok).sum()),
           hard447_correct=int(ok[hard].sum()),hard356_correct=int(ok[unresolved].sum()),
           fold_net=[int(ok[fold==k].sum()-(~wrong)[fold==k].sum()) for k in np.unique(fold)])
    np.savez_compressed(out/f"{SEED}.npz",row=row,y=y,fold=fold,base=prior,catboost=p[:,0],tabicl=p[:,1],p=learned,**outputs)
    dest=ROOT/"reports/nov23_nested_rule_gate/result.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(dict(seed=SEED,checkpoint=DIGEST,results=summary,outer_fits=evidence),indent=2)+"\n")
    print(summary,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
