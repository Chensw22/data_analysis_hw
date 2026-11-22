"""Discover rule paths from strict inner out-of-booking teacher residuals."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import CATEGORICAL,compact_features
from grouped_five_way_coverage_n23 import verify
from grouped_sparse_rules_n23 import remaining,rule_basis
from relational_error_corrector import write



def teachers(frame,y,group,t,q,seed):
    train=np.full(len(t),np.nan)
    query=[]
    for a,b in GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=group[t]):
        assert not np.intersect1d(group[t[a]],group[t[b]]).size
        m=CatBoostClassifier(iterations=420,depth=5,learning_rate=.055,thread_count=2,
                             verbose=False,allow_writing_files=False,random_seed=235)
        m.fit(frame.iloc[t[a]],y[t[a]],cat_features=CATEGORICAL)
        train[b]=m.predict_proba(frame.iloc[t[b]])[:,1]
        query.append(m.predict_proba(frame.iloc[q])[:,1])
    assert np.isfinite(train).all()
    return train,np.column_stack(query)


def fit_offset(basis,offset,labels,t,q):
    # Screen both dictionaries with the same standardized score statistic on
    # the outer training outcomes. A compact set makes the penalized fit stable.
    a=basis[t]
    p=expit(offset[t])
    fisher=a.T@(p*(1-p))
    score=np.abs(a.T@(labels[t]-p))/np.sqrt(fisher+1)
    columns=np.argsort(-score,kind="stable")[:96]
    a=a[:,columns].toarray()
    width=len(columns)
    def objective(v):
        w=v[:width]-v[width:]
        log_odds=offset[t]+a@w
        gradient=a.T@(expit(log_odds)-labels[t])+w
        loss=np.logaddexp(0,log_odds).sum()-labels[t]@log_odds+5*v.sum()+.5*w@w
        return loss,np.r_[gradient+5,5-gradient]
    fit=minimize(objective,np.zeros(2*width),jac=True,method="L-BFGS-B",
                 bounds=[(0,None)]*(2*width),options={"maxiter":4000,"ftol":1e-10})
    if not fit.success:
        raise RuntimeError(fit.message)
    w=fit.x[:width]-fit.x[width:]
    return basis[q][:,columns]@w,int((np.abs(w)>1e-6).sum())


def run():
    catalog=pd.read_csv(ART/'nov23_global_error_inventory.csv')
    row,y=catalog.row.to_numpy(),catalog.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in ["Transported","Name"])
                   for part in ["train","test"]],ignore_index=True)
    booking=raw.PassengerId.str.split("_").str[0]
    group=booking.iloc[row].to_numpy()
    frame=compact_features(raw.iloc[row]).reset_index(drop=True)
    frame["BookingSize"]=booking.map(booking.value_counts()).to_numpy()[row]
    x=pd.get_dummies(frame,columns=CATEGORICAL).fillna(-1).to_numpy(np.float32)
    hard=remaining(row,y)
    out=ART/"nov23_residual_discovery"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        preds={n:np.full(len(y),np.nan) for n in ["teacher","outcome_rules","residual_rules"]}
        active=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            pt,pq=teachers(frame,y,group,t,q,seed+int(k))
            preds["teacher"][q]=pq.mean(axis=1)
            offset=np.zeros(len(y))
            offset[t]=logit(np.clip(pt,.001,.999))
            # Use a zero reference logit on q to recover the fitted additive correction.
            # Apply the same correction to each matched-size teacher before averaging.
            selected={}
            for name,response in [("outcome_rules",y[t].astype(float)),("residual_rules",y[t]-pt)]:
                m=GradientBoostingRegressor(n_estimators=90,learning_rate=.04,max_depth=3,
                                             min_samples_leaf=26,max_features=.8,subsample=.8,random_state=663)
                m.fit(x[t],response)
                _,basis=rule_basis(m,x)
                adjustment,selected[name]=fit_offset(basis,offset,y.astype(float),t,q)
                preds[name][q]=expit(logit(np.clip(pq,.001,.999))+adjustment[:,None]).mean(axis=1)
            active.append(selected)
        result={}
        for name,p in preds.items():
            assert np.isfinite(p).all()
            for key,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                ok=(score>=.5)==y
                base_wrong=(base>=.5)!=y
                teacher_wrong=(preds["teacher"]>=.5)!=y
                result[key]=dict(accuracy=float(ok.mean()),base_fixed=int((base_wrong&ok).sum()),
                    base_harmed=int((~base_wrong&~ok).sum()),teacher_fixed=int((teacher_wrong&ok).sum()),
                    teacher_harmed=int((~teacher_wrong&~ok).sum()),hard356_correct=int(ok[hard].sum()))
        results[str(seed)]=dict(results=result,active_rules=active)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**preds)
        print(seed,result,flush=True)
    write(ROOT/"reports/nov23_residual_discovery/result.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
