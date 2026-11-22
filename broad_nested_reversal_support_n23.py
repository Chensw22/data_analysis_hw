"""Measure what more independent examples of helpful and harmful reversals do for a selector."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_dispute_leaf_evidence_n23 import ART,ROOT,SEEDS,REPORT,cohorts
from broad_shared_passenger_modes_n23 import expanded_design,axes,peer_pairs
from broad_matched_companions_n23 import fit
from exaone_validation_data import compact_features
from relational_error_corrector import write


def choose(scores,attrs,labels,train_n,query_scores,query_attrs,with_attributes):
    # Reference and expert input have the same semantics in each inner fit.
    ref=scores[:,0]>=.5
    query_ref=query_scores[:,0]>=.5
    def inputs(p,x):
        logits=logit(p.clip(.002,.998))
        return np.column_stack([logits,x]) if with_attributes else logits
    x=inputs(scores,attrs)
    z=inputs(query_scores,query_attrs)
    out=query_scores[:,0].copy()
    for side in (False,True):
        fit_rows=train_n[(ref[train_n]==side)&((scores[train_n,1]>=.5)!=side)]
        take=(query_ref==side)&((query_scores[:,1]>=.5)!=side)
        if not take.any():
            continue
        model=LogisticRegression(C=.09,max_iter=400).fit(x[fit_rows],labels[fit_rows])
        out[take]=model.predict_proba(z[take])[:,1]
    return out


def run():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Transported","Name"))
                   for s in ("train","test")],ignore_index=True)
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    size=pd.Series(groups).groupby(groups).transform("size").to_numpy()
    frame=compact_features(raw)
    hits={}
    refs=[]
    paired={}
    sizes={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            row,y,fold,base=[ref[n] for n in ("row","y","fold","base")]
        gs=groups[row]
        saved={}
        sizes[seed]=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(gs[t],gs[q]).size
            held=np.isin(groups,gs[q])
            unsup=np.flatnonzero(~held)
            phi,z=expanded_design(frame,row[t],unsup)
            a,b,_,w=peer_pairs(groups,unsup)
            context=z[row]@axes(z,a,b,w)
            own=np.column_stack([phi[row],np.log(size[row])])
            train=np.full((len(t),2),np.nan)
            test=[]
            parts=[]
            for fit_ix,cal_ix in GroupKFold(3,shuffle=True,random_state=313).split(t,groups=gs[t]):
                assert not np.intersect1d(gs[t[fit_ix]],gs[t[cal_ix]]).size
                query=np.r_[t[cal_ix],q]
                scores=np.column_stack([fit(x[t[fit_ix]],y[t[fit_ix]]).predict_proba(x[query])[:,1] for x in (own,context)])
                train[cal_ix]=scores[:len(cal_ix)]
                test.append(scores[len(cal_ix):])
                parts.append(cal_ix)
            assert np.isfinite(train).all()
            full=np.arange(len(t))
            sizes[seed].append(int(((train[:,0]>=.5)!=(train[:,1]>=.5)).sum()))
            def save(key,vals):
                saved.setdefault(key,np.full(len(y),np.nan))[q]=vals
            surrogate=np.mean([m[:,0] for m in test],axis=0)
            save("native_reference",surrogate)
            for name,units,attrs in (("one_third",parts,True),("full", [full],True),("full_scores",[full],False)):
                pred=[]
                transfer=[]
                for n,query in enumerate(test):
                    subset=units[n%len(units)]
                    pred.append(choose(train,own[t],y[t],subset,query,own[q],attrs))
                    transfer.append(choose(train,own[t],y[t],subset,np.column_stack([base[q],query[:,1]]),own[q],attrs))
                save(name+"_native",np.mean(pred,axis=0))
                save(name+"_transfer",np.mean(transfer,axis=0))
        for name,p in saved.items():
            assert np.isfinite(p).all()
            hits.setdefault(name,[]).append((p>=.5)==y)
        refs.append((base>=.5)==y)
        print(seed,{n:round(float(((p>=.5)==y).mean()),4) for n,p in saved.items()},flush=True)
    masks=cohorts(row,y)
    ref=np.stack(refs)
    surrogate=np.stack(hits["native_reference"])
    for n,v in hits.items():
        ok=np.stack(v)
        paired[n]=dict(accuracy=ok.mean(1).tolist(),fixed=(ok&~ref).sum(1).tolist(),harmed=(~ok&ref).sum(1).tolist(),
                       vs_native_net=(ok.astype(int)-surrogate).sum(1).tolist(),
                       stable={k:int((m&ok.all(0)).sum()) for k,m in masks.items()})
    write(REPORT/"nested_support.json",dict(results=paired,outer_training_disputes=sizes))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
