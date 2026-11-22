"""Rank errors against correct predictions at comparable Earth passenger base risk."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, predictions
from relational_error_corrector import write

STRONG={"bayesian":("bart_long","bart"),"boarding":("boarding_sequence","sequence"),
        "context":("tabicl_context","global_context")}
WEAK={"registration":("bill_matched_register","register"),"budget":("simplex_likelihood","budget_prior"),
      "projections":("projection_shapes","projection_shapes"),"fields":("field_factors","order3"),
      "density":("manifest_tree","manifest_tree"),"gradient":("group_kernel","learned_twice")}


def pairs(rows,bands,errors,booking):
    left,right=[],[]
    for b in np.unique(bands[rows]):
        local=rows[bands[rows]==b]
        a=local[errors[local]]
        c=local[~errors[local]]
        i,j=np.where(booking[a,None]!=booking[c])
        left.extend(a[i]);right.extend(c[j])
    return np.asarray(left,int),np.asarray(right,int)


def corrector(score,x,error,train,query,pair,cols):
    left,right=pair
    x=x[:,cols]
    spread=x[train].std(0).clip(.3)
    x=((x-x[train].mean(0))/spread).clip(-5,5)
    d=x[left]-x[right]
    def objective(w):
        z=d@w
        return (np.logaddexp(0,-z).mean()+.075*w@w,
                -d.T@expit(-z)/len(z)+.15*w)
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method="BFGS")
    assert fit.success
    evidence=x@fit.x
    design=np.column_stack([np.ones(len(x)),evidence])
    def calibration(w):
        z=score[train]+design[train]@w
        return (np.logaddexp(0,z).sum()-z@error[train]+5*w@w,
                design[train].T@(expit(z)-error[train])+10*w)
    fit=minimize(calibration,[0.,0.],jac=True,method="BFGS")
    assert fit.success or np.max(abs(fit.jac))<1e-5
    return expit(score[query]+design[query]@fit.x),evidence[query]


def summarize(ids,risk,error,tests,order_score):
    a,b=tests
    d=risk[a]-risk[b]
    e=order_score[a]-order_score[b]
    flips=risk[ids]>.5
    # The oracle selects the best threshold after seeing evaluation labels,
    # including the no-change option and respecting ties.
    o=ids[np.argsort(-risk[ids],kind="stable")]
    cuts=np.r_[np.where(np.diff(risk[o])!=0)[0],len(o)-1]
    gains=np.cumsum(np.where(error[o],1,-1))[cuts]
    return dict(rows=len(ids),errors=int(error[ids].sum()),cross_booking_pairs=len(a),
                auc=float(roc_auc_score(error[ids],risk[ids])),
                matched_auc=float(((d>0)+.5*(d==0)).mean()),
                evidence_only_matched_auc=float(((e>0)+.5*(e==0)).mean()),
                fixed=int(error[ids][flips].sum()),harmed=int((~error[ids][flips]).sum()),
                best_possible_threshold_net=int(max(0,gains.max())))


def run():
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    data=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","HomePlanet"]).iloc[row]
    earth=data.HomePlanet.eq("Earth").to_numpy()
    booking=data.PassengerId.str.split("_").str[0].to_numpy()
    result={}
    shared=earth.copy()
    flips_by_seed={}
    for seed in SEEDS:
        base,op,fold=predictions(row,y,seed,{**STRONG,**WEAK})
        positive=base>=.5
        error=y!=positive
        shared &= error
        prior=np.minimum(base,1-base).clip(.00001,.499999)
        prior_logit=logit(prior)
        bands=np.floor(40*prior).astype(int)
        x=logit(np.where(positive[:,None],1-op,op).clip(.001,.999))-prior_logit[:,None]
        summary={}
        flips=np.zeros(len(y),bool)
        for flag in (False,True):
            ids=np.flatnonzero(earth & (positive==flag))
            risk={"base":prior.copy(),"strong":np.full(len(y),np.nan),"weak":np.full(len(y),np.nan),"all":np.full(len(y),np.nan)}
            evidence={name:v.copy() for name,v in risk.items()}
            held_pairs=[]
            for f in np.unique(fold):
                outer=ids[fold[ids]==f]
                for a,b in GroupKFold(3,shuffle=True,random_state=470).split(outer,groups=booking[outer]):
                    t,q=outer[a],outer[b]
                    assert not np.intersect1d(booking[t],booking[q]).size
                    pair=pairs(t,bands,error,booking)
                    held_pairs.append(pairs(q,bands,error,booking))
                    for name,cols in (("strong",slice(0,len(STRONG))),("weak",slice(len(STRONG),None)),("all",slice(None))):
                        risk[name][q],evidence[name][q]=corrector(prior_logit,x,error,t,q,pair,cols)
            held=tuple(np.concatenate(v) for v in zip(*held_pairs))
            assert (booking[held[0]]!=booking[held[1]]).all() and (bands[held[0]]==bands[held[1]]).all()
            summary[str(flag)]={k:summarize(ids,v,error,held,evidence[k]) for k,v in risk.items()}
            flips[ids]=risk["all"][ids]>.5
        flips_by_seed[seed]=flips
        initial=int((~error).sum())
        a,b=summary["False"]["all"],summary["True"]["all"]
        summary["all_passengers"]=dict(initial=initial,
            accuracy=(initial+a["fixed"]+b["fixed"]-a["harmed"]-b["harmed"])/len(y),
            two_threshold_oracle=(initial+a["best_possible_threshold_net"]+b["best_possible_threshold_net"])/len(y))
        result[seed]=summary
        print(seed,summary["all_passengers"],[(v["all"]["matched_auc"],v["all"]["fixed"],v["all"]["harmed"]) for v in (summary["False"],summary["True"])])
    assert earth.sum()==3659 and shared.sum()==804
    write(ROOT/"reports/nov23_earth_direction_audit/result.json",dict(earth_rows=int(earth.sum()),base_wrong_three=int(shared.sum()),
          fixed_on_persistent=[int((shared&v).sum()) for v in flips_by_seed.values()],
          fixed_every_split=int((shared&np.logical_and.reduce(list(flips_by_seed.values()))).sum()),results=result))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
