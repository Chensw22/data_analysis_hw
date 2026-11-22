"""Test how much passenger information remains after conditioning on honest model scores."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_n23 import ACTIONS
from archive_error_recovery_n23 import ART, ROOT, SEEDS, predictions
from broad_dispute_leaf_evidence_n23 import cohorts
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from relational_error_corrector import write

REPORT=ROOT/"reports/nov23_broad_conditional_raw_signal"
OUT=ART/"nov23_broad_conditional_raw_signal"


def donor(score,group):
    # A bijection preserves the full marginal distribution and dependence
    # between passenger fields. A passenger cannot receive their own booking.
    distances=cdist(score,score,"sqeuclidean")
    distances[group[:,None]==group]=1e8
    i,j=linear_sum_assignment(distances)
    np.testing.assert_array_equal(i,np.arange(len(group)))
    assert (group[i]!=group[j]).all()
    return j,np.sqrt(distances[i,j]/score.shape[1])


def innovation(score,raw,t,q):
    z=(score-score[t].mean(0))/score[t].std(0).clip(.3)
    points=z[t[np.random.default_rng(724).choice(len(t),64,replace=False)]]
    dist=cdist(z,points,"sqeuclidean")
    width=max(np.median(dist[t]),.1)
    h=np.column_stack([np.ones(len(z)),z,np.exp(-dist/width),np.exp(-4*dist/width)])
    train=h[t]
    gram=train.T@train+3*np.eye(train.shape[1])
    ch=cho_factor(gram)
    coef=cho_solve(ch,train.T@raw[t])
    leverage=np.sum(train*cho_solve(ch,train.T).T,axis=1)
    # Leave-one-row-out prediction error for the unlabeled nuisance regression.
    return (raw[t]-train@coef)/(1-leverage[:,None]),raw[q]-h[q]@coef,z


def score_fit(x,y,z):
    m=LogisticRegression(C=.08,max_iter=400).fit(x,y)
    return m.predict_proba(z)[:,1]


def summarize(all_probs,base,row,y,group):
    ref=(base>=.5)==y
    masks=cohorts(row,y)
    codes=np.unique(group,return_inverse=True)[1]
    sizes=np.bincount(codes)
    def nll(prob):
        v=np.where(y,prob,1-prob)
        return -np.log(v.clip(1e-9,1))
    summary={}
    for source,experiments in all_probs.items():
        control=np.stack(experiments["scores"])
        summary[source]={}
        for name,probs in experiments.items():
            p=np.stack(probs)
            good=(p>=.5)==y
            accdiff=good.astype(float)-((control>=.5)==y)
            lossdiff=nll(control)-nll(p)
            group_acc=np.bincount(codes,weights=accdiff.mean(0))
            group_loss=np.bincount(codes,weights=lossdiff.mean(0))
            rng=np.random.default_rng(190)
            boot=[]
            for _ in range(32):
                bag=rng.integers(len(sizes),size=(64,len(sizes)))
                den=sizes[bag].sum(1)
                boot.extend(zip(100*group_acc[bag].sum(1)/den,group_loss[bag].sum(1)/den))
            summary[source][name]=dict(accuracy=good.mean(1).tolist(),fixed=(good&~ref).sum(1).tolist(),
                harmed=(~good&ref).sum(1).tolist(),stable={k:int((m&good.all(0)).sum()) for k,m in masks.items()},
                gain_pp_vs_score=100*accdiff.mean(),logloss_reduction=float(lossdiff.mean()),
                bootstrap95_vs_score=np.quantile(boot,[.025,.975],axis=0).tolist())
    return summary


def run():
    inv=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=inv.row.to_numpy(),inv.target.to_numpy(bool)
    full=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name"))
    group=full.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[row]
    group=group.to_numpy()[row]
    frame=compact_features(full.iloc[row])
    OUT.mkdir(exist_ok=True)
    all_probs={}
    distances={}
    refs=[]
    for seed in SEEDS:
        base,members,fold=predictions(row,y,seed,ACTIONS)
        refs.append(base)
        src={"registration":members[:,[list(ACTIONS).index("registration_only")]],"baseline":base[:,None],
             "full_panel":np.column_stack([base,members])}
        saved={}
        distances[seed]={}
        for name,scores in src.items():
            pbank={}
            distances[seed][name]=[]
            score=logit(scores.clip(.002,.998))
            for f in np.unique(fold):
                train,outer=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(group[train],group[outer]).size
                _,phi=design(frame,train)
                feat=np.column_stack([phi,np.log(size)])
                feat=((feat-feat[train].mean(0))/feat[train].std(0).clip(.25)).clip(-5,5)
                rng=np.random.default_rng(139)
                w=rng.normal(size=(feat.shape[1],64))/np.sqrt(feat.shape[1])
                x=np.column_stack([feat,np.cos(feat@w+rng.uniform(0,2*np.pi,64))])
                for a,b in GroupKFold(3,shuffle=True,random_state=387).split(outer,groups=group[outer]):
                    t,q=outer[a],outer[b]
                    assert not np.intersect1d(group[t],group[q]).size
                    ra,rb,z=innovation(score,x,t,q)
                    ja,_=donor(z[t],group[t])
                    jb,mismatch=donor(z[q],group[q])
                    distances[seed][name].append(float(np.median(mismatch)))
                    # Use the jointly tuned regression only once on real rows;
                    # apply it to swapped evaluation rows as a conditional test.
                    controls=score_fit(score[t],y[t],score[q])
                    sx=np.column_stack([score[t],ra])
                    model=LogisticRegression(C=.08,max_iter=400).fit(sx,y[t])
                    true=model.predict_proba(np.column_stack([score[q],rb]))[:,1]
                    # Replace raw fields before nuisance centering at the focal
                    # person's scores; in training and testing use separate swaps.
                    swapped_test=rb+x[q[jb]]-x[q]
                    swapped_train=ra+(x[t[ja]]-x[t])
                    alt=np.column_stack([score[q],swapped_test])
                    vals={"scores":controls,"real_fields":true,"swap_at_test":model.predict_proba(alt)[:,1],
                          "retrained_swap":score_fit(np.column_stack([score[t],swapped_train]),y[t],alt)}
                    for key,p in vals.items():
                        pbank.setdefault(key,np.full(len(y),np.nan))[q]=p
            for key,p in pbank.items():
                assert np.isfinite(p).all()
                all_probs.setdefault(name,{}).setdefault(key,[]).append(p)
                saved[name+"_"+key]=p
            print(seed,name,{k:round(float(((p>=.5)==y).mean()),4) for k,p in pbank.items()},flush=True)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**saved)
    write(REPORT/"result.json",dict(scores=summarize(all_probs,np.stack(refs),row,y,group),swap_distance=distances))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
