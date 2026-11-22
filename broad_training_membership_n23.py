"""Test which training passengers gain disproportionately from having their own booking included."""
from itertools import combinations
import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_boosting_path_n23 import ART,ROOT,SEEDS,load,historical,evaluate,interval,write,calibrated
from broad_refit_dissent_n23 import features

OUT=ART/"nov23_broad_training_membership"
REPORT=ROOT/"reports/nov23_broad_training_membership"


def model(n):
    return HistGradientBoostingClassifier(max_iter=n,max_leaf_nodes=20,learning_rate=.065,
               l2_regularization=3.,min_samples_leaf=18,early_stopping=False,random_state=948)


def membership(x,y,group):
    folds=np.full(len(y),-1)
    for f,(_,v) in enumerate(GroupKFold(4,shuffle=True,random_state=731).split(x,groups=group)):
        folds[v]=f
    seen=np.zeros((len(y),2))
    absent=np.zeros_like(seen)
    for ab in combinations(range(4),2):
        sel=np.isin(folds,ab)
        assert not np.intersect1d(group[sel],group[~sel]).size
        fit=model(240).fit(x[sel],y[sel])
        stages=[v[:,1] for i,v in enumerate(fit.staged_predict_proba(x),start=1) if i in (40,240)]
        prob=np.column_stack(stages)
        seen[sel]+=prob[sel]/3.
        absent[~sel]+=prob[~sel]/3.
    return seen,absent


def contrasts(seen,absent,y):
    sign=2*y.astype(float)-1
    gain=(seen-absent)*sign[:,None]
    growth=gain[:,1]-gain[:,0]
    # Compare cases with the same label and similar *out-of-training* score.
    # Both weighting directions and the placebo have identical total weight.
    matched=np.zeros(len(y))
    shuffled=np.zeros(len(y))
    rng=np.random.default_rng(625)
    parts=[]
    for label in (False,True):
        ids=np.flatnonzero(y==label)
        for block in np.array_split(ids[np.argsort(absent[ids,1],kind="stable")],10):
            parts.append(block)
            matched[block]=(rankdata(growth[block])-.5)/len(block)-.5
            shuffled[block]=rng.permutation(matched[block])
    return gain,growth,matched,shuffled,parts


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    x=features(frame)[0]
    OUT.mkdir(exist_ok=True)
    report,history={},{}
    control=None
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for label,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[label],v)
            base,fold,ref=z["base"],z["fold"],z["ref"]
        scores={}
        diagnostics=[]
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            seen,absent=membership(x[t],y[t],g[t])
            gain,growth,rank,sham,_=contrasts(seen,absent,y[t])
            diagnostics.append(dict(train=len(t),mean_membership_gain=gain.mean(0).tolist(),
                       correctly_classified_only_when_included=(((seen>=.5)==y[t,None])&((absent>=.5)!=y[t,None])).sum(0).tolist()))
            if control is None:
                # Simulated corruption occurs only inside this diagnostic's
                # training population. Its scores are never reused downstream.
                rng=np.random.default_rng(437)
                corrupt=rng.random(len(t))<.15
                noisy=y[t]^corrupt
                si,ao=membership(x[t],noisy,g[t])
                dif,_,normalized,_,_=contrasts(si,ao,noisy)
                difficulty=np.where(noisy,1-ao[:,1],ao[:,1])
                control=dict(n=len(t),flips=int(corrupt.sum()),difficulty_auc=float(roc_auc_score(corrupt,difficulty)),
                             membership_auc=float(roc_auc_score(corrupt,dif[:,1])),
                             matched_growth_auc=float(roc_auc_score(corrupt,normalized)))
            variants={"ordinary":np.ones(len(t)),"favor_shared":1-1.6*rank,
                      "favor_members":1+1.6*rank,"shuffled":1+1.6*sham,"reverse_shuffled":1-1.6*sham}
            for s,w in variants.items():
                assert np.isclose(w.mean(),1.) and w.min()>0
                f=model(240).fit(x[t],y[t],sample_weight=w)
                p=f.predict_proba(x[q])[:,1]
                scores.setdefault(s,np.full(len(y),np.nan))[q]=p
            print(name,"membership fold",k,flush=True)
        anchors=np.column_stack([logit(base.clip(.002,.998)),logit(scores["ordinary"].clip(.002,.998))])
        for policy,terms in (("ordinary",[]),("membership",["favor_shared","favor_members"]),("sham",["shuffled","reverse_shuffled"])):
            ds=np.column_stack([anchors,*[logit(scores[s].clip(.002,.998))-anchors[:,1] for s in terms]])
            p=np.full(len(y),np.nan)
            for k in np.unique(fold):
                ids=np.flatnonzero(fold==k)
                for a,b in GroupKFold(3,shuffle=True,random_state=436).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(ds[t],ds[q],y,t)
            scores["update_"+policy]=p
        summaries,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        report[name]=dict(models=summaries,diagnostics=diagnostics)
        if name!="v131":
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        write(REPORT/"results.json",dict(plans=report,control=control))
    stable={}
    for k,rows in history.items():
        good=np.logical_and.reduce([v for v,_ in rows])
        stable[k]={s:int((m&good).sum()) for s,m in masks.items()}
        stable[k]["gain_ci95_pp"]=interval(np.mean([d for _,d in rows],0),g)
    write(REPORT/"results.json",dict(plans=report,control=control,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
