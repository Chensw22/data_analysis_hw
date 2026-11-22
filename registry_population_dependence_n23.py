"""Compare registry-to-itinerary dependence in paying and non-paying manifest populations."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,verify
from exaone_validation_data import compact_features
from population_error_accounting_n23 import BILLS
from relational_error_corrector import write


def counterexamples(t, strata, groups, rng):
    left,right=[],[]
    for s in np.unique(strata[t]):
        part=t[strata[t]==s]
        # Derange at booking level by cyclic rotation of contiguous booking
        # blocks. Every eligible itinerary occurs exactly once as a donor.
        keys,inv,n=np.unique(groups[part],return_inverse=True,return_counts=True)
        largest=n.max()
        if 2*largest>len(part):
            continue
        order=np.lexsort((rng.random(len(part)),rng.permutation(len(keys))[inv]))
        a=part[order]
        b=np.roll(a,rng.integers(largest,len(part)-largest+1))
        assert np.all(groups[a]!=groups[b])
        left.extend(a);right.extend(b)
    return np.asarray(left),np.asarray(right)


def transductive(raw):
    x=compact_features(raw)
    party=raw.PassengerId.str.split("_").str[0]
    size=party.map(party.value_counts()).to_numpy()
    strata=pd.DataFrame({"origin":x.HomePlanet,"size":np.minimum(size,3)}).groupby(["origin","size"]).ngroup().to_numpy()
    # Spend is used only to assign source populations; the discriminators see
    # the two registration blocks and never a bill or a competition label.
    paying=raw[BILLS].gt(0).any(axis=1).to_numpy()
    group=party.to_numpy()
    a=np.column_stack([pd.get_dummies(x[["HomePlanet","CryoSleep","VIP"]],dtype=float),x.Age,size])
    b=np.column_stack([pd.get_dummies(x[["Deck","Side","Destination"]],dtype=float),x.CabinNumber])
    age=x.Age.fillna(-1).to_numpy()
    cabin=x.CabinNumber.fillna(-1).to_numpy()
    a=np.column_stack([a,np.digitize(age,[0,7,13,18,25,40,60])])
    b=np.column_stack([b,np.digitize(cabin,np.arange(0,2000,100))])
    def crosses(left,right):
        # Exchanging whole blocks exactly preserves every marginal. Explicit
        # pair terms allow a tree to detect dependence on its first split.
        aa=np.nan_to_num(a[left],nan=-1.)
        bb=np.nan_to_num(b[right],nan=-1.)
        return np.column_stack([aa,bb,(aa[:,:,None]*bb[:,None,:]).reshape(len(left),-1)])
    full=crosses(np.arange(len(raw)),np.arange(len(raw)))
    names=("pooled","pays","no_observed_payment","random_pays","random_nonpayer")
    features={n:np.full(len(raw),np.nan) for n in names}
    auc={n:[] for n in names}
    for fold,(train,query) in enumerate(GroupKFold(5,shuffle=True,random_state=522).split(raw,groups=group)):
        assert not np.intersect1d(group[train],group[query]).size
        rng=np.random.default_rng(612+fold)
        random=paying.copy()
        for sub in (train,query):
            for s in np.unique(strata[sub]):
                ids=sub[strata[sub]==s]
                random[ids]=rng.permutation(paying[ids])
        subsets=[np.ones(len(raw),bool),paying,~paying,random,~random]
        for name,mask in zip(names,subsets):
            left,right=counterexamples(train[mask[train]],strata,group,rng)
            data=np.vstack([full[left],crosses(left,right)])
            labels=np.repeat([1,0],len(left))
            model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=11,l2_regularization=7,
                min_samples_leaf=25,learning_rate=.07,early_stopping=False,random_state=101)
            model.fit(data,labels)
            features[name][query]=model.decision_function(full[query])
            l,r=counterexamples(query[mask[query]],strata,group,rng)
            negative=model.decision_function(crosses(l,r))
            auc[name].append(float(roc_auc_score(np.repeat([1,0],len(l)),np.r_[features[name][l],negative])))
    assert all(np.isfinite(v).all() for v in features.values())
    return features,auc


def calibrate(base,metrics,y,t,q):
    risk=logit(base.clip(.001,.999))
    x=np.column_stack([np.ones(len(y)),risk])
    if metrics.shape[1]:
        v=((metrics-metrics[t].mean(0))/metrics[t].std(0).clip(.15)).clip(-5,5)
        x=np.column_stack([x,v,v*(base[:,None]-.5)])
    penalty=np.full(x.shape[1],55.)
    penalty[:2]=[10.,45.]
    def loss(w):
        z=risk[t]+x[t]@w
        return ((np.logaddexp(0,z).sum()-z@y[t]+.5*(penalty*w)@w)/len(t),
                (x[t].T@(expit(z)-y[t])+penalty*w)/len(t))
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method="BFGS",options={"gtol":1.e-6})
    assert fit.success
    return expit(risk[q]+x[q]@fit.x)


def run():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Transported","Name"})
                   for s in ("train","test")],ignore_index=True)
    scores,auc=transductive(raw)
    inv=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=inv.row.to_numpy(),inv.target.to_numpy(bool)
    party=raw.PassengerId.str.split("_").str[0].iloc[row].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()[row]
    ref={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            p,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as z:
            verify(z,row,y,fold)
            np.testing.assert_array_equal(z["baseline"],p)
            ref[seed]=(p,fold,z["eligible"])
    repeated=np.logical_and.reduce([(p>=.5)!=y for p,_,_ in ref.values()])
    out=ART/"nov23_registry_population_dependence"
    out.mkdir(exist_ok=True)
    report={}
    for seed,(base,fold,eligible) in ref.items():
        hard=earth & repeated & eligible
        assert hard.sum()==430
        designs={"calibration":[],"pooled":["pooled"],"population":["pooled","pays","no_observed_payment"],
                 "random_population":["pooled","random_pays","random_nonpayer"]}
        pred={}
        metrics={}
        old=(base>=.5)==y
        for method,columns in designs.items():
            proj=np.column_stack([scores[c][row] for c in columns]) if columns else np.empty((len(y),0))
            p=np.full(len(y),np.nan)
            for f in np.unique(fold):
                outer=np.flatnonzero(fold==f)
                for tr,te in GroupKFold(3,shuffle=True,random_state=721).split(outer,groups=party[outer]):
                    t,q=outer[tr],outer[te]
                    assert not np.intersect1d(party[t],party[q]).size
                    p[q]=calibrate(base,proj,y,t,q)
            assert np.isfinite(p).all()
            pred[method]=p
            ok=(p>=.5)==y
            metrics[method]={}
            for n,m in [("all",np.ones(len(y),bool)),("identical",eligible),("earth_repeat430",hard)]:
                fixed=ok&~old&m
                harmed=~ok&old&m
                metrics[method][n]=dict(rows=int(m.sum()),accuracy=float(ok[m].mean()),fixed=int(fixed.sum()),
                    harmed=int(harmed.sum()),fixed_fn=int((fixed&y).sum()),fixed_fp=int((fixed&~y).sum()),
                    new_fn=int((harmed&y).sum()),new_fp=int((harmed&~y).sum()))
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,hard=hard,**pred)
        report[seed]=metrics
        print(seed,metrics["population"],flush=True)
    write(ROOT/"reports/nov23_registry_population_dependence/result.json",dict(results=report,registry_integrity_auc=auc))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
