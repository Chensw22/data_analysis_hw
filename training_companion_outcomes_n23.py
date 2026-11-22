"""Learn from companion outcomes during training, then predict from observed fields alone."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from earth_bill_order_n23 import BILLS
from exaone_validation_data import compact_features
from relational_error_corrector import write


def peer_info(groups, sleep, paid, outcomes):
    # Only other observed members of the same booking contribute.
    _,g=np.unique(groups,return_inverse=True)
    cols=[]
    for mask in [np.ones(len(g),bool),sleep,paid]:
        num=np.bincount(g,weights=mask)[g]-mask
        support=outcomes*mask
        total=np.bincount(g,weights=support)[g]-support
        cols.extend([np.log1p(num),np.divide(total,num,out=np.full_like(total,np.nan,dtype=float),where=num>0)])
    return np.column_stack(cols)


def classifier(x,labels,soft=False):
    clf=HistGradientBoostingClassifier(max_iter=145,max_leaf_nodes=15,min_samples_leaf=20,
                         l2_regularization=2,early_stopping=False,random_state=443)
    if soft:
        n=len(labels)
        # The two weighted labels implement Bernoulli cross entropy with a soft target.
        return clf.fit(np.vstack([x,x]),np.r_[np.ones(n),np.zeros(n)],
                       sample_weight=np.r_[labels,1-labels])
    return clf.fit(x,labels)


def run():
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            data[seed]={k:z[k] for k in ("row","y","base","fold")}
    row,y=data[SEEDS[0]]["row"],data[SEEDS[0]]["y"]
    full=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    group=full.PassengerId.str.split("_").str[0]
    size=group.map(group.value_counts()).to_numpy()[row]
    group=group.to_numpy()[row]
    raw=full.iloc[row].reset_index(drop=True)
    frame=compact_features(raw)
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    paid=raw[BILLS].gt(0).any(axis=1).to_numpy()
    sleeping=raw.CryoSleep.eq(True).to_numpy()
    repeat=np.logical_and.reduce([(v["base"]>=.5)!=y for v in data.values()])
    out=ART/"nov23_training_companion_outcomes"
    out.mkdir(exist_ok=True)
    report={}
    def augmented(x,idx,values):
        return np.column_stack([x[idx],peer_info(group[idx],sleeping[idx],paid[idx],values)])
    observable_peers=pd.Series(group).map(pd.Series(group).value_counts()).to_numpy()>1
    for seed,v in data.items():
        np.testing.assert_array_equal(v["row"],row);np.testing.assert_array_equal(v["y"],y)
        base,fold=v["base"],v["fold"]
        names=["ordinary","ordinary_distilled","companion_distilled","simulated_companions","revealed_diagnostic"]
        pred={k:np.full(len(y),np.nan) for k in names}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            _,phi=design(frame,t)
            x=np.column_stack([phi,np.log1p(size)])
            targets={k:np.full(len(t),np.nan) for k in ("ordinary_distilled","companion_distilled")}
            for a,b in GroupKFold(3,shuffle=True,random_state=982).split(t,groups=group[t]):
                fit,ask=t[a],t[b]
                assert not np.intersect1d(group[fit],group[ask]).size
                teach=classifier(augmented(x,fit,y[fit]),y[fit])
                targets["companion_distilled"][b]=teach.predict_proba(augmented(x,ask,y[ask]))[:,1]
                targets["ordinary_distilled"][b]=classifier(x[fit],y[fit]).predict_proba(x[ask])[:,1]
            for name,target in targets.items():
                assert np.isfinite(target).all()
                pred[name][q]=classifier(x[t],.5*y[t]+.5*target,soft=True).predict_proba(x[q])[:,1]
            ordinary=classifier(x[t],y[t])
            p=ordinary.predict_proba(x[q])[:,1]
            pred["ordinary"][q]=p
            teacher=classifier(augmented(x,t,y[t]),y[t])
            rng=np.random.default_rng(619)
            samples=[teacher.predict_proba(augmented(x,q,rng.random(len(q))<p))[:,1] for _ in range(24)]
            pred["simulated_companions"][q]=np.mean(samples,axis=0)
            # Separate hypothetical test of releasing the outcomes of other query passengers.
            # This is evaluated only after all feasible predictions for this fold have been made.
            pred["revealed_diagnostic"][q]=teacher.predict_proba(augmented(x,q,y[q]))[:,1]
        for name in ("ordinary","ordinary_distilled","companion_distilled","simulated_companions"):
            pred[name+"_quarter"]=.75*base+.25*pred[name]
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as old:
            np.testing.assert_array_equal(row,old["row"])
            hard=earth&repeat&old["eligible"]
        scores={}
        for name,p in pred.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            before=(base>=.5)==y
            scores[name]=dict(accuracy=float(ok.mean()),peer_accuracy=float(ok[observable_peers].mean()),
                        fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),repeat=int((ok&repeat).sum()),
                        earth430=int((ok&hard).sum()))
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**pred)
        report[seed]=scores
        write(ROOT/"reports/nov23_training_companion_outcomes/result.json",
              dict(with_development_companion=int(observable_peers.sum()),results=report))
        print(seed,scores["companion_distilled"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
