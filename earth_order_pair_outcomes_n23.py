"""Predict the four outcome states of pairs selected by bills before consulting any labels."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from class_analogue_experts_n23 import design
from earth_bill_order_n23 import BILLS
from exaone_validation_data import compact_features
from relational_error_corrector import write


def partners(x, bills, mask, groups, query, training):
    left, right = [], []
    for i in query:
        allowed = training[(groups[training] != groups[i]) & (mask[training] == mask[i])]
        delta = bills[allowed]-bills[i]
        below, above = (delta <= 0).all(1), (delta >= 0).all(1)
        for select in (below & ~above, above & ~below, above & below):
            available = allowed[select]
            if len(available):
                dist = cdist(x[i:i+1],x[available],"sqeuclidean").ravel()
                near = available[np.argsort(dist,kind="stable")[:4]]
                left.extend([i]*len(near))
                right.extend(near)
    return np.asarray(left), np.asarray(right)


def features(x,i,j):
    return np.column_stack([x[i], x[j], x[i]-x[j]]).astype(np.float32)


def run():
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        row,y = z["row"],z["y"]
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda k:k not in {"Name","Transported"})
    group = raw.PassengerId.str.split("_").str[0]
    size = group.map(group.value_counts()).to_numpy()[row]
    group = group.to_numpy()[row]
    raw = raw.iloc[row].reset_index(drop=True)
    frame = compact_features(raw)
    earth = raw.HomePlanet.eq("Earth").to_numpy()
    bills = raw[BILLS].fillna(0).to_numpy()*[1,1,-1,-1,-1]
    mask = raw[BILLS].isna().to_numpy() @ (2**np.arange(5))
    data={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            data[seed]=z["base"],z["fold"]
    repeat=np.logical_and.reduce([(base>=.5)!=y for base,_ in data.values()])&earth
    results={}
    out=ART/"nov23_earth_order_pair_outcomes"
    out.mkdir(exist_ok=True)
    for seed,(base,fold) in data.items():
        predictions={key:base.copy() for key in ("marginal","conditioned","shuffled_outcome")}
        details=[]
        for f in np.unique(fold):
            t=np.flatnonzero(earth&(fold!=f))
            q=np.flatnonzero(earth&(fold==f))
            assert not np.intersect1d(group[t],group[q]).size
            coords,x=design(frame,t)
            x=np.column_stack([x,np.log1p(size)])
            i,j=partners(coords,bills,mask,group,t,t)
            model=CatBoostClassifier(loss_function="MultiClass",iterations=225,depth=5,learning_rate=.065,
                    thread_count=2,random_seed=918,verbose=False,allow_writing_files=False)
            model.fit(features(x,i,j),2*y[i].astype(int)+y[j],sample_weight=1/np.bincount(i,minlength=len(y))[i])
            assert np.array_equal(model.classes_,[0,1,2,3])
            i,j=partners(coords,bills,mask,group,q,t)
            joint=model.predict_proba(features(x,i,j))
            marginal=joint[:,2]+joint[:,3]
            given_no=joint[:,2]/(joint[:,0]+joint[:,2])
            given_yes=joint[:,3]/(joint[:,1]+joint[:,3])
            null=y.copy()
            rng=np.random.default_rng(496)
            null[t]=rng.permutation(y[t])
            counts=np.bincount(i,minlength=len(y))
            active=counts>0
            for key,p in [("marginal",marginal),("conditioned",np.where(y[j],given_yes,given_no)),
                          ("shuffled_outcome",np.where(null[j],given_yes,given_no))]:
                predictions[key][active]=np.bincount(i,weights=p,minlength=len(y))[active]/counts[active]
            strict=np.any(bills[i]!=bills[j],axis=1)
            query_below=(bills[i]<=bills[j]).all(1)
            # Outcomes of labelled training partners are observable during prediction.
            # A strict reversal has positive outcome below and negative outcome above.
            possible=strict&np.where(query_below,~y[j],y[j])
            is_reversal=np.where(query_below,y[i],~y[i])
            risk=np.where(query_below,given_no,1-given_yes)
            details.append(dict(query_people=len(q),scored_people=int(active.sum()),unique_pairs=len(i),
                eligible_strict_pairs=int(possible.sum()),observed_reversals=int(is_reversal[possible].sum()),
                reversal_auc=float(roc_auc_score(is_reversal[possible],risk[possible])),
                reversal_ap=float(average_precision_score(is_reversal[possible],risk[possible]))))
        for name,p in list(predictions.items()):
            predictions[name+"_quarter"]=.75*base+.25*p
        scores={}
        before=(base>=.5)==y
        for name,p in predictions.items():
            ok=(p>=.5)==y
            scores[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before).sum()),harmed=int((~ok&before).sum()),
                              repeat=int((repeat&ok).sum()))
        results[seed]=dict(scores=scores,folds=details)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**predictions)
        print(seed,scores["conditioned"],flush=True)
    write(ROOT/"reports/nov23_earth_order_pair_outcomes/result.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
