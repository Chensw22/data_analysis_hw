"""Class-subsampled feature-block PCA followed by full-feature CART ensembles."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from fresh_family_rescue_n23 import NEW

TREES = 96


def rotate_and_fit(x, y, t, q, seed):
    rng = np.random.default_rng(seed)
    total = np.zeros(len(q))
    for i in range(TREES):
        rotation = np.zeros((x.shape[1],x.shape[1]))
        for cols in np.array_split(rng.permutation(x.shape[1]), int(np.ceil(x.shape[1]/3))):
            # For binary labels the nonempty class subsets are {0}, {1}, {0,1}.
            choice = rng.integers(3)
            support = t if choice == 2 else t[y[t] == choice]
            bootstrap = rng.choice(support, round(.75*len(support)), replace=True)
            sample = x[np.ix_(bootstrap,cols)]
            centered = sample - sample.mean(0)
            _, vectors = np.linalg.eigh(centered.T @ centered)
            rotation[np.ix_(cols,cols)] = vectors
        transformed = x @ rotation
        tree = DecisionTreeClassifier(min_samples_leaf=4,random_state=seed+i)
        tree.fit(transformed[t], y[t])
        total += tree.predict_proba(transformed[q])[:,1]
    return total/TREES


def run():
    rows,y,base,old,_,_,_=panel()
    others=[predictions(rows,y,s,dict(**NEW,native_carte=("carte","p"))) for s in SEEDS]
    historical = (np.concatenate([old,np.stack([v[1] for v in others])],axis=2)>=.5)==y[None,:,None]
    correct_base=(base>=.5)==y
    hard=~correct_base.any(0)
    ever=~correct_base.all(0)
    never=hard & ~historical.any(axis=(0,2))
    no_stable=hard & ~historical.all(0).any(1)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Transported","Name")).iloc[rows]
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    frame=compact_features(raw)
    out=ART/"nov23_broad_rotation_forest"
    out.mkdir(exist_ok=True)
    hits,metrics={},{}
    for i,(seed,(b,_,fold)) in enumerate(zip(SEEDS,others)):
        np.testing.assert_array_equal(b,base[i])
        scores={k:np.full(len(y),np.nan) for k in ("rotation","random_forest")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t],groups[q]).size
            x,_=design(frame,t)
            x=StandardScaler().fit(x[t]).transform(x)
            scores["rotation"][q]=rotate_and_fit(x,y,t,q,1821+int(f))
            forest=RandomForestClassifier(n_estimators=TREES,min_samples_leaf=4,n_jobs=2,random_state=1821+int(f))
            scores["random_forest"][q]=forest.fit(x[t],y[t]).predict_proba(x[q])[:,1]
        scores["rotation_quarter"]=(scores["rotation"]+3*b)/4
        scores["forest_quarter"]=(scores["random_forest"]+3*b)/4
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,**scores)
        metrics[seed]={}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            hits.setdefault(k,[]).append(ok)
            metrics[seed][k]=dict(accuracy=ok.mean(),fixed=int((ok & ~correct_base[i]).sum()),
                                  harmed=int((~ok & correct_base[i]).sum()))
        print(seed,metrics[seed],flush=True)
    coverage={}
    for k,h in hits.items():
        ok=np.asarray(h)
        coverage[k]=dict(ever_stable=int((ever & ok.all(0)).sum()),
                         persistent_stable=int((hard & ok.all(0)).sum()),newly_stable=int((no_stable & ok.all(0)).sum()),
                         never_any=int((never & ok.any(0)).sum()),never_stable=int((never & ok.all(0)).sum()))
    out=ROOT/"reports/nov23_broad_rotation_forest"
    out.mkdir(exist_ok=True)
    (out/"result.json").write_text(json.dumps(dict(metrics=metrics,coverage=coverage),indent=2)+"\n")


if __name__=="__main__":
    with threadpool_limits(2):
        run()
