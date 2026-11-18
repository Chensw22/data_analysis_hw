"""Learn when to flip raw Mitra, using only out-of-fold evidence from the training partition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import mechanism_combo as gam
import class_density_stack as density
import relational_error_corrector as ref

ROOT = Path(__file__).resolve().parent


def arguments(probs, rows):
    p = np.column_stack(probs)
    label = p[:,0]>=.5
    against = ((p>=.5)!=label[:,None]).any(axis=1)
    # Positive evidence supports the original class, negative evidence challenges it.
    logits = logit(np.clip(p,1e-6,1-1e-6)) * np.where(label,1,-1)[:,None]
    stats = np.column_stack([logits,label,rows.CryoSleep.eq("True"),rows.HomePlanet.eq("Earth"),
                             rows.Age,rows.CabinNumber,rows.Side.eq("P")])
    return stats,against


def run():
    inputs,splits,_,bases,evidence = ref.overlap.load_authenticated()
    dev,y = inputs["dev"],inputs["y"]
    x = inputs["features"].iloc[dev]
    raw = bases["rawMitra"]
    outer = [raw,bases["Kumo"]]
    for file,key,ids in [("top3_stage/density_oof.npz","density","dev"),
                         ("mechanism_combo_empty_query_fix/oof.npz","mechanism","dev_ids")]:
        with np.load(ROOT/"artifacts"/file) as z:
            np.testing.assert_array_equal(z[ids],dev)
            outer.append(z[key])
    output = (raw>=.5).copy()
    eligible = np.zeros(len(y),bool)
    for fold,(a,b) in enumerate(splits):
        inner_raw = ref.load_inner(fold,a,b,dev,y,evidence)
        with np.load(ROOT/f"artifacts/planet_mixture_20261117/outer{fold}_inner_oof.npz") as z:
            inner_fold,inner_kumo = z["inner_fold_id"],z["kumo"]
        tan,mech = np.full(len(a),np.nan),np.full(len(a),np.nan)
        for j in range(3):
            tr,va = a[inner_fold!=j],a[inner_fold==j]
            tan[inner_fold==j] = density.predict_density(x.iloc[tr],y[tr],x.iloc[va])
            design,theta,_ = gam.fit_model(x.iloc[tr],y[tr])
            mech[inner_fold==j] = expit(design.transform(x.iloc[va])@theta)
        z,mask = arguments([inner_raw,inner_kumo,tan,mech],x.iloc[a])
        target = (inner_raw>=.5)!=y[a]
        gate = HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=5,min_samples_leaf=60,
                        l2_regularization=8,learning_rate=.06,early_stopping=False,random_state=20261112)
        gate.fit(z[mask],target[mask])
        z,mask = arguments([p[b] for p in outer],x.iloc[b])
        eligible[b] = mask
        output[b[mask]] ^= gate.predict_proba(z[mask])[:,1]>=.5
    cat = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cat.row_id,dev)
    hard = cat.all_recipes_still_wrong.to_numpy(bool)
    result = dict(accuracy=float((output==y).mean()),challenged=int(eligible.sum()),challenged_hard=int((eligible&hard).sum()),
                  paired=ref.pair(output,raw,y),hard546=ref.pair(output,raw,y,hard),
                  folds=[ref.pair(output[b],raw[b],y[b]) for _,b in splits])
    ref.write(ROOT/"reports/top3_stage/router.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    with threadpool_limits(4):
        run()
