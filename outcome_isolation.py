"""Estimate class-conditional passenger atypicality with cross-fitted isolation forests."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import relational_error_corrector as ref
from peer_bill_correction import BILLS,load_v96,fit_offset,apply
from exaone_validation_data import CATEGORICAL

ROOT=Path(__file__).resolve().parent


def anomalies(frame, train, query, labels, seed):
    numeric=frame[["Age","CabinNumber",*BILLS]].copy()
    numeric[BILLS]=np.log1p(numeric[BILLS])
    med=numeric.iloc[train].median()
    encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(frame.iloc[train][CATEGORICAL])
    x=np.column_stack([encoder.transform(frame[CATEGORICAL]),numeric.fillna(med),numeric.isna()])
    values=[]
    for members in [train,train[labels==0],train[labels==1]]:
        fit=IsolationForest(n_estimators=96,max_samples=256,random_state=seed,n_jobs=2)
        fit.fit(x[members])
        values.append(-fit.score_samples(x[query]))
    return np.column_stack(values)


def design(scores, frame, probability, variant):
    pooled,neg,pos=scores.T
    exposure=pos-neg
    anomalies=np.column_stack([pooled]) if variant=="pooled" else np.column_stack([pooled,exposure])
    sleeping=frame.CryoSleep.eq("True").to_numpy()
    margin=np.asarray(probability)-.5
    return np.column_stack([anomalies,anomalies*sleeping[:,None],anomalies*margin[:,None]])


def run():
    inp,splits,folds,bases,evidence=ref.overlap.load_authenticated()
    dev,y,raw=inp["dev"],inp["y"],bases["rawMitra"]
    frame=inp["features"].iloc[dev].reset_index(drop=True)
    v96=load_v96(dev,folds)
    results={}
    predictions={(variant,base):np.empty(len(y)) for variant in ["pooled","conditional"] for base in ["raw","v96"]}
    for fold,(a,b) in enumerate(splits):
        inner=ref.load_inner(fold,a,b,dev,y,evidence)
        cross=np.zeros((len(a),3))
        for i,(s,q) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261117+fold).split(a,y[a])):
            cross[q]=anomalies(frame,a[s],a[q],y[a[s]],20261118+i)
        outer=anomalies(frame,a,b,y[a],20261118)
        for variant in ["pooled","conditional"]:
            pack=fit_offset(design(cross,frame.iloc[a],inner,variant),inner,y[a])
            for name,p in [("raw",raw),("v96",v96)]:
                shift=apply(pack,design(outer,frame.iloc[b],p[b],variant))
                predictions[(variant,name)][b]=expit(ref.logit(p[b])+shift)
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    for (variant,name),p in predictions.items():
        base=raw if name=="raw" else v96
        results[variant+"_"+name]=dict(metrics=ref.score(p,y),vs_base=ref.pair(p,base,y),
            vs_pooled=ref.pair(p,predictions[("pooled",name)],y),hard546=ref.pair(p,base,y,hard),
            fold_net=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(ROOT/"reports/outcome_isolation/result.json",results)
    print(results)


if __name__=="__main__":
    run()
