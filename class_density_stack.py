"""Class-conditional dependency trees and nested out-of-fold probability stacking."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/top3_stage"


def discretize(train, query):
    columns = []
    sizes = []
    for name in train:
        if name in CATEGORICAL:
            levels = sorted(train[name].unique())
            codes = [pd.Categorical(part[name], categories=levels).codes.astype(int)+1 for part in (train, query)]
            size = len(levels)+1
        else:
            positive = train.loc[train[name] > 0, name]
            cuts = np.unique(positive.quantile([.125,.25,.375,.5,.625,.75,.875])) if len(positive) else []
            def encode(values):
                return np.where(values.isna(), 0, np.where(values==0, 1, 2+np.searchsorted(cuts,values)))
            codes = [encode(part[name]) for part in (train,query)]
            size = 3+len(cuts)
        columns.append(codes)
        sizes.append(size)
    return *(np.column_stack([col[i] for col in columns]) for i in (0,1)), sizes


def predict_density(train, labels, query):
    x,q,sizes = discretize(train, query)
    scores = []
    for cls in (0,1):
        rows = x[labels==cls]
        n,d = rows.shape
        affinity = np.zeros((d,d))
        for i in range(d):
            for j in range(i):
                affinity[j,i] = affinity[i,j] = 1+mutual_info_score(rows[:,i],rows[:,j])
        tree = minimum_spanning_tree(-affinity)
        order, parent = breadth_first_order(tree,0,directed=False)
        assert len(order)==d
        # Dirichlet smoothing toward the per-class marginal for sparse parent levels.
        logp = np.full(len(q),np.log(n/len(labels)))
        for child in order:
            width = sizes[child]
            marg = (np.bincount(rows[:,child],minlength=width)+1)/(n+width)
            if child==0:
                prob = marg[q[:,child]]
            else:
                par = parent[child]
                counts = np.zeros((sizes[par],width))
                np.add.at(counts,(rows[:,par],rows[:,child]),1)
                conditional = (counts+16*marg)/(counts.sum(axis=1)[:,None]+16)
                prob = conditional[q[:,par],q[:,child]]
            logp += np.log(prob)
        scores.append(logp)
    return expit(scores[1]-scores[0])


def stack(train, y, query):
    def transform(values):
        return logit(np.clip(np.column_stack(values),1e-6,1-1e-6))
    model = make_pipeline(StandardScaler(),LogisticRegression(C=1))
    model.fit(transform(train),y)
    return model.predict_proba(transform(query))[:,1]


def run():
    inputs,splits,_,bases,evidence = ref.overlap.load_authenticated()
    dev,y = inputs["dev"],inputs["y"]
    x = inputs["features"].iloc[dev]
    raw,kumo = bases["rawMitra"],bases["Kumo"]
    names = ["density","two_model_stack","three_model_stack"]
    pred = {key:np.full(len(y),np.nan) for key in names}
    for fold,(a,b) in enumerate(splits):
        inner_raw = ref.load_inner(fold,a,b,dev,y,evidence)
        cache = ROOT/f"artifacts/planet_mixture_20261117/outer{fold}_inner_oof.npz"
        with np.load(cache) as saved:
            inner_kumo, fold_ids = saved["kumo"], saved["inner_fold_id"]
            np.testing.assert_array_equal(saved["outer_train_ids"],dev[a])
        inner_density = np.full(len(a),np.nan)
        for idx in range(3):
            tr,va = a[fold_ids != idx], a[fold_ids == idx]
            inner_density[fold_ids==idx] = predict_density(x.iloc[tr],y[tr],x.iloc[va])
        base = predict_density(x.iloc[a],y[a],x.iloc[b])
        pred["density"][b] = base
        inner=[inner_raw,inner_kumo]
        outer=[raw[b],kumo[b]]
        pred["two_model_stack"][b] = stack(inner,y[a],outer)
        pred["three_model_stack"][b] = stack(inner+[inner_density],y[a],outer+[base])
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    np.testing.assert_array_equal(catalog.y,y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    assert hard.sum()==546
    result = {k:dict(metrics=ref.score(p,y), versus_raw=ref.pair(p,raw,y),
                     hard546=ref.pair(p,raw,y,hard), folds=[ref.pair(p[b],raw[b],y[b])['net_correct'] for _,b in splits])
              for k,p in pred.items()}
    result["three_vs_two"] = ref.pair(pred["three_model_stack"],pred["two_model_stack"],y)
    np.savez_compressed(ROOT/"artifacts/top3_stage/density_oof.npz",dev=dev,**pred)
    ref.write(OUT/"density_result.json",result)
    print(json.dumps(result))


if __name__ == "__main__":
    run()
