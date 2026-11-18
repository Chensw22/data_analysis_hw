"""Compare pooled training, solo-only and covariate-shift transfer for solo passengers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL, compact_features
from observation_collision_audit import BILLS, metrics

ROOT = Path(__file__).resolve().parent
MODES = ["pooled", "solo_only", "solo_odds_transfer"]


def transfer_weights(x, solo, groups):
    # In each split estimate target/source density ratio using only other parties.
    weights = np.ones(len(x))
    for train, val in GroupKFold(3).split(x, groups=groups):
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=12,
                                                min_samples_leaf=35, random_state=20261118)
        model.fit(x.iloc[train], solo[train])
        source = val[~solo[val]]
        p = np.clip(model.predict_proba(x.iloc[source])[:, 1], .01, .99)
        prior = solo[train].mean()
        weights[source] = np.clip((p / (1-p)) * ((1-prior) / prior), .1, 10)
    weights[~solo] /= weights[~solo].mean()
    return weights


def run():
    art = ROOT / "artifacts"
    with np.load(art / "v108_exact_oof/oof.npz") as z:
        dev, fold, y, prob, ids = (z[k] for k in ["dev_ids","fold_id","true","v108","PassengerId"])
    y = y.astype(bool)
    with np.load(art / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(dev,z["dev"])
        np.testing.assert_array_equal(fold,z["fold"])
        base = (prob >= .5) ^ z["combined_by_historical_agreement"]
    catalog = pd.read_csv(art / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.PassengerId,ids)
    np.testing.assert_array_equal(catalog.y,y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in ["Name","Transported"])
    test = pd.read_csv(ROOT / "data/raw/test.csv",usecols=["PassengerId"])
    groups = raw.PassengerId.str.split("_").str[0]
    roster_sizes = pd.concat([groups,test.PassengerId.str.split("_").str[0]]).value_counts()
    solo = groups.map(roster_sizes).iloc[dev].eq(1).to_numpy()
    groups = groups.iloc[dev].to_numpy()
    frame = raw.iloc[dev].reset_index(drop=True)
    np.testing.assert_array_equal(frame.PassengerId,ids)
    x = compact_features(frame)
    x["spend_known_sum"] = frame[BILLS].sum(axis=1,min_count=1)
    x["spend_missing_count"] = frame[BILLS].isna().sum(axis=1)
    x["spend_positive_count"] = frame[BILLS].gt(0).sum(axis=1)
    domain = pd.get_dummies(x, columns=CATEGORICAL, dtype=float)
    pred = {m: np.full(len(y),np.nan) for m in MODES}
    support = []
    for f in np.unique(fold):
        train = np.flatnonzero(fold != f)
        q = np.flatnonzero((fold == f) & solo)
        assert not set(groups[train]) & set(groups[q])
        weights = transfer_weights(domain.iloc[train],solo[train],groups[train])
        for m in MODES:
            fit = train[solo[train]] if m == "solo_only" else train
            model = CatBoostClassifier(iterations=700,depth=6,learning_rate=.045,l2_leaf_reg=5,
                                       random_seed=20261118,thread_count=2,verbose=False,
                                       allow_writing_files=False)
            model.fit(x.iloc[fit],y[fit],cat_features=CATEGORICAL,
                      sample_weight=weights if m == "solo_odds_transfer" else None)
            pred[m][q] = model.predict_proba(x.iloc[q])[:,1]
        support.append(dict(solo=int(solo[train].sum()),multi=int((~solo[train]).sum()),
                            multi_effective_n=float(weights[~solo[train]].sum()**2 /
                                                    (weights[~solo[train]]**2).sum())))
    records = {}
    for m in MODES:
        assert np.isfinite(pred[m][solo]).all() and np.isnan(pred[m][~solo]).all()
        for gate in ["always","confident"]:
            use = solo if gate == "always" else solo & (np.abs(pred[m] - .5) >= .3)
            ans = np.where(use,pred[m] >= .5,base)
            records[m+"_"+gate] = dict(**metrics(ans,base,y,fold,hard),
                                       solo_correct=int((ans[solo] == y[solo]).sum()),
                                       solo_accuracy=float((ans[solo] == y[solo]).mean()))
    positive = np.all(np.stack([pred[m] >= .8 for m in MODES]), axis=0)
    negative = np.all(np.stack([pred[m] <= .2 for m in MODES]), axis=0)
    joint = np.where(positive|negative,positive,base)
    records["all_three_confident"] = dict(**metrics(joint,base,y,fold,hard),
                                           solo_correct=int((joint[solo] == y[solo]).sum()),
                                           solo_accuracy=float((joint[solo] == y[solo]).mean()))
    result = dict(solo_count=int(solo.sum()),solo_baseline_correct=int((base[solo]==y[solo]).sum()),
                  baseline_correct=int((base==y).sum()),catboost_features=list(x.columns),
                  confidence_cutoff=.8,support=support,results=records)
    out = ROOT / "reports/solo_transfer_experiment/result.json"
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(records,indent=2))


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
