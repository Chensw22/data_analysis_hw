"""Predict a party's excess transported count from its unlabeled manifest."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import expit

import relational_error_corrector as ref
from peer_bill_correction import BILLS, load_v96

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports/party_outcome_count"


def manifest():
    fields = ["PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Age", *BILLS]
    frame = pd.concat([pd.read_csv(ROOT / "data/raw" / f, usecols=fields)
                       for f in ["train.csv", "test.csv"]], ignore_index=True)
    group = frame.PassengerId.str.split("_").str[0].astype(int)
    deck = frame.Cabin.str.split("/").str[0]
    bill = frame[BILLS]
    age = frame.Age
    attributes = pd.DataFrame(dict(age=age, child=age.lt(13).where(age.notna()),
        sleep=frame.CryoSleep.astype(float), zero=bill.sum(axis=1, min_count=5).eq(0),
        earth=frame.HomePlanet.eq("Earth"), europa=frame.HomePlanet.eq("Europa"),
        side=frame.Cabin.str.endswith("/S").astype(float), deck_g=deck.eq("G")))
    attributes = pd.concat([attributes, np.log1p(bill), bill.gt(0).add_suffix("_paid")], axis=1)
    summary = attributes.astype(float).groupby(group).agg(["mean", "std"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary["size"] = group.value_counts()
    summary["cabins"] = frame.Cabin.groupby(group).nunique()
    return frame.PassengerId, group.to_numpy(), summary


def learn(groups, summary, labels, prior, query_groups):
    residuals = pd.DataFrame(dict(group=groups, excess=labels-prior,
                                  variance=prior*(1-prior))).groupby("group").agg(
                                      excess=("excess","mean"), variance=("variance","mean"), observed=("excess","size"))
    rows = residuals.index[summary.loc[residuals.index, "size"].gt(1)]
    assert not np.intersect1d(rows, query_groups).size
    # First-order excess log odds, weighted by observed information per party.
    denom = residuals.loc[rows, "variance"].clip(lower=.08)
    response = residuals.loc[rows, "excess"] / denom
    model = CatBoostRegressor(iterations=200, depth=3, learning_rate=.035, l2_leaf_reg=10,
                              random_seed=20261118, thread_count=2, verbose=False, allow_writing_files=False)
    model.fit(summary.loc[rows], response, sample_weight=residuals.loc[rows, "observed"] * denom)
    return model


def run():
    inputs, splits, folds, bases, evidence = ref.overlap.load_authenticated()
    dev, y, raw = inputs["dev"], inputs["y"], bases["rawMitra"]
    ids, groups, summary = manifest()
    np.testing.assert_array_equal(ids.iloc[dev], inputs["passengers"])
    g = groups[dev]
    eligible = summary.loc[g, "size"].to_numpy() > 1
    shift = np.zeros(len(dev))
    for fold, (a, b) in enumerate(splits):
        teacher = ref.load_inner(fold, a, b, dev, y, evidence)
        for block in range(5):
            keep = g[a] % 5 != block
            query = b[(g[b] % 5 == block) & eligible[b]]
            model = learn(g[a][keep], summary, y[a][keep], teacher[keep], g[query])
            shift[query] = model.predict(summary.loc[g[query]])
    catalog = pd.read_csv(ROOT / "artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    result = {}
    for name, base in [("raw", raw), ("v96", load_v96(dev, folds))]:
        p = base.astype(float).copy()
        p[eligible] = expit(ref.logit(p[eligible]) + shift[eligible])
        result[name] = dict(metrics=ref.score(p,y), vs_base=ref.pair(p,base,y),
                            hard546=ref.pair(p,base,y,hard), folds=[ref.pair(p[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(REPORT/"result.json",result)
    np.savez_compressed(ROOT/"artifacts/party_outcome_count.npz", dev=dev, shift=shift)
    print(result)


if __name__ == "__main__":
    run()
