"""Test whether the attributes of nearby cabins explain residual variation."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import relational_error_corrector as baseline

ROOT = Path(__file__).resolve().parent
BILLS = ["FoodCourt", "RoomService", "ShoppingMall", "Spa", "VRDeck"]
OUT = ROOT / "reports/adjacent_cabin_context"


def neighborhood():
    columns = ["PassengerId", "Cabin", "Age", "CryoSleep", *BILLS]
    roster = pd.concat([pd.read_csv(ROOT / "data/raw" / name, usecols=columns)
                        for name in ("train.csv", "test.csv")], ignore_index=True)
    deck, number, side = roster.Cabin.str.split("/", expand=True).values.T
    number = pd.to_numeric(number, errors="coerce")
    groups = roster.PassengerId.str.split("_").str[0].to_numpy()
    age = roster.Age.to_numpy()
    values = np.column_stack([
        roster.CryoSleep.astype(float), np.where(np.isnan(age), np.nan, age < 13),
        np.log1p(roster[BILLS].sum(axis=1, min_count=5))
    ])
    context = np.full((len(roster), 4), np.nan)
    for indices in roster.groupby([deck, side]).indices.values():
        order = indices[np.argsort(number[indices], kind="stable")]
        positions = number[order]
        for i in order:
            left, right = np.searchsorted(positions, [number[i]-4, number[i]+5])
            neighbors = order[left:right]
            neighbors = neighbors[(number[neighbors] != number[i]) & (groups[neighbors] != groups[i])]
            context[i, 0] = np.log1p(len(neighbors))
            if len(neighbors):
                local = values[neighbors]
                count = np.isfinite(local).sum(axis=0)
                context[i, 1:] = np.nansum(local, axis=0) / np.where(count, count, np.nan)
    return context


def design(train, query):
    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    return [np.nan_to_num((part-center)/np.where(scale > 0, scale, 1)) for part in (train, query)]


def run():
    inputs, splits, _, bases, evidence = baseline.overlap.load_authenticated()
    dev, labels = inputs["dev"], inputs["y"]
    raw = bases["rawMitra"]
    features = neighborhood()[dev]
    candidate = np.empty(len(dev))
    intercept_only = np.empty(len(dev))
    folds = []
    for fold, (train, query) in enumerate(splits):
        inner = baseline.load_inner(fold, train, query, dev, labels, evidence)
        x, q = design(features[train], features[query])
        beta, _ = baseline.fit_corrector(inner, labels[train], x)
        bias, _ = baseline.fit_corrector(inner, labels[train], x[:, :0])
        offset = baseline.logit(raw[query])
        candidate[query] = expit(offset + beta[0] + q @ beta[1:])
        intercept_only[query] = expit(offset + bias[0])
        folds.append(baseline.pair(candidate[query], raw[query], labels[query]))
    catalog = pd.read_csv(ROOT / "artifacts/error_rescue_20261117/catalog_rows.csv", dtype={"PassengerId": str})
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, labels)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    assert hard.sum() == 546 and np.isfinite(candidate).all()
    result = dict(rows=len(dev), folds=folds, radius=4, penalty=baseline.PENALTY,
                  covered=int(np.sum(features[:, 0] > 0)),
                  metrics={name: baseline.score(p, labels) for name, p in
                           [("candidate", candidate), ("raw", raw), ("intercept", intercept_only)]},
                  versus_raw=baseline.pair(candidate, raw, labels),
                  versus_intercept=baseline.pair(candidate, intercept_only, labels),
                  hard546=baseline.pair(candidate, raw, labels, hard))
    baseline.write(OUT / "results.json", result)
    print(result)


if __name__ == "__main__":
    run()
