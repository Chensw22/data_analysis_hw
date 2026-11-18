"""Compare pairwise ranking with binary log loss on the same development rows."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL

ROOT = Path(__file__).resolve().parent
SEED = 20261112


def opposite_pairs(passengers, labels):
    # Each positive anchors four within-stratum and four global comparisons.
    rng = np.random.default_rng(SEED)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    strata = passengers.HomePlanet.str.cat(passengers.CryoSleep, sep="/").to_numpy()
    pairs = []
    for anchor in positives:
        local = negatives[strata[negatives] == strata[anchor]]
        if len(local):
            pairs.extend((anchor, j) for j in rng.choice(local, 4))
        pairs.extend((anchor, j) for j in rng.choice(negatives, 4))
    return pairs


def run():
    inputs, splits, _, bases, _ = ref.overlap.load_authenticated()
    dev, y = inputs["dev"], inputs["y"]
    x = inputs["features"].iloc[dev].reset_index(drop=True)
    numeric = pd.get_dummies(x, columns=CATEGORICAL, dtype=float)
    config = dict(iterations=500, depth=6, learning_rate=.05, thread_count=4,
                  random_seed=SEED, verbose=False, allow_writing_files=False)
    outputs = {name: np.full(len(dev), np.nan) for name in ["rank", "classifier"]}
    for outer, (train, query) in enumerate(splits):
        fit, calibration = train_test_split(train, test_size=.2, stratify=y[train],
                                           random_state=SEED+outer)
        pairs = opposite_pairs(x.iloc[fit], y[fit])
        ranker = CatBoostRanker(loss_function="PairLogit", **config)
        ranker.fit(Pool(numeric.iloc[fit], y[fit], pairs=pairs, group_id=np.zeros(len(fit), dtype=int)))
        classifier = CatBoostClassifier(loss_function="Logloss", **config)
        classifier.fit(numeric.iloc[fit], y[fit])
        for name, model in [("rank", ranker), ("classifier", classifier)]:
            options = {"prediction_type": "RawFormulaVal"} if name == "classifier" else {}
            cal_score = model.predict(numeric.iloc[calibration], **options).reshape(-1, 1)
            query_score = model.predict(numeric.iloc[query], **options).reshape(-1, 1)
            scale = StandardScaler().fit(cal_score)
            calibrator = LogisticRegression().fit(scale.transform(cal_score), y[calibration])
            outputs[name][query] = calibrator.predict_proba(scale.transform(query_score))[:, 1]
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    raw = bases["rawMitra"]
    rank = outputs["rank"]
    for p in outputs.values():
        assert np.isfinite(p).all()
    result = dict(rows=len(y), metrics={k: ref.score(p, y) for k, p in outputs.items()},
                  rank_vs_classifier=ref.pair(rank, outputs["classifier"], y),
                  rank_vs_raw=ref.pair(rank, raw, y),
                  hard546={k:ref.pair(p,raw,y,hard) for k,p in outputs.items()},
                  fold_net_vs_classifier=[ref.pair(rank[b], outputs["classifier"][b], y[b])["net_correct"]
                                          for _, b in splits])
    ref.write(ROOT/"reports/matched_passenger_ranker/results.json", result)
    print(result)


if __name__ == "__main__":
    run()
