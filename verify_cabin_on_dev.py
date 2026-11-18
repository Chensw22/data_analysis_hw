"""Estimate the effect of the cabin repair on unseen rows using a CPU classifier."""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import peer_cabin_impute as cabin
import relational_error_corrector as ref
from exaone_validation_data import CATEGORICAL

ROOT = Path(__file__).resolve().parent


def run():
    _, updates, _ = cabin.infer(test_only=False)
    inputs,splits,_,_,_ = ref.overlap.load_authenticated()
    dev,y = inputs["dev"], inputs["y"]
    original = inputs["features"].iloc[dev].copy()
    repaired = original.copy()
    affected = original.index.intersection(updates.index)
    repaired.loc[affected,updates.columns] = updates.loc[affected]
    p,q = np.empty(len(y)),np.empty(len(y))
    for train, query in splits:
        model = CatBoostClassifier(iterations=500,depth=6,learning_rate=.05,verbose=False,
                                   thread_count=4,random_seed=20261112,allow_writing_files=False)
        model.fit(original.iloc[train],y[train],cat_features=CATEGORICAL)
        p[query] = model.predict_proba(original.iloc[query])[:,1]
        q[query] = model.predict_proba(repaired.iloc[query])[:,1]
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev)
    np.testing.assert_array_equal(catalog.y,y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    eligible = original.index.isin(affected)
    np.testing.assert_array_equal(p[~eligible],q[~eligible])
    result = dict(classifier="CatBoost compact13; same fitted model before/after query repair",
                  eligible=int(eligible.sum()), all_rows=ref.pair(q,p,y),
                  eligible_hard=ref.pair(q,p,y,hard & eligible),
                  original=ref.score(p,y),patched=ref.score(q,y))
    ref.write(ROOT/"reports/cabin_mask_diagnostic/results.json", result)
    print(result)


if __name__ == "__main__":
    run()
