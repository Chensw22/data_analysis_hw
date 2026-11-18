"""Compare training on the same reconstructed manifest using a matched CPU classifier."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import reconstructed_mitra as candidate
import relational_error_corrector as ref


def evaluate():
    inputs, splits, _, _, _ = ref.overlap.load_authenticated()
    dev, y = inputs["dev"], inputs["y"]
    altered, _ = candidate.reconstruct()
    original = candidate.counts.load_roster()
    modes = {"original":original, "reconstructed":altered}
    p = {}
    for key, roster in modes.items():
        np.testing.assert_array_equal(roster.passenger_ids.iloc[dev],inputs["passengers"])
        x = roster.transform(roster.plain.iloc[dev])
        p[key] = np.zeros(len(y))
        for fold,(a,b) in enumerate(splits):
            model = CatBoostClassifier(iterations=500,depth=6,learning_rate=.05,
                random_seed=20261112+fold,thread_count=2,cat_features=candidate.counts.base.CATEGORICAL,
                allow_writing_files=False,verbose=False)
            model.fit(x.iloc[a],y[a])
            p[key][b] = model.predict_proba(x.iloc[b])[:,1]
    catalog = pd.read_csv(candidate.ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,catalog.row_id)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    changed = (original.plain.iloc[dev].fillna(-999)!=altered.plain.iloc[dev].fillna(-999)).any(axis=1).to_numpy()
    base, repaired = p["original"], p["reconstructed"]
    result = dict(original=ref.score(base,y),reconstructed=ref.score(repaired,y),
        versus_original=ref.pair(repaired,base,y), hard546=ref.pair(repaired,base,y,hard),
        changed_input=ref.pair(repaired,base,y,changed),same_input=ref.pair(repaired,base,y,~changed),
        fold_net=[ref.pair(repaired[b],base[b],y[b])["net_correct"] for _,b in splits])
    ref.write(candidate.ROOT/"reports/reconstructed_tree_check/result.json",result)
    print(result)


if __name__=="__main__":
    evaluate()
