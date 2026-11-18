"""Evaluate a unanimous challenger vote and account for overlapping repairs."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from relational_error_corrector import pair

ROOT = Path(__file__).resolve().parent


def run():
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    y = catalog.y.to_numpy()
    dev = catalog.row_id.to_numpy()
    with np.load(ROOT/"artifacts/hard_weighted_combo_oof.npz") as cat:
        np.testing.assert_array_equal(cat["dev_ids"],dev)
        raw = cat["raw"].copy()
        weighted = cat["weighted_cat"].copy()
    with np.load(ROOT/"artifacts/mechanism_combo_empty_query_fix/oof.npz") as gam:
        np.testing.assert_array_equal(gam["dev_ids"],dev)
        np.testing.assert_array_equal(gam["raw"],raw)
        mechanism = gam["mechanism"].copy()
    with np.load(ROOT/"artifacts/top3_stage/density_oof.npz") as tan:
        np.testing.assert_array_equal(tan["dev"],dev)
        density = tan["density"].copy()
    names = ["weighted_cat","mechanism","density"]
    votes = np.column_stack([weighted,mechanism,density]) >= .5
    correct = votes == y[:,None]
    unanimous = votes.min(axis=1)==votes.max(axis=1)
    decision = np.where(unanimous,votes[:,0],raw>=.5)
    raw_correct = (raw>=.5)==y
    assert hard.sum()==546 and not raw_correct[hard].any()
    result = dict(rows=len(y), accuracy=float(np.mean(decision==y)), paired=pair(decision,raw,y),
                  hard=pair(decision,raw,y,hard), folds=[pair(decision,raw,y,catalog.fold.to_numpy()==fold) for fold in range(3)],
                  hard_at_least_one=int(correct[hard].any(axis=1).sum()),
                  hard_still_all_wrong=int((~correct[hard].any(axis=1)).sum()),
                  hard_exact_vote_patterns={str(bits):int((correct[hard]==bits).all(axis=1).sum())
                                            for bits in itertools.product([0,1],repeat=3)},
                  vote_order=names, both_clause='Base rawMitra unless all three new models vote for the same class.',
                  correct_needed_for_over_85=int(.85*len(y))+1,raw_correct=int(raw_correct.sum()))
    path=ROOT/"reports/top3_stage/committee.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    run()
