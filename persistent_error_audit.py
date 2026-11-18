"""Check if historical shared errors survive training split changes for two model families."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import relational_error_corrector as ref

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def run():
    inputs, _, fold, models, _ = ref.overlap.load_authenticated()
    dev, y, groups = inputs["dev"], inputs["y"], np.asarray(inputs["groups"])
    cases = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cases.row_id, dev)
    np.testing.assert_array_equal(cases.y, y)
    hard = cases.all_recipes_still_wrong.to_numpy(bool)
    votes = []
    for family, key in [("mitra", "rawMitra"), ("v44", "v44")]:
        votes.append(models[key] >= .5)
        for split, seed in [("random", 20261113), ("random", 20261114),
                            ("group", 20261112), ("group", 20261115), ("group", 20261116)]:
            if split == "random":
                path = ART / "top5_reference_audit" / family / f"{split}_{seed}.npz"
            else:
                root = Path("top5_group_phase/refs") / family
                summary = json.loads((ROOT / "reports" / root / "summary.json").read_text())
                record = next(k for k in summary["results"] if k["seed"] == seed)
                path = ROOT / record["oof_path"]
                assert ref.sha(path) == record["oof_sha256"]
            with np.load(path) as z:
                np.testing.assert_array_equal(z["dev_ids"], dev)
                np.testing.assert_array_equal(z["group_ids"], groups)
                np.testing.assert_array_equal(z["passenger_ids"], inputs["passengers"])
                part = z["fold_id"]
                assert np.array_equal(np.unique(part), np.arange(3))
                if split == "group":
                    for j in range(3):
                        assert not np.intersect1d(groups[part == j], groups[part != j]).size
                prob = z["prob"]
                assert np.isfinite(prob).all() and prob.shape == y.shape
                votes.append(prob >= .5)
    correct_votes = np.sum(np.column_stack(votes) == y[:, None], axis=1)
    with np.load(ART / "composed_local_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], fold)
        base, best = z["base"], z["five_family_majority_with_specialist"]
    with np.load(ART / "confirm_minority_exceptions.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["base"], base)
        preserve = z["second_confirms"]
    with np.load(ART / "sequential_rescue_local.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        other = z["member"][:, :, 2].mean(axis=1) >= .5
    stats = {}
    for name, mask in [("all_splits_both_families_wrong", correct_votes == 0),
                       ("hard546", hard), ("hard_all_splits_wrong", hard & (correct_votes == 0)),
                       ("hard_split_sensitive", hard & (correct_votes > 0)),
                       ("at_least_one_split_correct", correct_votes > 0)]:
        stats[name] = {"rows": int(mask.sum()), "v115_correct": int((mask & (base == y)).sum())}
        for label, pred in [("majority_plus_specialist", best), ("second_confirms", preserve), ("second_alone", other)]:
            stats[name][label] = dict(correct=int((mask & (pred == y)).sum()),
                fixed=int((mask & (base != y) & (pred == y)).sum()),
                harmed=int((mask & (base == y) & (pred != y)).sum()))
    ref.write(ROOT / "reports/persistent_error_audit/result.json", dict(cohorts=stats,
        definition="Two models (raw Mitra, v44) evaluated on the same rows under three random and three party-group CV splits, 12 votes per row. "
        "Only after prediction, for diagnostics. This does not revalidate the new routes on new folds or new labels.",
        histogram_hard_by_number_correct=np.bincount(correct_votes[hard], minlength=13).tolist()))
    print(stats)


if __name__ == "__main__":
    run()
