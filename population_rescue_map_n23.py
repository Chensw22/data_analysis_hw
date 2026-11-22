"""Count complementary errors across whole-booking splits on every development passenger."""
import numpy as np
import pandas as pd

from boarding_sequence_n23 import ART, ROOT, SEEDS
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write


FAMILIES = {
    "sparse_rules": ("sparse_rules", "rules"),
    "tabular_in_context": ("tabicl_context", "global_context"),
    "constructed_features": ("expression_search", "selected"),
    "feature_projections": ("projection_shapes", "projection_shapes"),
    "peer_attributes": ("peer_attr_surprise", "self_peer_surprise"),
    "manifest_likelihood": ("manifest_tree", "manifest_tree"),
    "boarding_sequence": ("boarding_sequence", "sequence"),
    "gradient_kernel": ("group_kernel", "learned_twice"),
}


def read_panel():
    frame = pd.read_csv(ART / "nov23_global_error_inventory.csv")
    rows = frame.row.to_numpy()
    y = frame.target.to_numpy(bool)
    base_hits, challenger_hits = [], []
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as ref:
            fold = ref["fold"]
            verify(ref, rows, y, fold)
            base = ref["base"]
        hits = []
        for experiment, variant in FAMILIES.values():
            with np.load(ART / f"nov23_{experiment}/{seed}.npz") as cache:
                verify(cache, rows, y, fold)
                if "base" in cache:
                    np.testing.assert_array_equal(cache["base"], base)
                scores = cache[variant]
                assert scores.shape == y.shape and np.isfinite(scores).all()
                hits.append((scores >= .5) == y)
        base_hits.append((base >= .5) == y)
        challenger_hits.append(hits)
    return y, np.asarray(base_hits), np.asarray(challenger_hits)


def run():
    y, base, candidates = read_panel()
    wrong = ~base
    misses = wrong.sum(axis=0)
    # A repeatable rescue must cover every split that the baseline missed,
    # while preserving its correct splits for that same passenger.
    stable = candidates.all(axis=0)
    covered = stable.any(axis=0)
    ever = (candidates & wrong[:, None, :]).any(axis=(0, 1))
    counts = {}
    for nmiss in range(len(SEEDS) + 1):
        mask = misses == nmiss
        counts[nmiss] = dict(passengers=int(mask.sum()), transported=int(y[mask].sum()),
                             ever_repaired=int((ever & mask).sum()),
                             same_model_right_all_splits=int((covered & mask).sum()))
    by_family = {}
    for idx, name in enumerate(FAMILIES):
        hits = candidates[:, idx]
        by_family[name] = dict(
            repairs_each_split=(hits & wrong).sum(axis=1).tolist(),
            harms_each_split=(~hits & base).sum(axis=1).tolist(),
            correct_every_split_on_ever_wrong=int((stable[idx] & (misses > 0)).sum()),
            sole_stable_rescuer=int((stable[idx] & (stable.sum(axis=0) == 1) & (misses > 0)).sum()),
            repairs_all_splits_on_always_wrong=int((stable[idx] & (misses == len(SEEDS))).sum()),
            harms_all_splits_on_always_right=int((~hits.any(axis=0) & (misses == 0)).sum()))
    result = dict(rows=len(y), split_seeds=SEEDS, baseline_errors=wrong.sum(axis=1).tolist(),
                  grouped_by_number_of_baseline_errors=counts, families=by_family,
                  interpretation="The same passengers recur across splits. Coverage uses labels after prediction and gives no deployable selection rule.")
    write(ROOT / "reports/nov23_population_rescue/result.json", result)
    print(counts, by_family)


if __name__ == "__main__":
    run()
