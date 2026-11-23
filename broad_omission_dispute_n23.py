"""Estimate the chance a single-field reversal is justified, on separate bookings."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_observed_mask_shift_n23 import rows

REPORT = ROOT / "reports/nov23_field_robustness"


def predict(fold, groups, labels, baseline, ref, views, observed):
    full = views[:, 0]
    variants = views[:, 1:]
    opposite = ((variants >= .5) != ref[:, None]) & observed
    eligible = ((full >= .5) == ref) & (ref == (baseline >= .5)) & opposite.any(1)
    direction = np.where(ref, 1., -1.)
    confidence = np.where(ref, baseline, 1 - baseline)
    shifts = direction[:, None] * (full[:, None] - variants)
    small = np.c_[confidence, ref, abs(full - .5)]
    wide = np.c_[small, opposite.sum(1), shifts, opposite]
    errors = (ref != labels).astype(int)
    methods = {"prior_only": ref.copy(), "confidence": ref.copy(), "field_response": ref.copy()}
    for f in np.unique(fold):
        ids = np.flatnonzero((fold == f) & eligible)
        splits = [ids[b] for _, b in GroupKFold(5, shuffle=True, random_state=721).split(ids, groups=groups[ids])]
        for j, query in enumerate(splits):
            structure = np.r_[splits[(j+1) % 5], splits[(j+2) % 5]]
            calibration = np.r_[splits[(j+3) % 5], splits[(j+4) % 5]]
            assert not np.intersect1d(groups[structure], groups[calibration]).size
            assert not np.intersect1d(groups[query], groups[np.r_[structure,calibration]]).size
            for name, design in [("prior_only", None), ("confidence", small), ("field_response", wide)]:
                if design is None:
                    cal_leaf = np.zeros(len(calibration), int)
                    query_leaf = np.zeros(len(query), int)
                else:
                    tree = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=24, random_state=637)
                    tree.fit(design[structure], errors[structure])
                    cal_leaf, query_leaf = tree.apply(design[calibration]), tree.apply(design[query])
                for leaf in np.unique(query_leaf):
                    cal = calibration[cal_leaf == leaf]
                    sub = query[query_leaf == leaf]
                    # The prior for each leaf is the observed error rate expected from the incumbent.
                    expected_error = 1 - confidence[sub]
                    error_rate = (errors[cal].sum() + 12 * expected_error) / (len(cal) + 12)
                    flip = sub[error_rate > .5]
                    methods[name][flip] = ~ref[flip]
    return eligible, methods


def main():
    row, y, frame, groups, _ = load()
    _, missing = rows(frame)
    with np.load(ART / "nov23_release_vote_information/panel.npz") as panel:
        np.testing.assert_array_equal(row, panel["row"])
        wrong = (panel["votes"] != y[None, :, None]).all((0, 2))
        right = (panel["votes"] == y[None, :, None]).all((0, 2))
    bank, results = {}, {}
    for seed in [*map(str, SEEDS), "v131"]:
        with np.load(ART / f"nov23_single_field_value/{seed}.npz") as z:
            np.testing.assert_array_equal(row, z["row"])
            np.testing.assert_array_equal(y, z["y"])
            ref = z["ref"]
            eligible, methods = predict(z["fold"], groups, y, z["base"], ref, z["probs"], ~missing)
        results[seed] = {"eligible": int(eligible.sum()), "eligible_in_error": int((eligible & (ref != y)).sum()),
                         "models": {k: paired(v == y, ref == y) for k, v in methods.items()}}
        if seed != "v131":
            for k, v in {"reference": ref, **methods}.items():
                bank.setdefault(k, []).append(v == y)
    stable = {k: {"common_wrong_correct": int((np.all(a, axis=0) & wrong).sum()),
                  "common_right_missed": int((~np.any(a, axis=0) & right).sum())} for k, a in bank.items()}
    write(REPORT / "omission_dispute.json", {"plans": results, "stable": stable})
    print(results, stable)


if __name__ == "__main__":
    main()
