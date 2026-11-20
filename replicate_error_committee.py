"""Repeat the combined error selector on new calibration partitions and contiguous group blocks."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from exact_fold_error_selector import ART, ROOT, cached, counts
from route_consensus_errors import FAMILIES, cluster_interval
from stable_consensus_selector import make_context, persistent_mask
import relational_error_corrector as ref


def partitions(ids, groups, mode, outer):
    if mode != "boarding_blocks":
        seeds = [119, 220, 321] if mode == "original" else [709, 811, 919]
        for slot, seed in enumerate(seeds):
            for t, q in GroupKFold(3, shuffle=True, random_state=seed + outer).split(ids, groups=groups[ids]):
                yield slot, seed, ids[t], ids[q]
        return
    # Nine consecutive ranges of party IDs. Each query is assessed three times,
    # with its entire range and two neighboring ranges removed from calibration.
    order, index = np.unique(groups[ids].astype(int), return_inverse=True)
    part = np.minimum(8, index * 9 // len(order))
    for start in range(9):
        mask = ((part - start) % 9) < 3
        yield start % 3, 20261120 + start, ids[~mask], ids[mask]


def run():
    inp, _, fold, bases, _ = ref.overlap.load_authenticated()
    dev, groups, y = inp["dev"], np.asarray(inp["groups"]), inp["y"].astype(bool)
    def read(file, key):
        return cached(file + ".npz", key, dev, fold)
    p = read("v108_exact_oof/oof", "v108")
    base, sign = p >= .5, 1 - 2 * (p >= .5).astype(float)
    hist = np.column_stack([bases[k] for k in FAMILIES])
    eligible = ((hist >= .5) == base[:, None]).all(axis=1)
    voters = read("sequential_rescue_local", "member").mean(axis=1)
    critics = read("error_evidence_challengers/oof", "members").mean(axis=1)
    donor = read("conditional_donor_errors", "matched_donors_risk").mean(axis=1)
    proto = read("prototype_error_specialist", "four_centers_risk")
    geometric = read("prototype_error_specialist", "uncalibrated_four") != base
    second = (voters[:, 2] >= .5) != base
    hs, cs = ref.logit(hist) * sign[:, None], ref.logit(critics) * sign[:, None]
    lr_x = np.column_stack([base, np.abs(ref.logit(p)), hs, ref.logit(voters) * sign[:, None], cs,
                            ref.logit(donor), ref.logit(proto), make_context(inp)])
    tree_x = np.column_stack([-np.abs(ref.logit(p)), hs.mean(1), hs.min(1), ref.logit(voters[:, [0,2]]) * sign[:, None],
        cs, donor, read("prototype_error_specialist", "uncalibrated_four_risk"), read("directional_error_selection", "separate_risk"), base])
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, bases, hard)
    summary, cache = {}, {}
    for mode in ["original", "new_party_partitions", "boarding_blocks"]:
        tree_p = np.zeros((len(y), 3))
        logical = np.zeros_like(tree_p, dtype=bool)
        covered = np.zeros_like(tree_p, dtype=int)
        for outer in range(3):
            ids = np.flatnonzero(eligible & (fold == outer))
            for slot, seed, t, q in partitions(ids, groups, mode, outer):
                assert not np.intersect1d(groups[t], groups[q]).size
                gate = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=1000))
                gate.fit(lr_x[t], base[t] != y[t])
                logical[q, slot] = gate.predict_proba(lr_x[q])[:, 1] >= .5
                tree = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=5, min_samples_leaf=32,
                    l2_regularization=10., learning_rate=.06, early_stopping=False, random_state=seed)
                tree.fit(tree_x[t], base[t] != y[t])
                tree_p[q, slot] = tree.predict_proba(tree_x[q])[:, 1]
                covered[q, slot] += 1
        assert (covered[eligible] == 1).all() and not covered[~eligible].any()
        committee = logical.all(axis=1) & geometric & second
        interaction = tree_p.mean(axis=1) >= .5
        if mode == "original":
            np.testing.assert_array_equal(base ^ committee, read("corroborate_exact_errors", "manifest_and_both_specialists"))
            np.testing.assert_array_equal(base ^ interaction, read("monotone_error_confirmation", "unconstrained_mean"))
        for label, change in [("committee", committee), ("tree", interaction), ("either", committee | interaction)]:
            name = mode + "_" + label
            summary[name] = counts(base ^ change, base, y, fold, hard, persistent)
            summary[name]["party_bootstrap_pp"] = cluster_interval(change, base, y, groups)
            cache[name] = base ^ change
        print(mode, summary[mode + "_either"], flush=True)
    np.savez_compressed(ART / "replicate_error_committee.npz", dev=dev, fold=fold, base=base, **cache)
    ref.write(ROOT / "reports/replicate_error_committee/result.json", dict(results=summary,
        scope="Same 6954 development labels and frozen outer-fold models throughout. Only calibration splits change. "
              "The original rule is exploratory. Repeats and contiguous group blocks test sensitivity to "
              "which other labeled parties calibrate the gates, not private-test generalization."))


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
