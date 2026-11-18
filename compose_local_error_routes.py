"""Combine the new corrections and measure their incremental value over simple voting."""
from pathlib import Path

import numpy as np
import pandas as pd

import relational_error_corrector as ref
from route_consensus_errors import FAMILIES, cluster_interval

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def run():
    inputs, splits, fold, models, _ = ref.overlap.load_authenticated()
    dev, y = inputs["dev"], inputs["y"].astype(bool)
    groups = np.asarray(inputs["groups"])
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], fold)
        p = z["v108"]
    with np.load(ART / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["fold"], fold)
        locked = z["combined_by_historical_agreement"]
    anchor = p >= .5
    base = anchor ^ locked

    def candidate(file, key):
        with np.load(ART / (file + ".npz")) as z:
            np.testing.assert_array_equal(z["dev"], dev)
            np.testing.assert_array_equal(z["fold"] if "fold" in z else z["folds"], fold)
            pred = z[key]
            if "base" in z:
                np.testing.assert_array_equal(z["base"], base)
            assert pred.dtype == bool and not np.any(pred[locked] != base[locked])
            return pred != base

    review = candidate("coalition_error_evidence", "full_only")
    consensus = candidate("consensus_dispute_specialist", "agreement_trained_five_agree")
    bills = candidate("sequential_rescue_local", "first_only")
    probabilities = np.column_stack([models[k] for k in FAMILIES])
    agreeing = np.sum((probabilities >= .5) == anchor[:, None], axis=1)
    assert not (consensus & (agreeing != 5)).any()
    divided = (agreeing > 0) & (agreeing < 5)
    kumo_differs = (models["Kumo"] >= .5) != base
    routes = {
        "follow_kumo": ~locked & kumo_differs,
        "mean_kumo_v108": ~locked & (((p + models["Kumo"]) / 2 >= .5) != base),
        "five_family_majority": ~locked & (agreeing < 3),
        "five_family_plus_v108_soft_vote": ~locked & (((probabilities.sum(axis=1) + p) / 6 >= .5) != base),
        "full_only": review,
        "consensus_specialist": consensus,
        "consumption_first": bills,
        "combined": (review & (agreeing < 5)) | consensus,
        "combined_divided_only": (review & divided) | consensus,
        "combined_with_bills_confirming_divided": (review & divided & bills) | consensus,
    }
    for name in ["five_family_majority", "five_family_plus_v108_soft_vote"]:
        # The simple voter deals with disagreements, while the specialist only
        # acts where all five historical models supported the v108 anchor.
        routes[name + "_with_specialist"] = (routes[name] & (agreeing < 5)) | consensus
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    cold = np.zeros(len(y), bool)
    for a, b in splits:
        cold[b] = ~np.isin(groups[b], groups[a])
    subsets = dict(all=np.ones(len(y), bool), hard546=hard, unseen_party=cold,
                   shared_training_party=~cold, five_agree=(agreeing == 5), divided=divided, five_oppose=(agreeing == 0))
    results = {}
    for key, flip in routes.items():
        out = {}
        for cohort, take in subsets.items():
            mask = take & flip
            fixed, harmed = mask & (base != y), mask & (base == y)
            out[cohort] = dict(rows=int(take.sum()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                correct=int(((base ^ flip) == y)[take].sum()),
                fold_net=[int(fixed[b].sum()) - int(harmed[b].sum()) for _, b in splits])
        out["all"]["accuracy"] = float(np.mean((base ^ flip) == y))
        out["all"]["party_bootstrap_pp"] = cluster_interval(flip, base, y, groups)
        results[key] = out
    # The catalog stores correctness instead of decisions. Recover historical
    # votes for diagnostics only, then verify available originals row by row.
    vote_cols = [c for c in catalog if c.startswith("correct_")]
    votes = np.where(catalog[vote_cols].to_numpy(bool), y[:, None], ~y[:, None])
    for k in FAMILIES:
        np.testing.assert_array_equal(votes[:, vote_cols.index("correct_" + k)], models[k] >= .5)
    unanimous_votes = np.all(votes == votes[:, :1], axis=1)
    np.testing.assert_array_equal(hard, unanimous_votes & (votes[:, 0] != y))
    collision = {}
    for label in (False, True):
        cohort = unanimous_votes & (votes[:, 0] == label)
        collision[str(label)] = dict(rows=int(cohort.sum()), historically_wrong=int((cohort & hard).sum()),
            historically_correct=int((cohort & ~hard).sum()))
    ref.write(ROOT / "reports/compose_local_error_routes/result.json", dict(
        unanimous_vote_diagnostic=dict(recipes=len(vote_cols), counts=collision),
        baseline_correct=int((base == y).sum()), base_hard_correct=int((base[hard] == y[hard]).sum()),
        methods=results, interpretation="Adaptive evaluation on the same 6954 development passengers. "
        "Unseen-party subset is conditional on original passenger splits and overrepresents solo passengers."))
    np.savez_compressed(ART / "composed_local_error_routes.npz", dev=dev, fold=fold, base=base,
                        **{k: base ^ flip for k, flip in routes.items()})
    for k, row in results.items():
        print(k, row["all"], "hard", row["hard546"]["fixed"], row["hard546"]["harmed"])


if __name__ == "__main__":
    run()
