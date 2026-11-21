"""Protect v128 corrections while requiring a second source for new flips."""
import numpy as np
import pandas as pd

from confirm_sleeper_scope import NAME, confirm
from error_trained_experts import EXPERTS
from exact_fold_error_selector import cached, counts
from export_consensus_routes import BASE, THREE, official_decisions
from kaggle_workflow import validate_submission
from relational_error_corrector import overlap, write
from sleeper_consensus_scope import ROOT, ART
from stable_consensus_selector import persistent_mask


def decide(base, incumbent, margin, matched, triple, committee, donor):
    # Margin's two imputation models also occur in the triple. Matched-party
    # is the corroborating feature family that the margin model doesn't use.
    proposals = {"v133": margin & matched,
                 "v134": triple & (committee | donor)}
    return {key: incumbent ^ ((base == incumbent) & flip) for key, flip in proposals.items()}


def evaluate():
    inp, _, fold, models, _ = overlap.load_authenticated()
    dev, y = inp["dev"], inp["y"].astype(bool)

    def get(path, key):
        return cached(path + ".npz", key, dev, fold)

    base = get("sleeper_consensus_scope", "base")
    incumbent = confirm(base, get("sleeper_consensus_scope", "original"),
                        get("sleeper_manifest_risk", "manifest_berth_party_a"))
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    np.testing.assert_array_equal(catalog.y, y)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp, models, hard)
    expert = get("error_trained_experts", "experts")[:, EXPERTS.index("matched_party")]
    answers = decide(base, incumbent, get("consensus_error_routes", "margin_selector"),
                     (expert >= .5) != base, get("consensus_error_routes", "three_challengers"),
                     get("replicate_error_committee", "original_committee") != base,
                     get("conditional_donor_errors", "matched_donors_risk").mean(1) >= .5)
    return {key: counts(pred, incumbent, y, fold, hard, persistent)
            for key, pred in answers.items()}


def export():
    ids = pd.read_csv(ROOT / "data/raw/test.csv").PassengerId.to_numpy()
    base, _ = official_decisions(BASE, ids)
    incumbent, _ = official_decisions(NAME, ids)
    committee, _ = official_decisions("proposal_v119_error_committee", ids)
    triple, _ = official_decisions(THREE, ids)
    opinions = pd.read_csv(ART / THREE / "expert_opinions.csv")
    np.testing.assert_array_equal(opinions.PassengerId, ids)
    np.testing.assert_array_equal(opinions.flip, triple != base)
    with np.load(ART / "full_challenger_margin/test_components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"], ids)
        np.testing.assert_array_equal(z["margin_flip"], z["reload_margin_flip"])
        margin = z["margin_flip"].astype(bool)
    with np.load(ART / "nov21_donor/components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"], ids)
        donor = z["donor"] >= .5
    predictions = decide(base, incumbent, margin, (opinions.matched_party.to_numpy() >= .5) != base,
                         triple != base, committee != base, donor)
    folder = ART / "nov21_residual_vote"
    folder.mkdir(exist_ok=True)
    exports = {}
    for key, pred in predictions.items():
        path = folder / (key + ".csv")
        pd.DataFrame({"PassengerId": ids, "Transported": pred}).to_csv(path, index=False)
        exports[key] = dict(**validate_submission(path, ROOT / "data/raw/test.csv"),
                            changes_vs_v128=int((pred != incumbent).sum()))
    return exports


if __name__ == "__main__":
    result = dict(dev=evaluate(), test=export(), baseline=NAME,
                  v133="Margin model recommends flip and matched-party model also disagrees with v108",
                  v134="Three experts oppose v108 and either the unanimous error committee or matched donor error risk confirms")
    write(ROOT / "reports/nov21_residual_vote/result.json", result)
    print(result)
