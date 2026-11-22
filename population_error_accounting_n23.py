"""Measure how much of the full error population alternative models can recover."""
import numpy as np
import pandas as pd

from convex_portfolio_bound_n23 import read_scores
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, FAMILIES, ROOT, SEEDS
from relational_error_corrector import write

ADDITIONS = {
    "bayesian_forest": ("bart_long", "bart"),
    "tensor_train": ("tensor_train", "tensor_train"),
    "passenger_relation": ("passenger_relation", "relation_both"),
    "registration_only": ("bill_matched_register", "register"),
    "conditional_router": ("conflict_router", "conflict"),
}
BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def run():
    rows, labels, original = read_scores()
    original_votes = original >= .5
    unanimous_miss = (original_votes != labels[:, None]).all(axis=1)
    panel = {**FAMILIES, "budget_prior": ("simplex_likelihood", "budget_prior"), **ADDITIONS}
    hits, baseline_hits = [], []
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as ref:
            fold, base = ref["fold"], ref["base"]
            verify(ref, rows, labels, fold)
        baseline_hits.append((base >= .5) == labels)
        member_hits = []
        for source, key in panel.values():
            with np.load(ART / f"nov23_{source}/{seed}.npz") as member:
                verify(member, rows, labels, fold)
                prob = member[key]
                assert prob.shape == labels.shape and np.isfinite(prob).all()
                member_hits.append((prob >= .5) == labels)
        hits.append(member_hits)
    bank = np.array(hits)
    stable = bank.all(axis=0)
    prior = len(panel) - len(ADDITIONS)
    missed = (~np.array(baseline_hits)).sum(axis=0)
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c not in {"Name","Transported"})
    booking = raw.PassengerId.str.split("_").str[0]
    party = booking.map(booking.value_counts()).iloc[rows].to_numpy()
    raw = raw.iloc[rows].reset_index(drop=True)
    expenses = raw[BILLS]
    expense_state = np.select(
        [expenses.isna().any(axis=1), expenses.eq(0).all(axis=1)],
        ["incomplete", "all_zero"], default="positive")
    partitions = {
        "bill_state": expense_state,
        "booking_observed_in_train": np.where(party > 1, "multiple", "single"),
        "origin": raw.HomePlanet.fillna("missing").to_numpy(),
        "sleep": raw.CryoSleep.astype("string").fillna("missing").to_numpy(),
    }

    def describe(mask):
        return dict(rows=int(mask.sum()), true_positive_class=int(labels[mask].sum()),
                    true_negative_class=int((~labels[mask]).sum()),
                    prior_stable=int(stable[:prior,mask].any(axis=0).sum()),
                    expanded_stable=int(stable[:,mask].any(axis=0).sum()),
                    expanded_ever=int(bank[:,:,mask].any(axis=(0,1)).sum()))

    cohorts = {f"baseline_wrong_{k}_splits": describe(missed == k) for k in range(4)}
    cohorts["ten_wrong_on_mean"] = describe(unanimous_miss)
    cohorts["baseline_ever_wrong"] = describe(missed > 0)
    strata = {}
    for feature, values in partitions.items():
        strata[feature] = {}
        for value in np.unique(values):
            group = values == value
            strata[feature][str(value)] = {
                "population": int(group.sum()), "ever_wrong": int((group & (missed > 0)).sum()),
                "always_wrong": int((group & (missed == len(SEEDS))).sum()),
                "ten_wrong_on_mean": int((group & unanimous_miss).sum())}
        assert sum(g["ten_wrong_on_mean"] for g in strata[feature].values()) == unanimous_miss.sum()
    methods = {}
    for i, name in enumerate(panel):
        methods[name] = dict(hard_stable=int(stable[i,unanimous_miss].sum()),
                             hard_stable_pos=int(stable[i,unanimous_miss & labels].sum()),
                             hard_stable_neg=int(stable[i,unanimous_miss & ~labels].sum()),
                             spoiled_always_right_all_splits=int((~bank[:,i].any(axis=0) & (missed==0)).sum()),
                             fixes=(bank[:,i] & ~np.array(baseline_hits)).sum(axis=1).tolist(),
                             harms=(~bank[:,i] & np.array(baseline_hits)).sum(axis=1).tolist())
    write(ROOT / "reports/nov23_population_error_accounting/result.json",
          dict(seeds=SEEDS,cohorts=cohorts,partitions=strata,methods=methods))
    print(cohorts, strata)


if __name__ == "__main__":
    run()
