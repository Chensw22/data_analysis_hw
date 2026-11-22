"""Describe the 8-family error inventory using observable broad passenger groups."""
import numpy as np
import pandas as pd

from boarding_sequence_n23 import ART, ROOT, SEEDS
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def run():
    y, hits, challengers = read_panel()
    rows = pd.read_csv(ART / "nov23_global_error_inventory.csv", usecols=["row"]).row
    fields = ["PassengerId", "CryoSleep", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=fields).iloc[rows].reset_index(drop=True)
    spend = raw[fields[2:]]
    spend_state = np.select([spend.gt(0).any(axis=1), spend.isna().any(axis=1)],
                            ["positive", "no_positive_incomplete"], default="all_zero")
    scores = []
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as cache:
            np.testing.assert_array_equal(cache["row"], rows)
            scores.append(cache["base"])
    # A confidence band is available before labels. Use the least confident
    # split so one confident run cannot place someone in the high band.
    confidence = np.maximum(scores, 1 - np.asarray(scores)).min(axis=0)
    bands = np.select([confidence >= .8, confidence >= .65], ["high", "middle"], default="close")
    all_wrong = ~hits.any(axis=0)
    unresolved = all_wrong & ~challengers.any(axis=(0, 1))
    repeatable = all_wrong & challengers.all(axis=0).any(axis=0)
    modes = {"all": np.repeat("all", len(y)), "spend": spend_state,
             "cryo": raw.CryoSleep.astype("string").fillna("missing").to_numpy(),
             "confidence": bands}
    description = {}
    for key, groups in modes.items():
        counts = {}
        for group in sorted(set(groups)):
            mask = groups == group
            remain = mask & unresolved
            counts[str(group)] = dict(rows=int(mask.sum()), base_always_wrong=int((mask & all_wrong).sum()),
                                      repeatably_rescued=int((mask & repeatable).sum()),
                                      never_rescued=int(remain.sum()), never_rescued_rate=float(remain.sum()/mask.sum()),
                                      false_negative=int((remain & y).sum()), false_positive=int((remain & ~y).sum()))
        assert sum(v["rows"] for v in counts.values()) == len(y)
        assert sum(v["never_rescued"] for v in counts.values()) == unresolved.sum()
        description[key] = counts
    write(ROOT / "reports/nov23_population_rescue/profile.json", dict(
        breakdown=description, confidence="Minimum predicted-class probability over three booking-disjoint split assignments. Close <0.65, middle [0.65,0.80), high >=0.80.",
        interpretation="Rows are disjoint within one breakdown. False positives and negatives refer to all 3 baseline predictions on never-rescued passengers."))
    print(description)


if __name__ == "__main__":
    run()
