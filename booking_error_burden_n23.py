"""Locate persistent passenger errors across bookings, with a matched permutation reference."""
import numpy as np
import pandas as pd

from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def run():
    y, base_hits, challengers = read_panel()
    rows = pd.read_csv(ART / "nov23_global_error_inventory.csv").row.to_numpy()
    spend_fields = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    raw = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId", *spend_fields])
    booking = raw.PassengerId.str.split("_").str[0]
    size = booking.map(booking.value_counts()).to_numpy()[rows]
    _, group = np.unique(booking.iloc[rows], return_inverse=True)
    known = np.bincount(group)
    bills = raw.iloc[rows][spend_fields]
    spend = np.select([bills.gt(0).any(axis=1), bills.isna().any(axis=1)], [1, 2], default=0)
    scores = []
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as cache:
            np.testing.assert_array_equal(cache["row"], rows)
            np.testing.assert_array_equal(cache["y"], y)
            scores.append(cache["base"])
    # Direction across all splits, median risk at five-percentage-point resolution,
    # spending state, full booking size, and number of evaluated passengers.
    directions = (np.asarray(scores).T >= 0.5) @ np.array([1, 2, 4])
    bands = np.minimum((np.median(scores, axis=0) * 20).astype(int), 19)
    strata = pd.DataFrame(dict(directions=directions, bands=bands, spend=spend, size=size, known=known[group]))
    cells = list(strata.groupby(list(strata.columns)).indices.values())
    persistent = ~base_hits.any(axis=0)
    populations = {"persistent": persistent, "eight_families_also_wrong": persistent & ~challengers.any(axis=(0, 1))}
    report = {}
    for name, mask in populations.items():
        count = np.bincount(group[mask], minlength=len(known))
        same = count[group]
        pairs = int((count * (count - 1) // 2).sum())
        stats = dict(passengers=int(mask.sum()), bookings=int((count > 0).sum()),
                     actual_solo_bookings=int((mask & (size == 1)).sum()),
                     only_evaluated_person_in_booking=int((mask & (known[group] == 1)).sum()),
                     has_evaluated_peers_no_peer_in_error_set=int((mask & (known[group] >= 2) & (same == 1)).sum()),
                     shares_booking_with_another_error=int((mask & (same >= 2)).sum()),
                     everyone_evaluated_in_error_multi=int((mask & (known[group] >= 2) & (same == known[group])).sum()),
                     mixed_outcomes_in_error_set_multi=int((mask & (same >= 2) & (same < known[group])).sum()),
                     pairs_both_in_error_set=pairs)
        assert stats["only_evaluated_person_in_booking"] + stats["has_evaluated_peers_no_peer_in_error_set"] + stats["shares_booking_with_another_error"] == stats["passengers"]
        # Error status travels between passengers who have similar risks and the
        # same total and observed party sizes. The booking membership stays fixed.
        rng = np.random.default_rng(5091)
        movable = [idx for idx in cells if np.any(mask[idx]) and not np.all(mask[idx])]
        shuffled = mask.copy()
        null = []
        for _ in range(1200):
            for idx in movable:
                shuffled[idx] = rng.permutation(mask[idx])
            n = np.bincount(group[shuffled], minlength=len(known))
            null.append(int((n * (n - 1) // 2).sum()))
        stats["matched_pairs_mean"] = float(np.mean(null))
        stats["matched_pairs_reference_95"] = np.quantile(null, [.025, .975]).tolist()
        stats["shuffleable_passengers_in_error_set"] = int(sum(mask[idx].sum() for idx in movable))
        report[name] = stats
    write(ROOT / "reports/nov23_booking_error_burden/result.json",
          dict(evaluated_passengers=len(y), evaluated_bookings=len(known), data=report,
               permutation_repeats=1200, permutation_seed=5091))
    print(report)


if __name__ == "__main__":
    run()
