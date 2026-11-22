"""Check the learning-curve predictions on errors shared by earlier classifiers."""
import json

import numpy as np

from archive_error_recovery_n23 import panel, predictions
from broad_support_trajectories_n23 import LEVEL, OUT, REPORT, SEEDS
from fresh_family_rescue_n23 import NEW


def run():
    rows, labels, base, archive, _, names, _ = panel()
    extended = dict(**NEW, native_carte=("carte", "p"))
    models = [predictions(rows, labels, s, extended) for s in SEEDS]
    all_scores = np.concatenate([archive, np.stack([m[1] for m in models])], axis=2)
    old_ok = (all_scores >= .5) == labels[None, :, None]
    repeat = ((base >= .5) != labels).all(0)
    shared = repeat & ~old_ok.any(axis=(0, 2))
    no_stable = repeat & ~old_ok.all(0).any(1)
    result = dict(historical_models=names + list(extended), shared_all_runs=int(shared.sum()),
                  without_stable_fix=int(no_stable.sum()), fractions={})
    banks = []
    for seed, (_, _, folds) in zip(SEEDS, models):
        with np.load(OUT / f"{seed}.npz") as saved:
            for key, expected in [("row", rows), ("y", labels), ("fold", folds), ("subsets", LEVEL)]:
                np.testing.assert_array_equal(saved[key], expected)
            banks.append(saved["bank"])
    for size in range(1, 5):
        average = np.stack([b[LEVEL == size].mean(0) for b in banks])
        hit = (average >= .5) == labels
        result["fractions"][size / 4] = dict(
            shared_correct_at_least_once=int((shared & hit.any(0)).sum()),
            shared_correct_every_time=int((shared & hit.all(0)).sum()),
            newly_stable=int((no_stable & hit.all(0)).sum()))
    (REPORT / "shared_errors.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k:v for k,v in result.items() if k != "historical_models"})


if __name__ == "__main__":
    run()
