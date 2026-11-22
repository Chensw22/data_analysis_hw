"""Account for all baseline errors and check where geographic updates change decisions."""
import numpy as np
import pandas as pd

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from broad_booking_policy_n23 import EXTRA
from broad_booking_policy_replay_n23 import interval
from broad_error_placement_control_n23 import SOURCES
from relational_error_corrector import write


def run():
    row, y, base, old, _, names, _ = panel()
    standard = {**EXTRA, **dict(list(SOURCES.items())[:4])}
    recent = []
    spatial = []
    for seed in SEEDS:
        _, probabilities, fold = predictions(row, y, seed, standard)
        recent.append(probabilities)
        columns = []
        for path, key in [(f"broad_booking_kriging/{seed}", "baseline_random_signed"),
                          (f"broad_geographic_transfer/all_training_{seed}", "full"),
                          (f"broad_geographic_transfer/dense_{seed}", "dense")]:
            with np.load(ART / f"nov23_{path}.npz") as data:
                for field, value in (("row", row), ("y", y), ("fold", fold)):
                    np.testing.assert_array_equal(data[field], value)
                assert np.isfinite(data[key]).all()
                columns.append(data[key] >= .5)
        spatial.append(np.column_stack(columns))

    archive = np.concatenate([old, np.stack(recent)], axis=2) >= .5
    spatial = np.stack(spatial)
    groups = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId"]).PassengerId.iloc[row].str.split("_").str[0]
    ref = base >= .5
    correct = ref == y
    repeated = ~correct.any(0)
    ever = ~correct.all(0)
    archive_hits = archive == y[None, :, None]
    stable = archive_hits.all(0)
    comparable = (archive_hits.mean(1) >= correct.mean(1)[:, None] - .01).all(0)
    coverage = {}
    for key, cohort in (("ever", ever), ("repeat", repeated)):
        same_level = stable[:, comparable].any(1)
        any_stable = stable.any(1)
        occasional = archive_hits.any((0, 2))
        bands = [same_level, any_stable & ~same_level, occasional & ~any_stable, ~occasional]
        assert np.all(np.sum(bands, axis=0) == 1)
        coverage[key] = [dict(total=int((band & cohort).sum()), transported=int((band & cohort & y).sum()))
                         for band in bands]
    never46 = repeated & ~archive_hits[:, :, :old.shape[2]+len(EXTRA)].any((0, 2))
    assert never46.sum() == 212

    def scores(votes):
        hit = votes == y[None, :, None]
        all_hit = hit.all(0)
        return np.vstack([(hit.astype(float) - correct[:, :, None]).sum(1).mean(0),
                          all_hit[repeated].sum(0), all_hit[never46].sum(0)])

    ref_pattern = np.sum(ref * np.array([1, 2, 4])[:, None], axis=0)
    confidence = np.maximum(base, 1-base).mean(0)
    risk = np.digitize(confidence, np.quantile(confidence, [.2, .4, .6, .8]))
    dissent = (archive != ref[:, :, None]).mean((0, 2))
    blocks = ref_pattern * 20 + risk * 4 + np.digitize(dissent, [1e-9, .05, .2])
    cells = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(841)
    draws = []
    for _ in range(800):
        indices = np.arange(len(y))
        for cell in cells:
            indices[cell] = rng.permutation(cell)
        draws.append(scores(spatial[:, indices]))
    results = {}
    observed = scores(spatial)
    for j, key in enumerate(["fold_calibration", "all_training", "dense"]):
        hit = spatial[:, :, j] == y
        gains = hit.astype(float) - correct
        result = dict(accuracy=hit.mean(1).tolist(), fixed=(gains > 0).sum(1).tolist(),
                      harmed=(gains < 0).sum(1).tolist(), paired_ci_pp=interval(gains.mean(0), groups))
        for i, metric in enumerate(["mean_net_people", "stable_repeat", "stable_never46"]):
            null = np.asarray(draws)[:, i, j]
            result[metric] = dict(observed=float(observed[i, j]), shuffled_mean=float(null.mean()),
                                  shuffled_95=np.quantile(null, [.025, .975]).tolist())
        results[key] = result
    write(ROOT / "reports/nov23_broad_geographic_transfer/placement.json",
          dict(archive_size=archive.shape[-1], comparable=int(comparable.sum()), methods=names+list(standard),
               bands=["comparable_stable", "only_other_stable", "only_occasional", "never"],
               coverage=coverage, spatial=results, permutations=800, cells=len(cells)))


if __name__ == "__main__":
    run()
