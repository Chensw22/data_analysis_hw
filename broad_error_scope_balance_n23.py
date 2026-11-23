"""Account for all baseline errors and the cost of adopting a dissenting vote."""
import numpy as np

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_interaction_error_transfer_n23 import fetch
from broad_new_coverage_union_n23 import collect
from broad_partition_evidence_n23 import ART, EXTRA, SOURCES, SEEDS, write
from broad_partial_booking_likelihood_n23 import REPORT
from released_correction_audit_n23 import development


def archived_probabilities(rows, y, bases, archive):
    panels = []
    for i, seed in enumerate(SEEDS):
        base, extra, fold = predictions(rows, y, seed, {**EXTRA, **dict(list(SOURCES.items())[:4])})
        np.testing.assert_array_equal(base, bases[i])
        recent = collect(rows, y, base, seed, fold)
        additions = []
        for folder, keys in (
            ("context_interaction_audit", ["additive", "age_sleep_deck"]),
            ("billing_response_surface", ["additive", "interaction"]),
            ("adaptive_training_coalitions", ["native_local"]),
        ):
            additions.extend(fetch(folder, keys, rows, y, fold, base, seed))
        with np.load(ART / f"nov23_broad_support_trajectories/{seed}.npz") as z:
            for key, expected in (("row", rows), ("y", y), ("fold", fold)):
                np.testing.assert_array_equal(z[key], expected)
            additions.append(z["bank"][z["subsets"] == 2].mean(0))
        p = np.column_stack([archive[i], extra, *recent.values(), *additions])
        assert p.shape == (len(y), 67) and np.isfinite(p).all()
        panels.append(p)
    return np.stack(panels)


def archived_votes(rows, y, bases, archive):
    return archived_probabilities(rows, y, bases, archive) >= .5


def main():
    rows, y, bases, archive, *_ = panel()
    local_rows, targets, frame, _, _ = load()
    np.testing.assert_array_equal(rows, local_rows)
    np.testing.assert_array_equal(y, targets)
    targets, releases, _ = development()
    np.testing.assert_array_equal(y, targets)
    missed = np.logical_and.reduce([p != y for p in releases.values()])
    bill = strata(frame)
    core = missed & (((bill == 2) & y) | ((bill == 0) & ~y))
    votes = archived_votes(rows, y, bases, archive)
    hit = votes == y[None, :, None]
    base_hit = (bases >= .5) == y
    frequency = (~base_hit).sum(0)
    stable = hit.all(0).any(1)
    never = ~hit.any(axis=(0, 2))
    scopes = {"all": np.ones_like(y), "release": missed, "core": core,
              "either_error_definition": missed | (frequency > 0),
              "baseline_ever_outside_core": (frequency > 0) & ~core,
              "baseline_repeat_outside_core": (frequency == 3) & ~core}
    counts = {}
    for name, mask in scopes.items():
        buckets = []
        for n in range(4):
            m = (frequency == n) & mask
            buckets.append(dict(rows=int(m.sum()), stable_candidate=int((m & stable).sum()),
                                never_correct=int((m & never).sum())))
        counts[name] = buckets
    assert (core & stable).sum() == 263
    thresholds = [1, 17, 34, 51, 67]
    actions = []
    for i, seed in enumerate(SEEDS):
        disagreements = (votes[i] != (bases[i] >= .5)[:, None]).sum(1)
        records = []
        shortfall = int(np.floor(.85*len(y))) + 1 - int(base_hit[i].sum())
        for n in thresholds:
            chosen = disagreements >= n
            fixed = int((chosen & ~base_hit[i]).sum())
            harmed = int((chosen & base_hit[i]).sum())
            records.append(dict(dissent_at_least=n, changed=fixed+harmed, fixed=fixed, harmed=harmed,
                                accuracy=float((base_hit[i] ^ chosen).mean())))
        actions.append(dict(seed=int(seed), net_needed=shortfall, thresholds=records))
    write(REPORT / "full_scope.json", dict(rows=len(y), models=votes.shape[2],
                                           counts_by_number_of_baseline_errors=counts, actions=actions))
    print(counts, actions, flush=True)


if __name__ == "__main__":
    main()
