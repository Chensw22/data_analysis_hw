"""Count new stable corrections beyond the existing 46 classifiers."""
import json

import numpy as np
from archive_error_recovery_n23 import panel, predictions
from broad_noise_transition_n23 import ART, ROOT, SEEDS
from fresh_family_rescue_n23 import NEW


def run():
    rows, y, base, old, _, _, _ = panel()
    extra = dict(**NEW, native_carte=("carte", "p"))
    bundles = [predictions(rows, y, s, extra) for s in SEEDS]
    old = np.concatenate([old, np.stack([m[1] for m in bundles])], axis=2)
    old_ok = (old >= .5) == y[None, :, None]
    hard = ((base >= .5) != y).all(0)
    never = hard & ~old_ok.any(axis=(0, 2))
    unstable = hard & ~old_ok.all(0).any(1)
    hits = {}
    for seed, (_, _, fold) in zip(SEEDS, bundles):
        with np.load(ART / f"nov23_broad_noise_transition/{seed}.npz") as z:
            for key, expect in (("row", rows), ("y", y), ("fold", fold)):
                np.testing.assert_array_equal(z[key], expect)
            for mode in ("raw", "adjusted", "shuffled", "transfer"):
                hits.setdefault(mode, []).append((z[mode] >= .5) == y)
    raw_stable = np.asarray(hits["raw"]).all(0)
    result = {}
    for mode, a in hits.items():
        stable = np.asarray(a).all(0)
        result[mode] = dict(never_hit=int((never & np.any(a, axis=0)).sum()),
                            never_stable=int((never & stable).sum()),
                            newly_stable=int((unstable & stable).sum()),
                            persistent_new_vs_raw=int((hard & ~raw_stable & stable).sum()),
                            persistent_lost_vs_raw=int((hard & raw_stable & ~stable).sum()))
    print(result)
    (ROOT / "reports/nov23_broad_noise_transition/archive.json").write_text(
        json.dumps(dict(prior=old.shape[2], never=int(never.sum()), models=result), indent=2) + "\n")


if __name__ == "__main__":
    run()
