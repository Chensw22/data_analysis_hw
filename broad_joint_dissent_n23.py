"""Test whether identities of jointly dissenting classifiers improve held-out decisions."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_error_scope_balance_n23 import archived_votes
from broad_partial_booking_likelihood_n23 import REPORT
from broad_partition_evidence_n23 import SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def combine(vote, base, y, groups, fold):
    sign = np.where(base >= .5, 1., -1.)
    d = (vote != (base >= .5)[:, None]).astype(float)
    n = d.shape[1]
    # Directional opposition to a positive/negative baseline has opposite effects.
    count = d.sum(1)
    first = np.c_[logit(base.clip(.001, .999)), sign, sign[:, None] * d]
    i, j = np.triu_indices(n, 1)
    pairs = d[:, i] * d[:, j] * sign[:, None] / np.sqrt(n)
    # For random model identities with the same total votes, every pair has
    # the same inclusion probability. Collapsing the equal columns with
    # sqrt(number of pairs) preserves the L2 penalty on their shared effect.
    exchangeable = sign * count * (count - 1) / (n*(n-1)) * np.sqrt(len(i)/n)
    setups = {"model_votes": first, "anonymous_pairs": np.c_[first, exchangeable],
              "named_pairs": np.column_stack([first, pairs])}
    out = {k: np.full(len(y), np.nan) for k in setups}
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ids, groups=groups[ids]):
            t, q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            for k, x in setups.items():
                fit = LogisticRegression(C=.5, max_iter=600).fit(x[t], y[t])
                assert fit.n_iter_.max() < 600
                out[k][q] = fit.predict_proba(x[q])[:, 1]
    assert all(np.isfinite(p).all() for p in out.values())
    return out


def main():
    rows, y, bases, archive, *_ = panel()
    local_rows, labels, frame, groups, _ = load()
    np.testing.assert_array_equal(local_rows, rows)
    np.testing.assert_array_equal(labels, y)
    votes = archived_votes(rows, y, bases, archive)
    labels, releases, masks = development()
    np.testing.assert_array_equal(labels, y)
    missed = np.logical_and.reduce([p != y for p in releases.values()])
    bill = strata(frame)
    core = missed & (((bill == 2) & y) | ((bill == 0) & ~y))
    scopes = dict(ever=masks["ever"], repeat=masks["repeat"], release=missed, core=core,
                  repeat_outside=masks["repeat"] & ~core)
    history = {}
    for i, seed in enumerate(SEEDS):
        b, _, fold = predictions(rows, y, seed, {"check": ("class_analogues", "class_linear")})
        np.testing.assert_array_equal(b, bases[i])
        for k, p in combine(votes[i], b, y, groups, fold).items():
            history.setdefault(k, []).append((p >= .5) == y)
    baseline_hit = (bases >= .5) == y
    results = {k: summarize(np.stack(h), baseline_hit, scopes) for k, h in history.items()}
    write(REPORT / "joint_dissent.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
