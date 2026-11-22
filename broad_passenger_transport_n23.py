"""Transfer outcomes across whole passenger batches with or without balancing donors."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import SEEDS, evaluate
from broad_release_transfer_n23 import REPORT, matched, write
from released_correction_audit_n23 import development


def manifest_coordinates(frame):
    features = []
    for name in frame:
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col):
            rank = col.rank(pct=True).fillna(.5)
            features.extend([rank.to_numpy(), col.isna().to_numpy() * .5])
        else:
            features.extend(pd.get_dummies(col, dummy_na=True, dtype=float).to_numpy().T / np.sqrt(2))
    return np.column_stack(features) / np.sqrt(len(frame.columns))


def kernel(query, donors):
    # Select ties by the original training order, fixed independently of outcomes.
    k = min(len(donors), 48)
    weights, neighbors = [], []
    for batch in np.array_split(query, max(1, int(np.ceil(len(query) / 250)))):
        dist = cdist(batch, donors, metric="sqeuclidean")
        ix = np.argsort(dist, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(dist, ix, axis=1)
        gap = selected - selected[:, :1]
        scale = np.maximum(gap[:, k // 2], 1e-6)
        w = np.exp(-gap / scale[:, None])
        weights.append(w / w.sum(axis=1, keepdims=True))
        neighbors.append(ix)
    start = np.arange(len(query) + 1) * k
    return csr_matrix((np.concatenate(weights).ravel(), np.concatenate(neighbors).ravel(), start),
                      shape=(len(query), len(donors)))


def transfer(query, donors, values):
    sparse = kernel(query, donors)
    reverse = sparse.T.tocsr()
    m, n = sparse.shape
    def forward(v):
        return .98 * (sparse @ v) + .02 * v.mean(axis=0)
    def backward(v):
        return .98 * (reverse @ v) + .02 * v.sum() / n
    # Total donor weight equals m; force each reference passenger to supply m/n.
    v = np.ones(n)
    for step in range(1500):
        u = 1 / forward(v)
        supply = v * backward(u)
        if np.max(abs(supply * n / m - 1)) < 1e-6:
            break
        v *= (m / n) / supply
    else:
        raise RuntimeError("transport marginals did not converge")
    balanced = u[:, None] * forward(v[:, None] * values)
    np.testing.assert_allclose(balanced.mean(axis=0), values.mean(axis=0), atol=1e-6)
    return {"neighbor": forward(values), "balanced": balanced}


def main():
    row, y, frame, groups, _ = load()
    labels, deployed, masks = development()
    np.testing.assert_array_equal(labels, y)
    missed = np.logical_and.reduce([p != y for p in deployed.values()])
    assert missed.sum() == 941
    masks = {**masks, "all_released_wrong": missed}
    roster = manifest_coordinates(frame)
    early_row, early_y, early = matched()
    np.testing.assert_array_equal(early_row, row)
    np.testing.assert_array_equal(early_y, y)
    early_votes = np.column_stack([p["prob"] >= .5 for p in early])
    results, hits = {}, {}
    for panel in [*map(str, SEEDS), "v131", "v131_early"]:
        name = panel.split("_")[0]
        fold, base, ref, dissent, _ = bank(name, row, y)
        votes = early_votes if panel == "v131_early" else dissent ^ ref[:, None]
        base = base.clip(.001, .999)
        signature = np.c_[logit(base) / 3, votes / np.sqrt(votes.shape[1])]
        designs = {"votes": signature, "manifest": np.c_[signature, roster], "manifest_only": roster}
        scores = {}
        for outer in np.unique(fold):
            ids = np.flatnonzero(fold == outer)
            for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ids, groups=groups[ids]):
                t, q = ids[a], ids[b]
                assert not np.intersect1d(groups[t], groups[q]).size
                values = np.column_stack([y[t], base[t], base[t] * (1 - base[t])])
                for mode, x in designs.items():
                    for rule, avg in transfer(x[q], x[t], values).items():
                        # One damped logistic intercept update relative to the query's own score.
                        residual = (avg[:, 0] - avg[:, 1]) / (.04 + avg[:, 2])
                        options = {"direct": avg[:, 0], "update": expit(logit(base[q]) + residual)}
                        for kind, p in options.items():
                            key = "_".join([kind, mode, rule])
                            scores.setdefault(key, np.full(len(y), np.nan))[q] = p.clip(1e-8, 1-1e-8)
        result, h = evaluate(y, base, ref, scores, fold, name == "v131", masks)
        results[panel] = result
        if not panel.startswith("v131"):
            for key, (hit, _) in h.items():
                hits.setdefault(key, []).append(hit)
        print(panel, [(k, r["fixed"], r["harmed"]) for k, r in result.items()], flush=True)
    stable = {k: {s: int((mask & np.logical_and.reduce(v)).sum()) for s, mask in masks.items()}
              for k, v in hits.items()}
    write(REPORT / "passenger_transport.json", {"plans": results, "stable": stable})


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
