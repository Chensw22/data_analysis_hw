"""How fine must a label-free partition be to reach 85% by choosing one model per cell?"""
import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import predictions
from broad_bill_exception_n23 import load, strata
from broad_booking_policy_n23 import ACTIONS
from broad_expert_copula_n23 import REPORT
from broad_partition_evidence_n23 import ROOT, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def partition(x):
    # Use all unlabelled query features transductively. At each node split the
    # longest principal axis at its median, breaking exact ties at random.
    levels = [np.zeros(len(x), dtype=int)]
    rng = np.random.default_rng(873)
    for depth in range(8):
        parent = levels[-1]
        child = parent * 2
        for node in np.unique(parent):
            ids = rng.permutation(np.flatnonzero(parent == node))
            centered = x[ids] - x[ids].mean(0)
            _, axes = np.linalg.eigh(centered.T @ centered)
            order = np.argsort(centered @ axes[:, -1], kind="stable")
            child[ids[order[len(ids)//2:]]] += 1
        levels.append(child)
    return levels


def totals(keys, outcomes):
    _, ids = np.unique(keys, return_inverse=True, axis=0)
    sums = np.zeros((ids.max()+1, outcomes.shape[1]), dtype=int)
    np.add.at(sums, ids, outcomes)
    return sums[ids], np.bincount(ids)[ids]


def apply_policy(levels, vote, y, group):
    reward = (vote == y[:, None]).astype(int)
    # The root shrinkage prior is a tie preferring the incumbent. Each next
    # level borrows 32 observations' worth of its parent estimate.
    parent = np.zeros_like(reward, dtype=float)
    results, counts = {}, {}
    for depth, cell in enumerate(levels):
        sums, n = totals(cell, reward)
        own, own_n = totals(np.c_[cell, group], reward)
        available = n - own_n
        held_sum = sums - own
        assert (held_sum >= 0).all() and (held_sum <= available[:, None]).all()
        posterior = (held_sum + 32*parent) / (available[:, None] + 32)
        parent = posterior
        if depth % 2:
            continue
        counts[depth] = available
        for k, score in (("hindsight", sums), ("held_booking", held_sum), ("pooled_parent", posterior)):
            choice = np.argmax(score, axis=1)
            results[k, depth] = vote[np.arange(len(y)), choice]
    return results, counts


def run():
    row, y, frame, groups, _ = load()
    labels, releases, masks = development()
    np.testing.assert_array_equal(labels, y)
    common = np.logical_and.reduce([p != y for p in releases.values()])
    bill = strata(frame)
    core = common & (((bill == 2) & y) | ((bill == 0) & ~y))
    scopes = dict(ever=masks["ever"], repeat=masks["repeat"], release=common,
                  core=core, repeat_outside=masks["repeat"] & ~core)
    history, detail, refs = {}, {}, []
    for seed in SEEDS:
        base, p, fold = predictions(row, y, seed, ACTIONS)
        ballot = np.column_stack([base, p]) >= .5
        base_odd = logit(base.clip(.001, .999))
        delta = logit(p.clip(.001, .999)) - base_odd[:, None]
        all_maps, count = {}, {}
        for f in np.unique(fold):
            ids = np.flatnonzero(fold == f)
            assert not np.intersect1d(groups[ids], groups[fold!=f]).size
            risk = base[ids]
            # An equal conditional distribution of cloned opinions controls
            # for the number of cells and for the strong baseline score.
            ranks = rankdata(delta[ids], axis=0) / (len(ids)+1)
            bands = np.digitize(risk, np.quantile(risk, np.arange(1, 20)/20))
            perm = np.arange(len(ids))
            rng = np.random.default_rng(493)
            for band in np.unique(bands):
                local = np.flatnonzero(bands == band)
                perm[local] = rng.permutation(local)
            for geom, opinion in (("real", ranks), ("shuffled", ranks[perm])):
                x = np.c_[rankdata(risk)/(len(ids)+1), opinion/np.sqrt(p.shape[1])]
                levels = partition(x)
                choices, coverage = apply_policy(levels, ballot[ids], y[ids], groups[ids])
                for (policy, depth), v in choices.items():
                    key = f"{geom}/{policy}/{depth}"
                    all_maps.setdefault(key, np.zeros(len(y), bool))[ids] = v
                for depth, size in coverage.items():
                    count.setdefault(f"{geom}/{depth}", []).extend(size)
        ref = ballot[:, 0] == y
        refs.append(ref)
        for k, v in all_maps.items():
            history.setdefault(k, []).append(v == y)
        detail[str(seed)] = {k: np.quantile(v, [.1, .5, .9]).tolist() for k, v in count.items()}
    result = {k: summarize(np.stack(h), np.stack(refs), scopes) for k, h in history.items()}
    write(REPORT/"resolution.json", dict(policies=result, other_passengers_by_cell=detail,
                                        candidates=["incumbent", *ACTIONS]))
    for d in (0, 2, 4, 6, 8):
        print(d, {k: np.round(v["accuracy"], 4).tolist() for k, v in result.items()
                  if k.startswith("real/") and k.endswith("/"+str(d))}, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
