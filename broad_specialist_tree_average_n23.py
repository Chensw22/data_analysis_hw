"""Average over all prunings and choices of a forecasting model in each leaf."""
import numpy as np
from scipy.special import logit, logsumexp
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import predictions
from broad_bill_exception_n23 import load, strata
from broad_booking_policy_n23 import ACTIONS
from broad_expert_copula_n23 import REPORT
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from broad_specialist_resolution_n23 import partition
from released_correction_audit_n23 import development


def average(levels, prob, train, query, train_y, global_only=False):
    log_prob = np.log(np.where(train_y[:, None], prob[train], 1-prob[train]))
    log_prior = -np.log(prob.shape[1])
    children_logmass = None
    route = None
    for depth in reversed(range(1 if global_only else len(levels))):
        cells = levels[depth]
        node_count = 2**depth
        total = np.zeros((node_count, prob.shape[1]))
        np.add.at(total, cells[train], log_prob)
        stop_mass = logsumexp(total + log_prior, axis=1)
        weights = np.exp(total + log_prior - stop_mass[:, None])
        stop_prob = np.einsum("ij,ij->i", weights[cells[query]], prob[query])
        if children_logmass is None:
            children_logmass, route = stop_mass, stop_prob
        else:
            split_mass = children_logmass.reshape(node_count, 2).sum(1)
            joint = np.logaddexp(stop_mass, split_mass)
            weight = np.exp(stop_mass - joint)[cells[query]]
            route = weight*stop_prob + (1-weight)*route
            children_logmass = joint - np.log(2)
    return route


def routing_partitions(scores, shuffle=False):
    odds = logit(scores.clip(.001, .999))
    base = rankdata(scores[:, 0]) / (len(scores)+1)
    deltas = rankdata(odds[:, 1:]-odds[:, :1], axis=0) / (len(scores)+1)
    rng = np.random.default_rng(451)
    if shuffle:
        band = np.digitize(base, np.quantile(base, np.arange(1, 20)/20))
        idx = np.arange(len(base))
        for k in np.unique(band):
            pos = np.flatnonzero(band == k)
            idx[pos] = np.random.default_rng(493+k).permutation(pos)
        deltas = deltas[idx]
    trees = [partition(np.c_[base, deltas/np.sqrt(deltas.shape[1])])]
    for _ in range(15):
        # Random projection preserves the whole roster and changes which
        # combinations of model opinions define nearby passengers.
        proj = rng.normal(size=(deltas.shape[1], 4))/np.sqrt(4*deltas.shape[1])
        trees.append(partition(np.c_[base, deltas@proj]))
    return trees


def run_fold(scores, g, y):
    trees = routing_partitions(scores)
    probs = {"soft": scores.clip(.001, .999), "vote": .25 + .5*(scores >= .5)}
    output = {}
    for t, q in GroupKFold(5, shuffle=True, random_state=481).split(scores, groups=g):
        assert not np.intersect1d(g[t], g[q]).size
        for mode, p in probs.items():
            pred = np.stack([average(levels, p, t, q, y[t]) for levels in trees])
            candidates = {"global": average(trees[0], p, t, q, y[t], True),
                          "one_tree": pred[0], "average": pred.mean(0)}
            for key, value in candidates.items():
                output.setdefault(mode+"/"+key, np.zeros(len(y)))[q] = value
    return output


def main():
    row, y, frame, g, _ = load()
    targets, release, masks = development()
    np.testing.assert_array_equal(targets, y)
    common = np.logical_and.reduce([p != y for p in release.values()])
    bill = strata(frame)
    core = common & (((bill == 2) & y) | ((bill == 0) & ~y))
    scopes = dict(ever=masks["ever"], repeat=masks["repeat"], common=common,
                  core=core, repeat_outside=masks["repeat"] & ~core)
    history, refs = {}, []
    dest = ART / "nov23_specialist_tree_average"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        base, p, fold = predictions(row, y, seed, ACTIONS)
        ref = (base >= .5) == y
        refs.append(ref)
        out = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            for k, v in run_fold(np.column_stack([base[q], p[q]]), g[q], y[q]).items():
                out.setdefault(k, np.full(len(y), np.nan))[q] = v
        np.savez_compressed(dest/f"{seed}.npz", row=row, y=y, base=base, fold=fold, **out)
        for k, v in out.items():
            assert np.isfinite(v).all()
            choice = np.where(np.isclose(v, .5, atol=1e-12, rtol=0), base>=.5, v>.5)
            history.setdefault(k, []).append(choice == y)
        print(seed,[(k,int(v[-1].sum())-int(ref.sum())) for k,v in history.items()],flush=True)
    stats = {k: summarize(np.stack(v), np.stack(refs), scopes) for k, v in history.items()}
    write(REPORT/"tree_average.json", stats)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
