"""Use the probability of the passenger's manifest itself to arbitrate new models."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_class_transport_n23 import encode, joint_scores
from broad_expert_copula_n23 import REPORT
from broad_interaction_error_transfer_n23 import fetch
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def design(opinions, mass, t):
    # Typicality is the label-marginal likelihood of the full record, and
    # contrasts how much more mass is assigned by permitting feature dependence.
    z = (mass - mass[t].mean(0)) / mass[t].std(0).clip(.25)
    z = z.clip(-4, 4)
    shift = opinions[:, 1:] - opinions[:, :1]
    main = np.c_[opinions[:, 0], shift]
    interactions = (shift[:, :, None]*z[:, None, :]).reshape(len(z), -1)
    return main, np.c_[z, interactions]


def route(odds, mass, fold, groups, y):
    output = {}
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        risk = odds[ids, 0]
        bins = np.digitize(risk, np.quantile(risk, np.arange(1, 12)/12))
        placebo = mass.copy()
        rng = np.random.default_rng(326)
        for k in np.unique(bins):
            local = ids[bins == k]
            placebo[local] = mass[rng.permutation(local)]
        for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ids, groups=groups[ids]):
            t, q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            main, extra = design(odds, mass, t)
            _, sham = design(odds, placebo, t)
            for key, x in (("opinions", main), ("typicality", np.c_[main, extra]),
                           ("shuffled", np.c_[main, sham])):
                fit = LogisticRegression(C=.2, max_iter=600).fit(x[t], y[t])
                output.setdefault(key, np.full(len(y), np.nan))[q] = fit.predict_proba(x[q])[:, 1]
    return output


def main():
    row, y, frame, groups, _ = load()
    labels, release, masks = development()
    np.testing.assert_array_equal(labels, y)
    common = np.logical_and.reduce([p != y for p in release.values()])
    bill = strata(frame)
    core = common & (((bill==2)&y) | ((bill==0)&~y))
    scopes = dict(ever=masks["ever"], repeat=masks["repeat"], common=common, core=core,
                  repeat_outside=masks["repeat"] & ~core)
    x = encode(frame)
    history, refs = {}, []
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as cache:
            for k,v in (("row",row),("y",y)): np.testing.assert_array_equal(cache[k],v)
            base, fold = cache["base"], cache["fold"]
        old = fetch("class_transport", ["independent", "single_tree", "tree_mixture"], row,y,fold,base,seed)
        raw, shared = fetch("hidden_context", ["raw","shared"], row,y,fold,base,seed)
        mass = np.zeros((len(y), 2))
        for f in np.unique(fold):
            t, q = np.flatnonzero(fold!=f), np.flatnonzero(fold==f)
            assert not np.intersect1d(groups[t], groups[q]).size
            ll = joint_scores(x,y,t,q)
            np.testing.assert_allclose(expit(ll[1]-ll[0]), np.column_stack(old)[q], atol=1.e-12)
            marginal = np.logaddexp(ll[0], ll[1])
            mass[q] = np.c_[marginal[:, 0], marginal[:, 2]-marginal[:, 0]]
        scores = np.column_stack([base, old[0], old[2], raw, shared])
        out = route(logit(scores.clip(.001,.999)), mass, fold, groups, y)
        ref = (base >= .5) == y
        refs.append(ref)
        for k,p in out.items():
            assert np.isfinite(p).all()
            h = (p>=.5) == y
            history.setdefault(k, []).append(h)
    stats = {k: summarize(np.stack(h),np.stack(refs),scopes) for k,h in history.items()}
    write(REPORT/"typicality.json",stats)
    print(stats,flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
