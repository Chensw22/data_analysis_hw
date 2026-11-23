"""Estimate how ten continuous model opinions move together under either outcome."""
import numpy as np
from scipy.special import logit, ndtri
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import predictions
from broad_bill_exception_n23 import load, strata
from broad_booking_policy_n23 import ACTIONS
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_expert_dependence"


def margins(values, support, query, y):
    """Compare empirical class histograms on shared equal-frequency bins per expert."""
    dims = values.shape[1]
    normal = np.empty((2, len(query), dims))
    zfit = [np.empty((np.sum(y[support] == label), dims)) for label in (0, 1)]
    log_density = np.zeros((2, len(query)))
    for j in range(dims):
        v = values[:, j]
        edges = np.unique(np.quantile(v[support], np.linspace(0, 1, 9)[1:-1]))
        bins = np.searchsorted(edges, v, side="right")
        for label in (0, 1):
            train = support[y[support] == label]
            hist = np.bincount(bins[train], minlength=len(edges)+1).astype(float) + 2
            log_density[label] += np.log((hist / hist.sum())[bins[query]])
            # Use empirical midranks, including ties and both tails.
            sorted_values = np.sort(v[train])
            def transform(ids):
                left = np.searchsorted(sorted_values, v[ids], side="left")
                right = np.searchsorted(sorted_values, v[ids], side="right")
                return ndtri((left + right + 1) / (2*len(sorted_values) + 2))
            normal[label, :, j] = transform(query)
            zfit[label][:, j] = transform(train)
    return log_density[1] - log_density[0], normal, zfit


def dependence(z, fit):
    # A Gaussian copula compares the observed joint configuration to
    # independent normal ranks with the same univariate distributions.
    cov = OAS().fit(fit).covariance_
    sd = np.sqrt(cov.diagonal().clip(1.e-8))
    corr = cov / (sd[:, None] * sd[None, :])
    np.fill_diagonal(corr, 1.)
    corr = .995*corr + .005*np.eye(len(sd))
    factor = np.linalg.cholesky(corr)
    white = np.linalg.solve(factor, z.T).T
    return -np.log(factor.diagonal()).sum() - .5*((white**2 - z**2).sum(axis=1))


def evidence(base, log_scores, y, support, query):
    # Condition on the baseline probability before comparing the opinions.
    # Bin boundaries and empirical distributions are estimated on support only.
    bands = np.digitize(base, np.quantile(base[support], [.25, .5, .75]))
    result = np.zeros((len(query), 3))
    rng = np.random.default_rng(812)
    for band in np.unique(bands[query]):
        t = support[bands[support] == band]
        idx = np.flatnonzero(bands[query] == band)
        # Some very confident groups contain only one class on support.
        # Widen by adjacent score bands until both rank distributions can be fit.
        radius = 0
        while np.bincount(y[t].astype(int), minlength=2).min() < 12:
            radius += 1
            assert radius <= 3
            t = support[np.abs(bands[support] - band) <= radius]
        marginal, z, fitted = margins(log_scores, t, query[idx], y)
        real, shuffled = [], []
        for label, train in enumerate(fitted):
            sham = np.column_stack([rng.permutation(col) for col in train.T])
            real.append(dependence(z[label], train))
            shuffled.append(dependence(z[label], sham))
        result[idx] = np.c_[marginal, real[1]-real[0], shuffled[1]-shuffled[0]]
    return result


def combine(base, scores, fold, groups, y):
    odds = logit(base.clip(.001, .999))
    levels = logit(scores.clip(.001, .999))
    delta = levels - odds[:, None]
    predictions_out = {}
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        blocks = [ids[b] for _, b in GroupKFold(3, shuffle=True, random_state=925).split(ids, groups=groups[ids])]
        for i, query in enumerate(blocks):
            for j in (1, 2):
                support, cal = blocks[(i+j)%3], blocks[(i+3-j)%3]
                assert all(not np.intersect1d(groups[u], groups[v]).size
                           for u, v in ((support, query), (cal, query), (support, cal)))
                joined = np.r_[cal, query]
                ev = evidence(base, delta, y, support, joined)
                x = np.c_[odds[joined], delta[joined]/np.sqrt(delta.shape[1])]
                inputs = {"linear": x, "marginals": np.c_[x, ev[:, 0]],
                          "dependent": np.c_[x, ev[:, :2]], "shuffled": np.c_[x, ev[:, [0, 2]]]}
                for k, data in inputs.items():
                    train = data[:len(cal)]
                    mean, sd = train.mean(0), train.std(0).clip(.1)
                    standardized = (data - mean) / sd
                    fit = LogisticRegression(C=.25, max_iter=500).fit(standardized[:len(cal)], y[cal])
                    predictions_out.setdefault(k, np.zeros(len(y)))[query] += .5*fit.predict_proba(
                        standardized[len(cal):])[:, 1]
    return predictions_out


def main():
    row, y, frame, g, _ = load()
    targets, releases, masks = development()
    np.testing.assert_array_equal(y, targets)
    missed = np.logical_and.reduce([p != y for p in releases.values()])
    bill = strata(frame)
    core = missed & (((bill == 2) & y) | ((bill == 0) & ~y))
    scopes = dict(ever=masks["ever"], repeat=masks["repeat"], common=missed, core=core,
                  repeat_outside=masks["repeat"] & ~core)
    store = ART / "nov23_expert_dependence"
    store.mkdir(exist_ok=True)
    history, bases = {}, []
    for seed in SEEDS:
        base, scores, fold = predictions(row, y, seed, ACTIONS)
        out = combine(base, scores, fold, g, y)
        np.savez_compressed(store/f"{seed}.npz", row=row, y=y, base=base, fold=fold, **out)
        ref = (base >= .5) == y
        bases.append(ref)
        for k, p in out.items():
            assert (p>0).all() and (p<1).all()
            h = (p >= .5) == y
            history.setdefault(k, []).append(h)
        h = history["dependent"][-1]
        print(seed, float(h.mean()), int((h & ~ref).sum()), int((~h & ref).sum()), flush=True)
    stats = {k: summarize(np.stack(h), np.stack(bases), scopes) for k, h in history.items()}
    write(REPORT/"results.json", stats)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
