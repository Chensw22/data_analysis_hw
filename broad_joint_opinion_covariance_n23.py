"""Determine whether correlated expert disagreements identify mistakes across the full roster."""
import numpy as np
from scipy.special import logit
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import strata
from broad_class_cost_path_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import EXTRA, SEEDS, SOURCES, historical, write
from broad_release_miss_coverage_n23 import summarize


def evidence(points, error, test, mode):
    groups = [points[error == flag] for flag in [False, True]]
    if mode == 'shared':
        pooled = np.concatenate([group - group.mean(0) for group in groups])
        models = [LedoitWolf().fit(pooled)] * 2
    else:
        models = [LedoitWolf().fit(group) for group in groups]
    scores = []
    for model, group in zip(models, groups):
        diff = test - group.mean(0)
        precision = model.precision_
        scores.append((np.linalg.slogdet(precision)[1] - ((diff @ precision) * diff).sum(1)) / 2)
    return scores[1] - scores[0]


def calibrated_scores(probs, base, fold, groups, labels):
    baseline = base >= .5
    err = baseline != labels
    signed = np.where(baseline, 1., -1.)
    logits = logit(probs.clip(.005, .995)) * signed[:, None]
    confidence = abs(logit(base.clip(.005, .995)))
    outputs = {name: np.zeros(len(labels)) for name in ['confidence', 'shared', 'different']}
    counts = np.zeros(len(labels), dtype=int)
    for f in np.unique(fold):
        indices = np.flatnonzero(fold == f)
        parts = [indices[q] for _, q in GroupKFold(3, shuffle=True, random_state=925)
                 .split(indices, groups=groups[indices])]
        for i, q in enumerate(parts):
            for j in (1, 2):
                train, calibrate = parts[(i+j) % 3], parts[(i+3-j) % 3]
                for a, b in [(train, calibrate), (train, q), (calibrate, q)]:
                    assert not np.intersect1d(groups[a], groups[b]).size
                # Fit the opinion-space axes on training passengers alone.
                pca = PCA(n_components=10, svd_solver='full', whiten=True).fit(logits[train])
                ref = pca.transform(logits[train])
                test_indices = np.r_[calibrate, q]
                test = pca.transform(logits[test_indices])
                basic = np.c_[confidence[test_indices], baseline[test_indices]]
                for name in outputs:
                    design = basic
                    if name != 'confidence':
                        ratio = evidence(ref, err[train], test, name)
                        scale = max(ratio[:len(calibrate)].std(), .01)
                        design = np.column_stack([basic, ratio / scale])
                    fit = LogisticRegression(C=1., max_iter=300).fit(design[:len(calibrate)], err[calibrate])
                    outputs[name][q] += fit.predict_proba(design[len(calibrate):])[:, 1]
                counts[q] += 1
    np.testing.assert_array_equal(counts, np.full(len(labels), 2))
    return {name: baseline ^ (p / counts >= .5) for name, p in outputs.items()}


def main():
    row, y, base, old, _, names, _ = panel()
    new = {**EXTRA, **dict(list(SOURCES.items())[:4])}
    ids, labels, frame, groups, _ = load()
    np.testing.assert_array_equal(row, ids)
    np.testing.assert_array_equal(y, labels)
    _, _, masks = historical()
    refs = (base >= .5) == y
    hist, banks = {}, []
    for i, seed in enumerate(SEEDS):
        _, extra, fold = predictions(row, y, seed, new)
        prob = np.column_stack([old[i], extra])
        banks.append((prob >= .5) == y[:, None])
        for name, output in calibrated_scores(prob, base[i], fold, groups, y).items():
            hist.setdefault(name, []).append(output == y)
    methods = {name: summarize(np.stack(h), refs, masks) for name, h in hist.items()}
    hit = np.stack(banks)
    stable = hit.all(0).any(1)
    ever = hit.any((0, 2))
    # Only this retrospective table uses a passenger's true outcome or repeated misses.
    state = strata(frame)
    census = []
    for k, title in enumerate(['zero', 'incomplete_no_positive', 'positive']):
        for actual in [False, True]:
            ix = (state == k) & (y == actual)
            m = ix & masks['repeat']
            census.append(dict(state=title, transported=actual, passengers=int(ix.sum()), repeat=int(m.sum()),
                               stable_alternative=int((m & stable).sum()), occasional_only=int((m & ~stable & ever).sum()),
                               never_correct=int((m & ~ever).sum())))
    assert sum(c['repeat'] for c in census) == 1070
    result = dict(experts=len(names) + len(new), census=census, models=methods)
    write(REPORT / 'opinion_covariance.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
