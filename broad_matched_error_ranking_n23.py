"""Rank likely baseline mistakes among equally confident passengers with unseen bookings."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def risk_models(scores, y, ref, fold, groups):
    aligned = (1-2*ref.astype(float))[:, None]*logit(scores.clip(.002, .998))
    risk = aligned[:, 0]
    confidence = expit(-risk)
    dissent = (scores >= .5) != ref[:, None]
    consensus = ~dissent.any(1)
    context = np.c_[aligned[:, 1:]-risk[:, None], dissent.mean(1), aligned.std(1)]
    err = ref != y
    # Within each class, match the baseline's confidence to five percentage points.
    # Consensus status is also matched so that it cannot distinguish a pair by itself.
    bins = np.floor(confidence*20).astype(int) + 21*ref.astype(int) + 42*consensus.astype(int)
    risk_est = {'confidence': expit(risk)}
    for k in ['absolute', 'matched']:
        risk_est[k] = np.full(len(y), np.nan)
    for f in np.unique(fold):
        ix = np.flatnonzero(fold == f)
        for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ix, groups=groups[ix]):
            t, q = ix[a], ix[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            scaler = StandardScaler().fit(context[t])
            z = scaler.transform(context).clip(-8, 8)
            design = np.c_[risk, ref, consensus, z]
            lr = LogisticRegression(C=.25, max_iter=400)
            risk_est['absolute'][q] = lr.fit(design[t], err[t]).predict_proba(design[q])[:, 1]
            diff = []
            rng = np.random.default_rng(665 + int(f))
            for cell in np.unique(bins[t]):
                subset = t[bins[t] == cell]
                bad, good = subset[err[subset]], subset[~err[subset]]
                if len(bad) and len(good):
                    # Seven independent controls per case, allowed to repeat.
                    controls = rng.choice(good, (len(bad), 7))
                    diff.extend((z[bad, None, :] - z[controls]).reshape(-1, z.shape[1]))
            diff = np.asarray(diff)
            assert len(diff) and np.isfinite(diff).all()
            ranker = LogisticRegression(C=.05, fit_intercept=False, max_iter=400)
            ranker.fit(np.r_[diff, -diff], np.repeat([True, False], len(diff)),
                       sample_weight=np.full(2*len(diff), 1/14))
            incremental = z @ ranker.coef_[0]
            calibration = np.c_[risk, ref, consensus, incremental]
            risk_est['matched'][q] = lr.fit(calibration[t], err[t]).predict_proba(calibration[q])[:, 1]
    assert all(np.isfinite(p).all() for p in risk_est.values())
    return risk_est, consensus


def run():
    row, y, _, groups, _ = load()
    out = ART/'nov23_matched_error_ranking'
    out.mkdir(exist_ok=True)
    results, histories = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        fold, ref, scores, _ = read_opinions(seed, row, y)
        estimates, consensus = risk_models(scores, y, ref, fold, groups)
        # Use the same passengers and tie order across all ranking models; preserve release overrides.
        free = ref == (scores[:, 0] >= .5)
        rng = np.random.default_rng(633)
        tie = rng.random(len(y))
        groups_by_vote = {'all': free, 'unanimous': consensus & free, 'disputed': ~consensus & free}
        metrics = {}
        for subset, mask in groups_by_vote.items():
            ids = np.flatnonzero(mask)
            err = ref[ids] != y[ids]
            cutoff = int(np.floor(.85*len(y)))+1 - int((ref == y).sum())
            metrics[subset] = dict(rows=len(ids), errors=int(err.sum()), net_needed_to_exceed_85=cutoff)
            for key, p in estimates.items():
                ranked = ids[np.lexsort((tie[ids], -p[ids]))]
                methods = {}
                for frac in [.02, .05, .10, .20]:
                    n = round(frac*len(ids))
                    fixed = int((ref[ranked[:n]] != y[ranked[:n]]).sum())
                    methods[str(frac)] = dict(fixed=fixed, harmed=n-fixed, net=2*fixed-n)
                selected = mask & (p >= .5)
                h = (ref ^ selected) == y
                methods['estimated_majority_wrong'] = paired(h, ref == y)
                if seed != 'v131':
                    histories.setdefault(subset+'_'+key, []).append(h)
                metrics[subset][key] = methods
        results[seed] = metrics
        np.savez_compressed(out/f'{seed}.npz', row=row, y=y, fold=fold, ref=ref, unanimous=consensus, **estimates)
        print(seed, metrics['all']['matched']['estimated_majority_wrong'], flush=True)
    refhits = []
    all_unanimous_wrong = []
    for seed in SEEDS:
        with np.load(out/f'{seed}.npz') as z:
            refhits.append(z['ref'] == y)
            all_unanimous_wrong.append(z['unanimous'] & (z['ref'] != y))
    miss = (~np.stack(refhits)).sum(0)
    shared = np.stack(all_unanimous_wrong).all(0)
    stable = {}
    for k, h in histories.items():
        good = np.stack(h).all(0)
        stable[k] = dict(by_number_of_baseline_errors=[int((good & (miss == i)).sum()) for i in range(4)],
                         rescued_unanimous_wrong=int((good & shared).sum()))
    write(ROOT/'reports/nov23_pairwise_adjudication/ranking.json',
          dict(plans=results, stable=stable, unanimous_wrong_three_times=int(shared.sum())))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
