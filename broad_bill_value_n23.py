"""Estimate a passenger-specific weight for the no-bill view on nested held-out bookings."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, CATEGORICAL, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_refit_dissent_n23 import features

REPORT = ROOT/'reports/nov23_bill_value'


def fit_views(frame, y, t, q):
    scores = []
    for cols in [list(frame), [k for k in frame if k not in BILLS]]:
        model = CatBoostClassifier(iterations=245, learning_rate=.08, depth=6, thread_count=2,
                                   verbose=False, random_seed=542, allow_writing_files=False)
        model.fit(frame.iloc[t][cols], y[t], cat_features=[k for k in CATEGORICAL if k in cols])
        scores.append(model.predict_proba(frame.iloc[q][cols])[:, 1])
    return np.column_stack(scores)


def surprise(nonbill, bills, t, q):
    # Non-billing attributes determine all comparison passengers, with no outcomes supplied.
    center = np.median(nonbill[t], axis=0)
    width = np.percentile(nonbill[t], 90, axis=0)-np.percentile(nonbill[t], 10, axis=0)
    z = ((nonbill-center)/width.clip(1)).clip(-3, 3)
    out = []
    for batch in np.array_split(q, max(1, int(np.ceil(len(q)/200)))):
        dist = cdist(z[batch], z[t], 'sqeuclidean')
        order = np.argpartition(dist, 63, axis=1)[:, :64]
        neighbors = bills[t[order]]
        signal = bills[batch, None, :]
        have = np.isfinite(neighbors)
        denom = have.sum(1).clip(1)
        percentile = ((neighbors < signal).sum(1)+.5*(neighbors == signal).sum(1))/denom
        positive = (neighbors > 0).sum(1)/denom
        pct = np.where(np.isfinite(signal[:, 0]), percentile-.5, 0.)
        out.extend(np.c_[pct, positive, abs(pct).max(1), np.mean(pct**2, axis=1)])
    return np.array(out)


def shuffled(extra, views, bills, rng):
    bins = ((views[:, 0] >= .5).astype(int)*8 + (views[:, 1] >= .5)*4
            + np.any(bills > 0, axis=1)*2 + np.any(~np.isfinite(bills), axis=1))
    ans = extra.copy()
    for b in np.unique(bins):
        ids = np.flatnonzero(bins == b)
        ans[ids] = extra[rng.permutation(ids)]
    return ans


def mixture(a, b, xtrain, xquery, truth):
    # Fit the weight directly against log loss. A constant weight and feature-dependent
    # weights can therefore be compared with the same objective and same expert forecasts.
    scaler = StandardScaler().fit(xtrain)
    x = np.c_[np.ones(len(a)), scaler.transform(xtrain).clip(-5, 5)]
    q = np.c_[np.ones(len(b)), scaler.transform(xquery).clip(-5, 5)]
    penalty = np.full(x.shape[1], 4.)
    penalty[0] = .5
    prior = np.zeros(x.shape[1])
    prior[0] = -2.
    def objective(coef):
        w = expit(x @ coef)
        gap = a[:, 1]-a[:, 0]
        p = (a[:, 0]+w*gap).clip(1e-7, 1-1e-7)
        loss = -np.where(truth, np.log(p), np.log1p(-p)).sum()
        grad = x.T @ ((p-truth)/(p*(1-p))*gap*w*(1-w))
        reg = coef-prior
        return loss+.5*np.dot(penalty, reg**2), grad+penalty*reg
    opt = minimize(objective, prior, jac=True, method='L-BFGS-B')
    if not opt.success:
        raise RuntimeError(opt.message)
    return expit(q @ opt.x)


def inputs(p, context, bills, extra):
    margins = logit(p.clip(.003, .997))
    magnitude = np.log1p(np.nan_to_num(bills))
    obs = np.c_[context, magnitude, np.isnan(bills), np.log1p(np.nansum(bills, axis=1))]
    return {'constant': np.zeros((len(p), 1)), 'scores': margins,
            'observed': np.c_[margins, obs], 'surprise': np.c_[margins, obs, extra]}


def run():
    row, y, frame, groups, _ = load()
    nonbill = features(frame)[1]
    bills = frame[BILLS].to_numpy(float)
    output = ART/'nov23_bill_value'
    output.mkdir(exist_ok=True)
    plans, histories, wrong, right, rescue, damage, weights = {}, {}, [], [], [], [], {}
    for seed in map(str, SEEDS):
        fold, ref, opinions, _ = read_opinions(seed, row, y)
        with np.load(ART/f'nov23_broad_geographic_transfer/{seed}.npz') as z:
            for k, v in [('row', row), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(z[k], v)
            outer = np.c_[z['full'], z['no_bills']]
        runs = {}
        for f in np.unique(fold):
            t, q = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            assert not np.intersect1d(groups[t], groups[q]).size
            pv = np.full((len(t), 2), np.nan)
            local = np.full((len(t), 2*len(BILLS)+2), np.nan)
            for a, b in GroupKFold(3, shuffle=True, random_state=487).split(t, groups=groups[t]):
                assert not np.intersect1d(groups[t[a]], groups[t[b]]).size
                pv[b] = fit_views(frame, y, t[a], t[b])
                local[b] = surprise(nonbill, bills, t[a], t[b])
            assert np.isfinite(pv).all() and np.isfinite(local).all()
            near = surprise(nonbill, bills, t, q)
            train, test = inputs(pv, nonbill[t], bills[t], local), inputs(outer[q], nonbill[q], bills[q], near)
            rng = np.random.default_rng(409+int(f))
            for design, extra, views, jj in [(train, local, pv, t), (test, near, outer[q], q)]:
                design['shuffled'] = np.c_[design['observed'], shuffled(extra, views, bills[jj], rng)]
            for k in train:
                runs.setdefault(k, np.full(len(y), np.nan))[q] = mixture(pv, outer[q], train[k], test[k], y[t])
        consensus = ((opinions >= .5) == ref[:, None]).all(1)
        wrong.append(consensus & (ref != y))
        right.append(consensus & (ref == y))
        no_bill_hit = (outer[:, 1] >= .5) == y
        rescue.append(no_bill_hit)
        damage.append(~no_bill_hit)
        votes = {'reference': ref, 'full_tree': outer[:, 0] >= .5, 'no_bills': outer[:, 1] >= .5}
        for k, w in runs.items():
            assert np.isfinite(w).all()
            p = outer[:, 0]+w*(outer[:, 1]-outer[:, 0])
            votes[k] = p >= .5
            delta = logit(p.clip(.002, .998))-logit(outer[:, 0].clip(.002, .998))
            votes['transfer_'+k] = logit(opinions[:, 0].clip(.002, .998))+delta >= 0
            weights.setdefault(k, []).append(w)
        h = {k: v == y for k, v in votes.items()}
        for k, hit in h.items():
            histories.setdefault(k, []).append(hit)
        plans[seed] = {k: paired(hit, ref == y) for k, hit in h.items()}
        np.savez_compressed(output/f'{seed}.npz', row=row, y=y, fold=fold, ref=ref, full=outer[:, 0],
                            no_bills=outer[:, 1], base=opinions[:, 0], **runs)
        print(seed, plans[seed]['transfer_surprise'], flush=True)
    common = np.stack(wrong).all(0)
    common_right = np.stack(right).all(0)
    beneficiaries, casualties = common & np.stack(rescue).all(0), common_right & np.stack(damage).all(0)
    assert beneficiaries.sum() == 30 and casualties.sum() == 271
    mirror = beneficiaries | casualties
    # These historical cohorts are never supplied to a learner.
    hits = {k: np.stack(v) for k, v in histories.items()}
    nmiss = (~hits['reference']).sum(0)
    masks = {'common_wrong': common, 'common_right': common_right, 'no_bills_recovers': beneficiaries,
             'no_bills_harms': casualties, **{f'baseline_misses_{i}': nmiss == i for i in range(4)}}
    stats = {}
    for k, h in hits.items():
        comparator = hits['reference'] if k.startswith('transfer_') else hits['full_tree']
        stats[k] = dict(correct_three={key:int(h[:, mask].all(0).sum()) for key, mask in masks.items()},
            versus_comparator=[paired(a, b) for a, b in zip(h, comparator)],
            ci95_pp=interval((h.astype(float)-comparator).mean(0), groups))
    separation = {k: [float(roc_auc_score(beneficiaries[mirror], w[mirror])) for w in ww]
                  for k, ww in weights.items()}
    write(REPORT/'value.json', dict(plans=plans, cohorts={k:int(v.sum()) for k, v in masks.items()},
                                   grouped=stats, descriptive_weight_auc_30_vs_271=separation))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
