"""Compare opposite training curricula at exactly matched per-passenger exposure."""
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_refit_dissent_n23 import features
from broad_training_membership_n23 import model


def difficulty(x, y, groups, t):
    forecast = np.full(len(t), np.nan)
    for a, b in GroupKFold(3, shuffle=True, random_state=402).split(t, groups=groups[t]):
        assert not np.intersect1d(groups[t[a]], groups[t[b]]).size
        teacher = model(120).fit(x[t[a]], y[t[a]])
        forecast[b] = teacher.predict_proba(x[t[b]])[:, 1]
    assert np.isfinite(forecast).all()
    error = np.where(y[t], 1 - forecast, forecast)
    rank = np.zeros(len(t))
    for label in (False, True):
        mask = y[t] == label
        rank[mask] = 2*(rankdata(error[mask]) - .5)/mask.sum() - 1
        assert abs(rank[mask].sum()) < 1e-6
    return rank


def sequential(x, y, t, q, schedule):
    fitted = None
    for weight in schedule:
        pool = Pool(x[t], y[t], weight=weight)
        learner = CatBoostClassifier(iterations=110, learning_rate=.065, depth=5, random_seed=2027,
                                     verbose=False, thread_count=2, allow_writing_files=False)
        fitted = learner.fit(pool, init_model=fitted)
    return fitted.predict_proba(x[q])[:, 1]


def run():
    row, y, frame, groups, _ = load()
    x = features(frame)[0]
    out = ART/'nov23_training_order'
    out.mkdir(exist_ok=True)
    history, metrics = {}, {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'], row)
            np.testing.assert_array_equal(z['y'], y)
            fold, base, ref = z['fold'], z['base'], z['ref']
        preds = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[q]))
            rank = difficulty(x, y, groups, t)
            shuffled = rank.copy()
            rng = np.random.default_rng(282+int(f))
            for label in (False, True):
                subset = y[t] == label
                shuffled[subset] = rng.permutation(rank[subset])
            unit = np.ones(len(t))
            for name, order in [('uniform', 0*rank), ('hard_first', rank), ('hard_late', -rank),
                                ('shuffled', shuffled)]:
                weights = [unit + .9*order, unit - .9*order, unit]
                np.testing.assert_allclose(np.sum(weights, axis=0), 3*unit)
                preds.setdefault(name, np.full(len(y), np.nan))[q] = sequential(x, y, t, q, weights)
        odds = lambda p: logit(p.clip(.002, .998))
        preds['both_orders'] = expit((odds(preds['hard_first'])+odds(preds['hard_late']))/2)
        decisions = {'reference': ref}
        for name, p in preds.items():
            assert np.isfinite(p).all()
            decisions[name] = p >= .5
            blend = odds(base) + odds(p) >= 0
            decisions['blend_'+name] = np.where(ref == (base >= .5), blend, ref)
        hit = {k: v == y for k, v in decisions.items()}
        metrics[seed] = {k: dict(vs_reference=paired(h, hit['reference']),
                                 vs_uniform=paired(h, hit['blend_uniform' if k.startswith('blend_') else 'uniform']))
                         for k, h in hit.items()}
        if seed != 'v131':
            for k, h in hit.items():
                history.setdefault(k, []).append(h)
        np.savez_compressed(out/f'{seed}.npz', row=row, y=y, fold=fold, base=base, ref=ref, **preds)
        print(seed, {k: metrics[seed][k]['vs_reference'] for k in ['uniform', 'blend_uniform', 'blend_both_orders']}, flush=True)
    hits = {k: np.stack(v) for k, v in history.items()}
    ref = hits['reference']
    misses = (~ref).sum(0)
    results = {}
    for k, h in hits.items():
        other = hits['blend_uniform' if k.startswith('blend_') else 'uniform']
        delta = (h.astype(float)-other).mean(0)
        results[k] = dict(vs_uniform_mean_pp=float(100*delta.mean()), vs_uniform_ci95_pp=interval(delta, groups),
            broad_error_groups={str(i): dict(people=int((misses == i).sum()),
                correct_all_three=int(h[:, misses == i].all(0).sum()),
                fixed=(h[:, misses == i] & ~ref[:, misses == i]).sum(1).tolist(),
                harmed=(~h[:, misses == i] & ref[:, misses == i]).sum(1).tolist()) for i in range(4)})
    write(ROOT/'reports/nov23_pairwise_adjudication/order.json', dict(splits=metrics, grouped=results))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
