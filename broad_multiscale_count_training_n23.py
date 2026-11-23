"""Train row-level forecasts from individual labels and exact group-count likelihoods."""
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_refit_dissent_n23 import features

OUTPUT = ART/'nov23_multiscale_count_training'
REPORT = ROOT/'reports/nov23_multiscale_count_training'


def prefix_probabilities(p):
    n, width = p.shape
    mass = np.zeros((n, width+1, width+1))
    mass[:, 0, 0] = 1
    for k in range(width):
        mass[:, k+1] = mass[:, k]*(1-p[:, k, None])
        mass[:, k+1, 1:] += mass[:, k, :-1]*p[:, k, None]
    return mass


def conditional_means(p, count):
    """Condition independent Bernoulli variables on the observed total of each bag."""
    front = prefix_probabilities(p)
    back = prefix_probabilities(p[:, ::-1])
    rows = np.arange(len(p))
    width = p.shape[1]
    denom = front[rows, width, count]
    assert (denom > 0).all()
    rest = count[:, None]-1-np.arange(width+1)
    valid = rest >= 0
    idx = rest.clip(0, width)
    answer = np.empty_like(p)
    for k in range(width):
        suffix = np.take_along_axis(back[:, width-1-k], idx, axis=1)
        answer[:, k] = p[:, k]*(front[:, k]*suffix*valid).sum(1)/denom
    np.testing.assert_allclose(answer.sum(1), count, rtol=1e-8, atol=1e-8)
    return np.clip(answer, 0, 1)


def verify_conditional():
    p = np.array([[.17, .84, .51, .06]])
    configs = np.array(list(product([0, 1], repeat=p.shape[1])))
    for k in range(p.shape[1]+1):
        candidates = configs[configs.sum(1) == k]
        mass = np.prod(np.where(candidates, p, 1-p), axis=1)
        expected = np.average(candidates, weights=mass, axis=0)
        np.testing.assert_allclose(conditional_means(p, np.array([k]))[0], expected)


def bag_ids(order, size):
    ids = np.empty(len(order), int)
    ids[order] = np.arange(len(order))//size
    return ids


def sets(frame, group):
    i = np.arange(len(frame))
    region = (frame.Deck.astype(str)+'/'+frame.Side.astype(str)).to_numpy(str)
    b = frame[BILLS].to_numpy(float)
    state = np.where(np.isnan(b), 2, b > 0)
    pattern = state @ (3**np.arange(len(BILLS)))
    # Each partition covers even singleton and missing-record passengers.
    return [pd.factorize(group)[0],
            bag_ids(np.lexsort((i, group, region)), 12),
            bag_ids(np.lexsort((i, np.nansum(b, axis=1), pattern)), 24)]


def posterior_targets(y, p, memberships):
    parts = []
    for bags in memberships:
        n = np.bincount(bags)
        counts = np.bincount(bags, weights=y).astype(int)
        order = np.argsort(bags, kind='stable')
        ans = np.zeros(len(y))
        for width in np.unique(n):
            ids = np.flatnonzero(n == width)
            slots = order[np.isin(bags[order], ids)].reshape(-1, width)
            ans[slots] = conditional_means(p[slots], counts[ids])
        parts.append(ans)
    return np.mean(parts, axis=0)


def step(x, target, iters, prev=None):
    model = CatBoostClassifier(iterations=iters, learning_rate=.065, depth=5, loss_function='CrossEntropy',
                               allow_writing_files=False, random_seed=891, verbose=False, thread_count=2)
    return model.fit(x, target, init_model=prev)


def run_fold(x, y, frame, group, t, q):
    ytrain = y[t].astype(float)
    starter = step(x[t], ytrain, 120)
    prior = starter.predict_proba(x[t])[:, 1]
    parts = sets(frame.iloc[t], group[t])
    # Preserve the distribution of prior risk within each group to decile accuracy.
    rng = np.random.default_rng(9163)
    remap = np.arange(len(t))
    for bucket in np.array_split(np.argsort(prior, kind='stable'), 10):
        remap[bucket] = rng.permutation(bucket)
    placebo = [v[remap] for v in parts]
    for a, b in zip(parts, placebo):
        np.testing.assert_array_equal(np.bincount(a), np.bincount(b))
    result = {}
    for key, bags in [('ordinary', None), ('booking', parts[:1]), ('multiscale', parts), ('placebo', placebo)]:
        prev = starter
        for _ in range(3):
            if bags is None:
                target = ytrain
            else:
                prior = prev.predict_proba(x[t])[:, 1].clip(.001, .999)
                expected = posterior_targets(ytrain, prior, bags)
                # Half the weight stays on individual labels for every training passenger.
                target = .5*(ytrain+expected)
            prev = step(x[t], target, 85, prev)
        result[key] = prev.predict_proba(x[q])[:, 1]
    return result


def run():
    verify_conditional()
    row, y, frame, group, _ = load()
    x = features(frame)[0]
    OUTPUT.mkdir(exist_ok=True)
    ledger, histories, anchor_error = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, ref, op, _ = read_opinions(seed, row, y)
        base = op[:, 0]
        scores = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            t = np.flatnonzero((fold != f) & ~np.isin(group, group[q]))
            assert not np.intersect1d(group[t], group[q]).size
            for key, p in run_fold(x, y, frame, group, t, q).items():
                scores.setdefault(key, np.full(len(y), np.nan))[q] = p
        assert all(np.isfinite(p).all() for p in scores.values())
        np.savez_compressed(OUTPUT/f'{seed}.npz', row=row, y=y, fold=fold, base=base, ref=ref, **scores)
        hits = {'reference': ref == y}
        for key, p in scores.items():
            hits[key] = (p >= .5) == y
            offset = logit(p.clip(.001, .999))-logit(scores['ordinary'].clip(.001, .999))
            vote = np.where(ref == (base >= .5), logit(base.clip(.001, .999))+offset >= 0, ref)
            hits['transfer_'+key] = vote == y
        ledger[seed] = {k: dict(vs_reference=paired(h, hits['reference']), vs_ordinary=paired(h, hits['ordinary']),
                                net_in_folds=[int((h.astype(int)-hits['reference'])[fold == f].sum())
                                              for f in np.unique(fold)]) for k, h in hits.items()}
        if seed != 'v131':
            for k, h in hits.items():
                histories.setdefault(k, []).append(h)
            anchor_error.append(((op >= .5) != y[:, None]).all(1))
        print(seed, ledger[seed]['multiscale'], ledger[seed]['transfer_multiscale'], flush=True)
    h = {k: np.stack(v) for k, v in histories.items()}
    nmiss = (~h['reference']).sum(0)
    common = np.stack(anchor_error).all(0)
    assert common.sum() == 332
    stats = {}
    for k, hit in h.items():
        base = h['reference' if k.startswith('transfer_') else 'ordinary']
        gain = (hit.astype(float)-base).mean(0)
        stats[k] = dict(mean_gain_pp=float(100*gain.mean()), interval95_pp=interval(gain, group),
                        stable_shared=int((hit.all(0) & common).sum()),
                        stable_by_misses=[int((hit.all(0) & (nmiss == i)).sum()) for i in range(4)])
    write(REPORT/'result.json', dict(plans=ledger, grouped=stats))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
