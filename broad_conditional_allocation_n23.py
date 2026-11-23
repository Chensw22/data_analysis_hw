"""Compare learning bag totals with learning which individuals carry a given bag total."""
import numpy as np
import xgboost as xgb
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_multiscale_count_training_n23 import OUTPUT, REPORT, conditional_means, prefix_probabilities, sets
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import SEEDS, write
from broad_refit_dissent_n23 import features
from broad_booking_policy_replay_n23 import interval


def prep_bags(ids, y):
    counts = np.bincount(ids)
    sums = np.bincount(ids, weights=y).astype(int)
    order = np.argsort(ids, kind='stable')
    bundle = []
    for width in np.unique(counts):
        labels = np.flatnonzero(counts == width)
        slots = order[np.isin(ids[order], labels)].reshape(-1, width)
        bundle.append((slots, sums[labels]))
    return bundle


def conditional(p, packed):
    result = np.zeros(len(p))
    for partition in packed:
        for slots, sums in partition:
            result[slots] += conditional_means(p[slots], sums)
    return result/len(packed)


def run_fold(x, y, frame, group, t, q):
    target = y[t].astype(float)
    train = xgb.DMatrix(x[t], label=target, nthread=2)
    query = xgb.DMatrix(x[q], nthread=2)
    params = dict(objective='binary:logistic', tree_method='hist', max_depth=4, eta=.065,
                  reg_lambda=4, min_child_weight=5, nthread=2, seed=891, base_score=.5)
    warm = xgb.train(params, train, num_boost_round=80)
    groups = sets(frame.iloc[t], group[t])
    rng = np.random.default_rng(3275)
    remap = np.arange(len(t))
    for bucket in np.array_split(np.argsort(warm.predict(train), kind='stable'), 10):
        remap[bucket] = rng.permutation(bucket)
    real = [prep_bags(g, target) for g in groups]
    placebo = [prep_bags(g[remap], target) for g in groups]
    result = {}
    for arm, bags in [('ordinary', None), ('count', real), ('allocation', real), ('permuted_allocation', placebo)]:
        def objective(margins, batch):
            p = expit(margins.astype(float)).clip(.0003, .9997)
            gradient = p-target
            if bags is not None:
                mean = conditional(p, bags)
                component = p-mean if arm == 'count' else mean-target
                gradient = .5*(gradient+component)
            # The common positive preconditioner avoids indefinite diagonal curvature of the count loss.
            return gradient, p*(1-p)
        model = xgb.train(params, train, num_boost_round=140, obj=objective, xgb_model=warm)
        result[arm] = model.predict(query)
    return result


def run():
    row, y, frame, groups, _ = load()
    x = features(frame)[0]
    plans, history, wrong = {}, {}, []
    out = OUTPUT/'allocation'
    out.mkdir(parents=True, exist_ok=True)
    for seed in [*map(str, SEEDS), 'v131']:
        fold, ref, op, _ = read_opinions(seed, row, y)
        base = op[:, 0]
        pred = {}
        for f in np.unique(fold):
            ids = np.flatnonzero(fold == f)
            t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[ids]))
            assert not np.intersect1d(groups[t], groups[ids]).size
            for k, p in run_fold(x, y, frame, groups, t, ids).items():
                pred.setdefault(k, np.full(len(y), np.nan))[ids] = p
            print(seed, f, 'fold placed', flush=True)
        assert all(np.isfinite(p).all() for p in pred.values())
        np.savez_compressed(out/f'{seed}.npz', row=row, y=y, fold=fold, ref=ref, base=base, **pred)
        hits = {'reference':ref==y}
        def odds(p):
            return logit(p.clip(.0003, .9997))
        for k, p in pred.items():
            hits[k] = (p>=.5)==y
            p_adjusted = odds(base)+odds(p)-odds(pred['ordinary'])
            hits['transfer_'+k] = np.where(ref==(base>=.5), p_adjusted>=0, ref)==y
        plans[seed] = {}
        for k, h in hits.items():
            comparator = hits['reference' if k.startswith('transfer_') else 'ordinary']
            plans[seed][k] = dict(vs_paired_control=paired(h, comparator),
                                  vs_reference=paired(h, hits['reference']),
                                  fold_net_vs_control=[int((h.astype(int)-comparator)[fold==i].sum())
                                                       for i in np.unique(fold)])
            if seed != 'v131':
                history.setdefault(k, []).append(h)
        if seed != 'v131':
            wrong.append(((op>=.5)!=y[:, None]).all(1))
        print(seed, plans[seed]['transfer_allocation'], flush=True)
    h = {k:np.stack(v) for k, v in history.items()}
    misses = (~h['reference']).sum(0)
    common = np.stack(wrong).all(0)
    assert common.sum() == 332
    groups_summary = {}
    for k, hit in h.items():
        comparison = h['reference' if k.startswith('transfer_') else 'ordinary']
        delta = (hit.astype(float)-comparison).mean(0)
        groups_summary[k] = dict(mean_pp=float(100*delta.mean()), ci95_pp=interval(delta, groups),
                                 correct_common=int((hit.all(0)&common).sum()),
                                 correct_by_baseline_misses=[int((hit.all(0)&(misses==i)).sum()) for i in range(4)])
    write(REPORT/'allocation.json', dict(plans=plans, grouped=groups_summary))


def assess_count_scores():
    row, y, frame, groups, _ = load()
    ledger = {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(OUTPUT/'allocation'/f'{seed}.npz') as data:
            np.testing.assert_array_equal(row, data['row'])
            np.testing.assert_array_equal(y, data['y'])
            fold = data['fold']
            details = {}
            for key in ['ordinary', 'count', 'allocation', 'permuted_allocation']:
                p = data[key].clip(1e-6, 1-1e-6)
                personal = -np.where(y, np.log(p), np.log1p(-p)).mean()
                aggregate = np.zeros(3)
                for f in np.unique(fold):
                    q = np.flatnonzero(fold == f)
                    for scale, bags in enumerate(sets(frame.iloc[q], groups[q])):
                        for slots, counts in prep_bags(bags, y[q]):
                            probs = prefix_probabilities(p[q][slots])
                            mass = probs[np.arange(len(counts)), slots.shape[1], counts]
                            assert (mass > 0).all()
                            aggregate[scale] -= np.log(mass).sum()/len(row)
                details[key] = dict(row_nll=float(personal), count_nll=aggregate.tolist(),
                                    allocation_nll=(personal-aggregate).tolist())
            ledger[seed] = details
    write(REPORT/'heldout_likelihood.json', dict(scales=['booking', 'registration', 'spending'], plans=ledger))


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
        assess_count_scores()
