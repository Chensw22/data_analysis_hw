"""Learn which side wins model disagreements, with equal training weight per passenger."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def read_opinions(seed, row, target):
    with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as z:
        np.testing.assert_array_equal(z['row'], row)
        np.testing.assert_array_equal(z['y'], target)
        fold, base, ref = z['fold'], z['base'], z['ref']
    sources = {
        'crossfamily_unlabelled_distillation': ['teacher', 'native', 'target_mix'],
        'unlabelled_neural_smoothing': ['supervised'],
        'training_error_factorial': ['prob'],
        'broad_service_transport': ['learned'],
    }
    opinions = {'baseline': base}
    for folder, keys in sources.items():
        with np.load(ART / f'nov23_{folder}/{seed}.npz') as z:
            for k, v in [('row', row), ('y', target), ('fold', fold), ('base', base)]:
                np.testing.assert_array_equal(z[k], v)
            for key in keys:
                if key == 'prob':
                    opinions['specialist_FN'] = z[key][:, 2]
                    opinions['specialist_FP'] = z[key][:, 4]
                else:
                    opinions[key] = z[key]
    scores = np.column_stack(list(opinions.values()))
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    return fold, ref, scores, list(opinions)


def comparisons(scores, add_constants):
    votes = scores >= .5
    share = votes.mean(1)
    anchor = logit(scores[:, 0].clip(.003, .997))
    if add_constants:
        scores = np.column_stack([scores, np.full(len(scores), .55), np.full(len(scores), .45)])
        votes = scores >= .5
    # Each pair points from the model voting yes to the one voting no.
    row, yes, no = np.nonzero(votes[:, :, None] & ~votes[:, None, :])
    counts = np.bincount(row, minlength=len(scores))
    margin = logit(scores.clip(.003, .997))
    pos, neg = margin[row, yes], margin[row, no]
    base = np.column_stack([anchor[row], share[row] - .5, pos, neg, pos + neg])
    eye = np.eye(scores.shape[1])
    identities = np.c_[eye[yes], eye[no], eye[yes]*pos[:, None], eye[no]*neg[:, None]]
    return row, counts, base, np.c_[base, identities]


def forecast(scores, fold, groups, y, ref):
    output = {'reference': ref.copy(), 'ordinary_stack': ref.copy()}
    margin = logit(scores.clip(.003, .997))
    pairs = {k: comparisons(scores, k == 'open') for k in ['dispute', 'open']}
    for k in pairs:
        for model in ['anonymous', 'named']:
            output[k + '_' + model] = ref.copy()
    for f in np.unique(fold):
        ids = np.flatnonzero(fold == f)
        # Each level-one model excluded these passengers. Fit readouts on separate bookings inside this fold.
        for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ids, groups=groups[ids]):
            t, q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            lr = LogisticRegression(C=.25, max_iter=500)
            output['ordinary_stack'][q] = lr.fit(margin[t], y[t]).predict(margin[q])
            for k, (r, n, anonymous, named) in pairs.items():
                train, query = np.isin(r, t), np.isin(r, q)
                use = q[n[q] > 0]
                if not len(use):
                    continue
                for model, x in [('anonymous', anonymous), ('named', named)]:
                    lr.fit(x[train], y[r[train]], sample_weight=1/n[r[train]])
                    p = lr.predict_proba(x[query])[:, 1]
                    mean = np.bincount(r[query], weights=p, minlength=len(y))[use]/n[use]
                    output[k + '_' + model][use] = mean >= .5
    # v131 replay preserves pre-existing overrides, as in the deployed baseline.
    locked = ref != (scores[:, 0] >= .5)
    for vote in output.values():
        vote[locked] = ref[locked]
    return output


def paired(hit, reference):
    return dict(fixed=int(np.sum(hit & ~reference)), harmed=int(np.sum(~hit & reference)),
                accuracy=float(hit.mean()))


def main():
    row, y, _, groups, _ = load()
    out = ART / 'nov23_pairwise_adjudication'
    out.mkdir(exist_ok=True)
    history, per_split, attainable = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        fold, ref, scores, names = read_opinions(seed, row, y)
        votes = forecast(scores, fold, groups, y, ref)
        hit = {k: v == y for k, v in votes.items()}
        per_split[seed] = {k: paired(h, ref == y) for k, h in hit.items()}
        np.savez_compressed(out / f'{seed}.npz', row=row, y=y, fold=fold, **votes)
        if seed != 'v131':
            attainable.append((scores >= .5) == y[:, None])
            for k, h in hit.items():
                history.setdefault(k, []).append(h)
        print(seed, per_split[seed], flush=True)
    hits = {k: np.stack(h) for k, h in history.items()}
    ref = hits['reference']
    nwrong = (~ref).sum(0)
    bench = np.stack(attainable)
    strata, result = {}, {}
    for times in range(4):
        m = nwrong == times
        can = bench[:, m]
        strata[str(times)] = dict(passengers=int(m.sum()),
            any_same_model_correct_all_three=int(can.all(0).any(1).sum()),
            nobody_ever_correct=int((~can.any((0, 2))).sum()),
            every_model_always_correct=int(can.all((0, 2)).sum()))
        result[str(times)] = {k: dict(paired=[paired(v[m], b[m]) for v, b in zip(h, ref)],
                                      correct_all_three=int(h[:, m].all(0).sum()))
                              for k, h in hits.items()}
    assert sum(s['passengers'] for s in strata.values()) == len(row)
    contrasts = {}
    for k, h in hits.items():
        delta = (h.astype(float) - ref).mean(0)
        contrasts[k] = dict(mean_gain_pp=float(100*delta.mean()), group_ci95_pp=interval(delta, groups))
    write(ROOT / 'reports/nov23_pairwise_adjudication/result.json',
          dict(opinions=names, per_split=per_split, baseline_miss_count=strata,
               by_baseline_miss_count=result, versus_reference=contrasts))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
