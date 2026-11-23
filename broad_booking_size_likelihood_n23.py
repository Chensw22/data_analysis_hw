"""Estimate outcome evidence from the observed party size conditional on each possible outcome."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import REPORT
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development


def probabilities(frame, target, train, query, expected):
    model = CatBoostClassifier(iterations=200, depth=5, learning_rate=.08, thread_count=2,
                               verbose=False, random_seed=617, allow_writing_files=False)
    model.fit(frame.iloc[train], target[train], cat_features=CATEGORICAL)
    np.testing.assert_array_equal(model.classes_, expected)
    return model.predict_proba(frame.iloc[query])


def outcome_evidence(frame, y, size, t, q):
    individual = frame.drop(columns='BookingSize')
    bucket = np.minimum(size, 4) - 1
    own = probabilities(individual, y, t, q, [False, True])[:, 1]
    direct = probabilities(frame, y, t, q, [False, True])[:, 1]
    permuted = y.copy()
    strata_train = pd.DataFrame(dict(planet=frame.HomePlanet.iloc[t].to_numpy(),
                                     sleep=frame.CryoSleep.iloc[t].to_numpy(), spending=strata(frame.iloc[t])))
    rng = np.random.default_rng(223)
    for ix in strata_train.groupby(list(strata_train)).indices.values():
        permuted[t[ix]] = rng.permutation(y[t[ix]])
    output = dict(own=logit(own), direct=logit(direct))
    for name, split in [('true', y), ('sham', permuted)]:
        logz = []
        for flag in [False, True]:
            members = t[split[t] == flag]
            p = probabilities(individual, bucket, members, q, np.arange(4))
            logz.append(np.log(p[np.arange(len(q)), bucket[q]].clip(.001)))
        output[name] = logz[1] - logz[0]
    return output


def main():
    row, y, frame, group, _ = load()
    size = frame.BookingSize.to_numpy()
    ids, labels, masks = historical()
    np.testing.assert_array_equal(ids, row)
    np.testing.assert_array_equal(labels, y)
    bills = strata(frame)
    masks = {**masks, 'solo_repeat':masks['repeat'] & (size == 1),
             'paid_transported':masks['repeat'] & (bills == 2) & y,
             'zero_stayed':masks['repeat'] & (bills == 0) & ~y}
    _, release, _ = development()
    out = ART / 'nov23_booking_size_likelihood'
    out.mkdir(exist_ok=True)
    plans, hist, refs = {}, {}, []
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            np.testing.assert_array_equal(cache['row'], row)
            np.testing.assert_array_equal(cache['y'], y)
            fold, ref, base = cache['fold'], cache['ref'], cache['base']
        np.testing.assert_array_equal(ref, release[131] if seed == 'v131' else base >= .5)
        raw = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            t = np.flatnonzero((fold != f) & ~np.isin(group, group[q]))
            assert not np.intersect1d(group[t], group[q]).size
            for k, p in outcome_evidence(frame, y, size, t, q).items():
                raw.setdefault(k, np.zeros(len(y)))[q] = p
        prediction = {k:raw[k] >= 0 for k in ['own','direct']}
        prediction['bayes'] = raw['own'] + raw['true'] >= 0
        prediction['bayes_sham'] = raw['own'] + raw['sham'] >= 0
        for mode in ['direct','true','sham']:
            x = np.c_[logit(base.clip(.002,.998)), raw['direct'], raw['own']]
            if mode != 'direct':
                x = np.c_[x, raw[mode]]
            vote = ref.copy()
            for f in np.unique(fold):
                ix = np.flatnonzero((fold == f) & (ref == (base >= .5)))
                for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ix,groups=group[ix]):
                    train, query = ix[a], ix[b]
                    vote[query] = calibrated(x[train],x[query],y,train) >= .5
            prediction['combined_' + mode] = vote
        plans[seed] = {}
        anchor = ref == y
        if seed != 'v131':
            refs.append(anchor)
        for k, vote in prediction.items():
            h = vote == y
            plans[seed][k] = dict(accuracy=float(h.mean()), fixed=int((h & ~anchor).sum()), harmed=int((~h & anchor).sum()))
            if seed != 'v131':
                hist.setdefault(k,[]).append(h)
        np.savez_compressed(out / f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**raw)
        print(seed, plans[seed]['combined_true'], flush=True)
    write(REPORT / 'size_likelihood.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
