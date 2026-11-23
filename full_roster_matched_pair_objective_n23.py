"""Compare point and pair training on the same data, with clean within-fold score calibration."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_direct_reference_risk_n23 import choose_reference
from full_roster_same_model_calibration_n23 import learn
from full_roster_unexpected_peers_n23 import reference_score


def pair_indices(y, groups, band, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for k in np.unique(band):
        positive = np.flatnonzero((band == k) & y)
        negative = np.flatnonzero((band == k) & ~y)
        if len(negative) and len(positive):
            left = rng.choice(positive, 4*(len(positive)+len(negative)))
            right = rng.choice(negative, len(left))
            keep = groups[left] != groups[right]
            pairs.extend(np.column_stack([left[keep], right[keep]]))
    return np.asarray(pairs)


def centered(donors, query, score, band):
    means = np.bincount(band[donors], weights=score[donors], minlength=20)
    count = np.bincount(band[donors], minlength=20)
    # Five pseudo-observations shrink sparse cells toward the overall donor mean.
    mean = (means+5*score[donors].mean()) / (count+5)
    return (score[query] - mean[band[query]]) / max(score[donors].std(), 1e-6)


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    alt = names.index('balanced_full')
    p = reference_score(full, ref, alt)
    evidence = half[:,[names.index(k) for k in EVIDENCE]]
    proposal, previous = proposal_design(half[:,:5], evidence)
    earth = (raw.HomePlanet.eq('Earth') & raw.CryoSleep.eq(True)).to_numpy()
    focus = (earth & raw.Destination.eq('TRAPPIST-1e').to_numpy() & raw.Cabin.str.endswith(('/P','/S'), na=False).to_numpy() & (p <= .3))
    band = np.minimum((10*p).astype(int), 9) + 10*earth
    rawx = compact_features(raw)
    _, ids, counts = np.unique(group, return_inverse=True, return_counts=True)
    rawx['BookingSize'] = counts[ids]
    a = rawx.assign(**dict(zip(names,half.T,strict=True)))
    b = rawx.assign(**dict(zip(names,full.T,strict=True)))
    score_bank = {k:np.zeros(len(y)) for k in ['point', 'matched_pairs']}
    policies = {k:np.full(len(y), np.nan) for k in score_bank}
    pair_count = []
    for f in np.unique(fold):
        t, q = (fold != f) & (fold % 2 == f % 2), np.flatnonzero(fold == f)
        rt = np.zeros(len(y), bool)
        for h in np.unique(fold[t]):
            held, train = t & (fold == h), t & (fold != h)
            assert not np.isin(group[train], group[held | (fold == f)]).any()
            rt[held] = choose_reference(half, previous, proposal, y, train, held)
        hp = reference_score(half[t], rt[t], alt)
        train_band = np.minimum((10*hp).astype(int),9)+10*earth[t]
        order = np.argsort(train_band, kind='stable')
        train = np.flatnonzero(t)[order]
        grouping = train_band[order]
        opts = dict(iterations=170, depth=4, learning_rate=.055, random_seed=375,
                    thread_count=3, allow_writing_files=False, verbose=False)
        point = CatBoostClassifier(**opts).fit(a.iloc[train],y[train].astype(int),cat_features=CATEGORICAL)
        pair_ids = pair_indices(y[train],group[train],grouping,158+int(f))
        pair_count.append(len(pair_ids))
        data_pool = Pool(a.iloc[train],label=y[train].astype(int),group_id=grouping,
                         pairs=pair_ids,cat_features=CATEGORICAL)
        rank = CatBoostRanker(loss_function='PairLogit', **opts).fit(data_pool)
        score_bank['point'][q] = point.predict(b.iloc[q],prediction_type='RawFormulaVal')
        score_bank['matched_pairs'][q] = rank.predict(b.iloc[q])
        for aa,bb in GroupKFold(4,shuffle=True,random_state=462).split(q,groups=group[q]):
            donors, query = q[aa], q[bb]
            assert not np.isin(group[donors],group[query]).any()
            for k, score in score_bank.items():
                x = centered(donors,donors,score,band)[:,None]
                z = centered(donors,query,score,band)
                w = learn(x, y[donors],logit(p[donors].clip(.001,.999)),np.zeros(1))
                policies[k][query] = expit(logit(p[query].clip(.001,.999)) + z*w[0])
        print('finished fold',f,flush=True)
    result = {}
    assert focus.sum() == 121 and y[focus].sum() == 50
    shared = ((full >= .5) != y[:,None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    for k,prob in dict(reference=p, **policies).items():
        assert np.isfinite(prob).all()
        delta = ((prob >= .5) == y).astype(int) - (ref == y)
        auc = roc_auc_score(y[focus],prob[focus])
        result[k] = dict(accuracy=float(((prob>=.5)==y).mean()), fixed=int((delta==1).sum()), harmed=int((delta==-1).sum()),
            focus_auc=float(auc), focus_fixed=int((delta[focus]==1).sum()), focus_harmed=int((delta[focus]==-1).sum()),
            common_fixed=int((delta[shared]==1).sum()), earth_net=int(delta[earth].sum()),
            additional_net=int(delta.sum()-delta[dev].sum()), fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
            ci_pp=interval(delta,group))
    report = dict(results=result, pairs_by_fold=pair_count)
    (ROOT/'reports/nov23_matched_learning_curve/matched_pair_objective.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    run()
