"""Fit local-only and nonlocal reference models and learn to adjudicate their disagreements."""
import argparse
import json

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank, logistic
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_withheld_neighborhood_n23 import fit_neighborhoods

DEST = ROOT/'artifacts/nov23_neighborhood_disputes'
NAMES = ['without_near', 'control', 'only_near']


def train(raw, y, groups, fold, panel):
    split = fold if panel == 'full' else fold % 2
    scores = fit_neighborhoods(compact_features(raw), y, groups, split, include_local=True)[:3]
    assert all(np.isfinite(p).all() for p in scores)
    DEST.mkdir(exist_ok=True)
    np.savez_compressed(DEST/f'{panel}.npz', passenger=raw.PassengerId.to_numpy(str), y=y,
                        fold=split, **dict(zip(NAMES, scores, strict=True)))


def features(scores, neighborhood):
    proposal = neighborhood[:, 0] >= .5
    sign = (2*proposal.astype(int)-1)[:, None]
    confidence = logit(scores.clip(.001, .999))*sign
    local = logit(neighborhood.clip(.001, .999))*sign
    return np.column_stack([confidence, proposal]), np.column_stack([confidence, proposal, local]), proposal


def evaluate(raw, y, groups, fold):
    names = [*FAMILIES, 'mitra96']
    full = np.column_stack([load_scores(raw, y, fold, OUT/f'{k}.npz', ['p'])[:, 0] for k in names])
    half = half_bank(raw, y, fold)
    neighborhoods = {k: load_scores(raw, y, f, DEST/f'{k}.npz', NAMES)
                     for k, f in [('full', fold), ('half', fold%2)]}
    small = neighborhoods['half']
    large = neighborhoods['full']
    hx, hnet, hp = features(half, small)
    fx, fnet, fp = features(full, large)
    base, hb = full[:, -1] >= .5, half[:, -1] >= .5
    agree = ((full >= .5) == base[:, None]).all(1)
    hit = base == y
    rescued = agree & ~hit & (fp == y)
    harmed = agree & hit & (fp != y)
    assert rescued.sum() == 116
    cat = full[:, names.index('cat')] >= .5
    cat_fixed = (cat != y) & (fp == y)
    cat_harmed = (cat == y) & (fp != y)
    assert (cat_fixed.sum(), cat_harmed.sum()) == (362, 706)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    _, inverse, size = np.unique(groups, return_inverse=True, return_counts=True)
    paid = raw[BILLS].gt(0).any(axis=1).to_numpy()
    fields = np.column_stack([np.log1p(raw[BILLS].fillna(0)), raw[BILLS].isna(), raw.Age.fillna(-1),
                              raw.CryoSleep.eq(True), raw.CryoSleep.isna(), size[inverse]])
    profiles = {}
    for name, mask in {'all': np.ones(len(y), bool), 'vote_false': ~base, 'vote_true': base,
                       'positive_bill': paid, 'no_positive_bill': ~paid, 'solo': size[inverse] == 1,
                       'multi': size[inverse] > 1, 'historical_hard': hard,
                       'local_also_contrary': (large[:, 2] >= .5) == fp,
                       'control_also_contrary': (large[:, 1] >= .5) == fp}.items():
        profiles[name] = dict(rescued=int((mask & rescued).sum()), harmed=int((mask & harmed).sum()))
    policies = {k: base.copy() for k in ['prior', 'dispute', 'context', 'all_sources']}
    policies['only_near'] = large[:, 2] >= .5
    policies['only_near_mean_mitra'] = large[:, 2]+full[:, -1] >= 1
    for f in np.unique(fold):
        train = np.flatnonzero((fold%2 == f%2) & (fold != f))
        query = np.flatnonzero(fold == f)
        assert not np.isin(groups[train], groups[query]).any()
        t, q = train[hb[train] != hp[train]], query[base[query] != fp[query]]
        for name, tx, qx in [('prior', hx, fx), ('dispute', hnet, fnet)]:
            prob = logistic(tx[t], hp[t] == y[t], qx[q])
            selected = q[prob >= .5]
            policies[name][selected] = fp[selected]
        tree = CatBoostClassifier(iterations=150, depth=3, learning_rate=.05, random_seed=481,
                                  thread_count=2, verbose=False, allow_writing_files=False)
        tree.fit(np.column_stack([hnet[t], fields[t]]), (hp[t] == y[t]).astype(int))
        p = tree.predict_proba(np.column_stack([fnet[q], fields[q]]))[:, 1]
        sel = q[p >= .5]
        policies['context'][sel] = fp[sel]
        # All-order transport target permits decisions even when the nonlocal model agrees with Mitra.
        tx = logit(np.column_stack([half[train], small[train]]).clip(.001, .999))
        qx = logit(np.column_stack([full[query], large[query]]).clip(.001, .999))
        policies['all_sources'][query] = logistic(tx, y[train], qx) >= .5
    gates = {}
    for name, pred in policies.items():
        fixed, injury = ~hit & (pred == y), hit & (pred != y)
        delta = fixed.astype(int)-injury
        gates[name] = dict(accuracy=float((pred == y).mean()), fixed=int(fixed.sum()), harmed=int(injury.sum()),
            common_fixed=int((fixed & agree).sum()), common_harmed=int((injury & agree).sum()),
            recovered_of_116=int((rescued & fixed).sum()), historical_fixed=int((fixed & hard).sum()),
            historical_harmed=int((injury & hard).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
            fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,groups))
    report = ROOT/'reports/nov23_matched_learning_curve/neighborhood_disputes.json'
    local_correct = (large[:, 2] >= .5) == y
    union = ((fp == y) | local_correct) & ~hit
    reach = dict(all_base_errors_covered=int(union.sum()), common_covered=int((union & agree).sum()),
                 common_covered_by_both=int((rescued & local_correct).sum()),
                 historical_covered=int((union & hard).sum()), best_if_rescuing_common_only=float((hit.sum()+(union & agree).sum())/len(y)))
    report.write_text(json.dumps(dict(profiles=profiles, gates=gates, oracle_coverage=reach), indent=2)+'\n')
    print(profiles, gates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('panel', choices=['full', 'half', 'evaluate'])
    args = parser.parse_args()
    with threadpool_limits(2):
        if args.panel == 'evaluate':
            evaluate(*data())
        else:
            train(*data(), args.panel)
