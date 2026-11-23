"""Evaluate a balanced odds expert on cases left untouched by the queued gate."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_neighborhood_disputes_n23 import DEST, NAMES
from half_roster_disagreement_eval_n23 import DEST as METRIC, NEIGHBOR, inputs


EVIDENCE = ['cue','balanced_full','restored_context','offset_context']


def proposal_design(base_scores, evidence):
    counter = evidence[:,1] >= .5
    sign = (2*counter.astype(int)-1)[:,None]
    signed = sign*logit(np.column_stack([base_scores,evidence]).clip(.001,.999))
    return counter != (base_scores[:,-1] >= .5), np.column_stack([signed, counter])


def design(base_scores, neighbor, evidence, raw, groups):
    cols = [(*FAMILIES, 'mitra96').index(k) for k in ('mitra96','cat','budget')]
    orig, ref, pick = inputs(np.column_stack([base_scores[:,cols], neighbor]), raw, groups)
    conflict, signed = proposal_design(base_scores,evidence)
    return orig['linear'], ref, pick != ref, conflict, signed


def run():
    raw, y, groups, fold = data()
    full = np.column_stack([load_scores(raw,y,fold,OUT/f'{k}.npz',['p'])[:,0] for k in (*FAMILIES,'mitra96')])
    half = half_bank(raw,y,fold)
    with np.load(METRIC/'others.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('panel',fold%2)]:
            np.testing.assert_array_equal(z[k],v)
        hn = np.column_stack([z[k] for k in NEIGHBOR])
    fn = load_scores(raw,y,fold,ROOT/'artifacts/nov23_full_roster_metric_neighbors/oof.npz',NEIGHBOR)
    prior = ROOT/'artifacts/nov23_bill_prior_balance'
    h = load_scores(raw,y,fold%2,prior/'half.npz',EVIDENCE)
    b = load_scores(raw,y,fold,prior/'full.npz',EVIDENCE)
    hx, hb, hd, ha, hall = design(half,hn,h,raw,groups)
    fx, base, fd, fa, fall = design(full,fn,b,raw,groups)
    paths = {k:base.copy() for k in ['queued','general','extra','extra_scores_only','extra_consensus']}
    full_same = ((full >= .5) == base[:,None]).all(1)
    half_same = ((half >= .5) == hb[:,None]).all(1)
    for f in np.unique(fold):
        t,q = (fold%2 == f%2) & (fold != f), fold == f
        assert not np.isin(groups[t],groups[q]).any()
        for name, mt, mq, x, test in [
                ('queued',hd,fd,hx,fx),('general',ha,fa,hall,fall),
                ('extra',ha & ~hd,fa & ~fd,hall,fall),
                ('extra_scores_only',ha & ~hd,fa & ~fd,hall[:,:5],fall[:,:5]),
                ('extra_consensus',ha & ~hd & half_same,fa & ~fd & full_same,hall,fall)]:
            a,bq = np.flatnonzero(t & mt), np.flatnonzero(q & mq)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=.10,max_iter=500))
            prob = clf.fit(x[a], y[a] != hb[a]).predict_proba(test[bq])[:,1]
            idx = bq[prob >= .5]
            paths[name][idx] = ~base[idx]
    queued = paths['queued']
    assert (((queued == y) & (base != y)).sum(), ((queued != y) & (base == y)).sum()) == (107,93)
    for name in ['extra','extra_scores_only','extra_consensus']:
        paths[name][fd] = queued[fd]
    geo = load_scores(raw,y,fold,DEST/'full.npz',NAMES)
    seven = full_same & ((geo[:,[0,2]] >= .5) == base[:,None]).all(1)
    views = load_scores(raw,y,fold,ROOT/'artifacts/nov23_sleep_bill_factorial/oof.npz',['full'])
    residual = seven & (base != y)
    missed_all = residual & ((views >= .5) != y[:,None]).all(1)
    assert (residual.sum(), missed_all.sum()) == (819,532)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y),bool)
    hard[dev] = hard_cohorts(dev,y[dev])[2]
    result = {}
    for name, pred in paths.items():
        delta = (pred == y).astype(int)-(queued == y)
        result[name] = dict(accuracy=float((pred == y).mean()), fixed_vs_queued=int((delta == 1).sum()),
            harmed_vs_queued=int((delta == -1).sum()), of_819=int(((pred == y) & residual).sum()),
            of_532=int(((pred == y) & missed_all).sum()), historical_correct=int(((pred == y) & hard).sum()),
            additional_net=int(delta.sum()-delta[dev].sum()), by_fold=[int(delta[fold == f].sum()) for f in np.unique(fold)],
            ci_pp=interval(delta, groups))
    path = ROOT/'reports/nov23_matched_learning_curve/adaptive_bill_prior.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    run()
