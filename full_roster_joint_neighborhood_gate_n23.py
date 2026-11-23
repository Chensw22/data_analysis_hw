"""Compare a joint neighborhood gate with the exact cross-fitted gate in the 109-change candidate."""
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


def view(scores, neighbors, local, raw, groups):
    idx = [(*FAMILIES, 'mitra96').index(k) for k in ('mitra96', 'cat', 'budget')]
    design, base, pick = inputs(np.column_stack([scores[:, idx], neighbors]), raw, groups)
    opp = (1-2*base.astype(int))[:, None]
    broad = np.column_stack([scores, neighbors, local])
    terms = logit(broad.clip(.001, .999))*opp
    area = (local >= .5) != base[:, None]
    dispute = pick != base
    return base, dispute, area, design['linear'], np.column_stack([terms, base, dispute, area])


def run():
    raw, y, g, fold = data()
    half = half_bank(raw, y, fold)
    full = np.column_stack([load_scores(raw, y, fold, OUT/f'{k}.npz', ['p'])[:, 0] for k in (*FAMILIES,'mitra96')])
    with np.load(METRIC/'others.npz') as z:
        for k, v in [('passenger',raw.PassengerId), ('panel',fold%2), ('y',y)]:
            np.testing.assert_array_equal(z[k], v)
        hn = np.column_stack([z[k] for k in NEIGHBOR])
    fn = load_scores(raw, y, fold, ROOT/'artifacts/nov23_full_roster_metric_neighbors/oof.npz', NEIGHBOR)
    hc, hd, ha, hx, hall = view(half, hn, load_scores(raw,y,fold%2,DEST/'half.npz',NAMES), raw, g)
    fc, fd, fa, fx, fall = view(full, fn, load_scores(raw,y,fold,DEST/'full.npz',NAMES), raw, g)
    gates = {k: fc.copy() for k in ['queued', 'refit_existing', 'expand', 'preserve_and_expand', 'repeated_mean',
                                    'repeat_1', 'repeat_2', 'repeat_3']}
    for f in np.unique(fold):
        t, q = (fold%2 == f%2) & (fold != f), fold == f
        assert not np.isin(g[q], g[t]).any()
        for key, train_mask, test_mask, x, xt in [
                ('queued', hd, fd, hx, fx),
                ('refit_existing', hd, fd, hall, fall),
                ('expand', hd | ha[:, 0] | ha[:, 2], fd | fa[:, 0] | fa[:, 2], hall, fall),
                ('preserve_and_expand', ~hd & (ha[:, 0] | ha[:, 2]), ~fd & (fa[:, 0] | fa[:, 2]), hall, fall)]:
            a, b = np.flatnonzero(t & train_mask), np.flatnonzero(q & test_mask)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
            p = model.fit(x[a], y[a] != hc[a]).predict_proba(xt[b])[:, 1]
            swap = b[p >= .5]
            gates[key][swap] = ~fc[swap]
            if key == 'preserve_and_expand':
                estimates = []
                for i, omit in enumerate(np.unique(fold[a])):
                    keep = a[fold[a] != omit]
                    s = model.fit(x[keep], y[keep] != hc[keep]).predict_proba(xt[b])[:, 1]
                    estimates.append(s)
                    change = b[s >= .5]
                    gates[f'repeat_{i+1}'][change] = ~fc[change]
                change = b[np.mean(estimates, axis=0) >= .5]
                gates['repeated_mean'][change] = ~fc[change]
    for key in ['preserve_and_expand', 'repeated_mean', 'repeat_1', 'repeat_2', 'repeat_3']:
        gates[key][fd] = gates['queued'][fd]
    queued = gates['queued']
    assert (((queued == y) & (fc != y)).sum(), ((queued != y) & (fc == y)).sum()) == (107, 93)
    agree = ((full >= .5) == fc[:, None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    result = {}
    for key, pred in gates.items():
        delta = (pred == y).astype(int)-(queued == y)
        fixed, harm = delta > 0, delta < 0
        result[key] = dict(correct=int((pred == y).sum()), accuracy=float((pred == y).mean()), relative_to_queued=dict(
            fixed=int(fixed.sum()), harmed=int(harm.sum()), original_common_recovered=int((fixed & agree & (fc != y)).sum()),
            original_common_lost=int((harm & agree & (fc != y)).sum()), historical_fixed=int((fixed & hard).sum()),
            historical_harmed=int((harm & hard).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
            fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,g)))
    path = ROOT/'reports/nov23_matched_learning_curve/joint_neighborhood_gate.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    run()
