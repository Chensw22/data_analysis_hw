"""Replay the queued linear gate and measure what independent transformer votes add."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from half_roster_disagreement_eval_n23 import DEST, NEIGHBOR, inputs


def run():
    raw, y, groups, fold = data()
    names = [*FAMILIES, 'mitra96']
    half = half_bank(raw, y, fold)
    full = np.column_stack([load_scores(raw, y, fold, OUT/f'{k}.npz', ['p'])[:, 0] for k in names])
    cols = [names.index(k) for k in ('mitra96', 'cat', 'budget')]
    with np.load(DEST/'others.npz') as z:
        for k, expected in [('passenger', raw.PassengerId), ('panel', fold%2), ('y', y)]:
            np.testing.assert_array_equal(z[k], expected)
        hn = np.column_stack([z[k] for k in NEIGHBOR])
    fn = load_scores(raw, y, fold, ROOT/'artifacts/nov23_full_roster_metric_neighbors/oof.npz', NEIGHBOR)
    hx, hb, hp = inputs(np.column_stack([half[:, cols], hn]), raw, groups)
    fx, base, fp = inputs(np.column_stack([full[:, cols], fn]), raw, groups)
    gate = base.copy()
    for f in np.unique(fold):
        t = np.flatnonzero((fold%2 == f%2) & (fold != f) & (hb != hp))
        q = np.flatnonzero((fold == f) & (base != fp))
        assert not np.isin(groups[t], groups[q]).any()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
        model.fit(hx['linear'][t], (hp[t] == y[t]).astype(int))
        use = q[model.predict_proba(fx['linear'][q])[:, 1] >= .5]
        gate[use] = fp[use]
    assert ((gate == y) & (base != y)).sum() == 107 and ((gate != y) & (base == y)).sum() == 93
    positive, exception = (load_scores(raw, y, fold, DEST/'transformer_oof.npz', ['transport','error']) >= .5).T
    approve = gate != base
    first, second = positive != base, exception != base
    agree = (full >= .5) == base[:, None]
    unanimous = agree.all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y), bool)
    hard[dev] = hard_cohorts(dev, y[dev])[2]
    result = {'patterns': {}, 'policies': {}}
    for a in [False, True]:
        for b in [False, True]:
            for c in [False, True]:
                sel = (approve == a) & (first == b) & (second == c)
                result['patterns'][f'{int(a)}{int(b)}{int(c)}'] = dict(rows=int(sel.sum()),
                    baseline_errors=int((sel & (base != y)).sum()), common_errors=int((sel & unanimous & (base != y)).sum()))
    for key, flips in {'queued_gate': approve, 'add_exception': approve | second,
                       'add_both_transformers': approve | (first & second),
                       'majority': (approve.astype(int)+first+second) >= 2}.items():
        pred = base ^ flips
        correct = pred == y
        changed = correct.astype(int)-(base == y)
        vs_gate = correct.astype(int)-(gate == y)
        result['policies'][key] = dict(accuracy=float(correct.mean()), fixed=int((changed == 1).sum()),
            harmed=int((changed == -1).sum()), net=int(changed.sum()), extra_fixed=int((vs_gate == 1).sum()),
            extra_harmed=int((vs_gate == -1).sum()), extra_common_fixed=int(((vs_gate == 1) & unanimous & (base != y)).sum()),
            extra_historical_fixed=int(((vs_gate == 1) & hard).sum()), net_additional=int(vs_gate.sum()-vs_gate[dev].sum()),
            vs_gate_by_fold=[int(vs_gate[fold == f].sum()) for f in np.unique(fold)],
            vs_gate_ci_pp=interval(vs_gate, groups))
    path = ROOT/'reports/nov23_matched_learning_curve/joint_error_choices.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(result['policies'])


if __name__ == '__main__':
    run()
