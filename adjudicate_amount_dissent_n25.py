"""Test whether information available before changing a passenger's label distinguishes rescues from damage."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from disjoint_tree_fields_n25 import ART, DEST
from evaluate_bill_precision_n25 import BASE, COLS, GROUPS, OUT, REF, TAB
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds


def odds(probability):
    return logit(np.clip(probability, .002, .998))


def load_fold(k, q, t, raw, y):
    def read(path, names):
        with np.load(path) as z:
            np.testing.assert_array_equal(z['q'], q)
            if 't' in z.files:
                np.testing.assert_array_equal(z['t'], t)
            if 'labels' in z.files:
                np.testing.assert_array_equal(z['labels'], y[t])
            if 'passenger' in z.files:
                np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[q])
            return [z[n] for n in names]

    native, prior, low, high = read(REF/f'release{k}.npz',
        ['native', 'v131', 'nine_on_low_branch', 'nine_on_high_branch'])
    full = np.column_stack(read(OUT/f'{k}_exact_bill_position.npz', COLS))
    binned = np.column_stack(read(BASE/f'fold{k}.npz', COLS))
    observed, neutral = read(OUT/f'{k}_query_intervention.npz',
        ['exact_bill_position_observed', 'exact_bill_position_none'])
    np.testing.assert_array_equal(full[:, 0], observed)
    tab = np.column_stack([read(TAB/f'{k}_{s}.npz', ['p'])[0] for s in GROUPS])
    ref = native >= .5
    sign = np.where(ref, -1., 1.)
    signed = sign[:, None] * odds(full)
    tab_evidence = sign[:, None] * odds(tab)
    change = sign * (odds(observed) - odds(neutral))
    # Positive scores always support reversing the incumbent's native label.
    anchor = np.column_stack([ref, -abs(odds(native))])
    features = dict(anchor=anchor, exact=np.column_stack([anchor, signed, change]),
        paired=np.column_stack([anchor, signed, change, sign[:, None]*(odds(full)-odds(binned)),
                                tab_evidence, np.minimum(signed[:, 1:], tab_evidence)]))
    return ref, prior, low, high, features, (full >= .5) != ref[:, None], change


def counts(delta, scope, fold):
    return dict(rows=int(scope.sum()), fixed=int((delta[scope]>0).sum()),
        harmed=int((delta[scope]<0).sum()), by_fold=[int(delta[scope & (fold==k)].sum()) for k in range(8)])


def run():
    raw, y, group, fold = data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows, hf = z['rows'], z['fold']
    _, panels = prediction_sets(raw, y, group, fold, rows, hf)
    full_votes, old_votes = [(np.column_stack(p) >= .5).sum(1) for p in panels]
    majority = full_votes > 34
    common = (full_votes==0) | (full_votes==68)
    repeated = np.zeros(len(y), bool)
    repeated[rows] = common[rows] & ((old_votes==0) | (old_votes==68)) & ((old_votes>34)==majority[rows])
    new = ~np.isin(group, group[rows])
    records = {}
    for k in range(8):
        q, t = np.flatnonzero(fold==k), np.flatnonzero(fold!=k)
        ref, prior, low, high, features, alternatives, change = load_fold(k, q, t, raw, y)
        assert not np.isin(group[q], group[t]).any()
        eligible = alternatives.any(1)
        truth = y[q]
        error = ref != truth
        known = prior >= 0
        # Reconstruct the original half-step using the same numeric tie handling as the baseline.
        half = np.where(ref, .5*change > -features['anchor'][:, 1], .5*change >= -features['anchor'][:, 1])
        fits = {'all_bookings': list(GroupKFold(4, shuffle=True, random_state=3207).split(q, groups=group[q])),
                'new_bookings': [(np.flatnonzero(~new[q]), np.flatnonzero(new[q]))]}
        for design, splits in fits.items():
            for a, b in splits:
                assert not np.isin(group[q[a]],group[q[b]]).any()
                # Train only on people whom at least one precision view disputes. At evaluation,
                # preserve the incumbent unless probability of a successful correction exceeds half.
                a = a[eligible[a]]
                gates = {'half_transfer': half[b]}
                for name, matrix in features.items():
                    fit = LogisticRegression(C=.2, solver='liblinear', max_iter=500)
                    fit.fit(matrix[a], error[a])
                    accepted = fit.predict_proba(matrix[b])[:, 1] > .5
                    gates[name+'_broad'] = accepted & eligible[b]
                    gates[name+'_half'] = accepted & half[b] & eligible[b]
                for name, accept in gates.items():
                    pred = ref[b] ^ accept
                    lower, upper = edit_effect_bounds(pred, ref[b], prior[b], low[b], high[b], truth[b])
                    np.testing.assert_array_equal(lower[known[b]], upper[known[b]])
                    assert ((lower<=upper)&(lower>=-1)&(upper<=1)).all()
                    records.setdefault((design,name),[]).append((q[b],known[b],lower,upper))
    report = {}
    for (design, name), batches in records.items():
        order, known, lower, upper = (np.concatenate([v[i] for v in batches]) for i in range(4))
        expected = np.arange(len(y)) if design=='all_bookings' else np.flatnonzero(new)
        np.testing.assert_array_equal(np.sort(order), expected)
        result = dict(full_net_bounds=[int(lower.sum()),int(upper.sum())])
        # Jointly report corrections and collateral errors on the exact same scope.
        for key, scope in [('known',known), ('unanimous',known&common[order]), ('repeated',known&repeated[order]),
                           ('unanimous_wrong',known&common[order]&(majority[order]!=y[order]))]:
            result[key] = counts(lower,scope,fold[order])
        report.setdefault(design,{})[name] = result
        print(design,name,result['known']['fixed'],result['known']['harmed'],result['full_net_bounds'])
    original = report['all_bookings']['half_transfer']['known']
    assert (original['fixed'],original['harmed'])==(87,94)
    DEST.with_name('adjudicate_amount_dissent.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
