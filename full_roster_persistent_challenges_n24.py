"""Test which alternative predictions persist when the training orders change."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_repartition_gate_n24 import ROOT, panel_data
from full_roster_same_model_calibration_n23 import learn


def run():
    raw, y, group, fold, names, half, full, ref, _, cols, rotated = panel_data()
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, ref)
    banks = np.stack([full[:, cols], half[:, cols], rotated], axis=2)
    # Positive margins are votes to reverse the common, fixed reference decision.
    margin = logit(banks.clip(.005, .995)) * np.where(ref, -1., 1.)[:, None, None]
    current = margin[:, :, 0] > 0
    stable = np.all(margin > 0, axis=2)
    shared = ((full >= .5) != y[:, None]).all(1)
    wrong = ref != y
    disputed = (full >= .5).any(1) & ~(full >= .5).all(1)
    assert disputed.sum() == 5061 and (wrong & disputed).sum() == 1318
    assert not (stable.any(1) & ~disputed).any()
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as saved:
        additional = np.ones(len(y), bool)
        additional[saved['development']] = False
    def summary(flip):
        change = np.where(flip, np.where(wrong, 1, -1), 0)
        return dict(fixed=int((change==1).sum()), harmed=int((change==-1).sum()),
                    accuracy=float(((ref ^ flip)==y).mean()), common_fixed=int((flip & shared).sum()),
                    additional_net=int(change[additional].sum()),
                    fold_net=[int(change[fold==f].sum()) for f in np.unique(fold)],
                    ci_pp=interval(change,group))
    def count(mask):
        return dict(rows=int(mask.sum()),fixed=int((mask & wrong).sum()),harmed=int((mask & ~wrong).sum()))
    group_coverage = {}
    by_family = {}
    for j, col in enumerate(cols):
        by_family[names[col]] = {'single_fit':count(current[:,j]), 'three_fits':count(stable[:,j]),
                                'only_stable_challenger':count(stable[:,j] & (stable.sum(1)==1))}
    for k in range(banks.shape[1]+1):
        group_coverage[str(k)] = count(disputed & (stable.sum(1) == k))
    risk = np.minimum(gate, 1-gate)
    anchor = logit(risk.clip(.001,.999))
    fresh = np.maximum(margin[:, :, 0], 0)
    persistent = np.maximum(margin.min(2), 0)
    temporary = fresh - persistent
    assert np.all(temporary>=0)
    results = {}
    for key, x in [('current_only',fresh),('three_fit_minimum',persistent),
                   ('persistent_and_remainder',np.column_stack([persistent,temporary]))]:
        eligible = x.any(1)
        pred = np.zeros(len(y),bool)
        for f in np.unique(fold):
            outer = np.flatnonzero(fold==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
                t,q = outer[a],outer[b]
                t,q = t[eligible[t]],q[eligible[q]]
                assert not np.isin(group[t], group[q]).any()
                w = learn(x[t], wrong[t], anchor[t], np.zeros(x.shape[1]))
                pred[q] = expit(anchor[q] + x[q] @ w) > .5
        results[key] = summary(pred)
    out = dict(families=by_family, disputed_by_stable_count=group_coverage, rules=results,
               any_current=count(current.any(1)), any_persistent=count(stable.any(1)))
    path = ROOT/'reports/nov23_matched_learning_curve/persistent_challenges.json'
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(out['any_current'],out['any_persistent'],results)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
