"""Transfer held-out model errors from similar bookings to a reference prediction."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_booking_transport_n24 import vote
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_same_model_calibration_n23 import learn


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(reference, gate >= .5)
    anchor = logit(gate.clip(.001, .999))
    shared = ((full >= .5) != y[:, None]).all(1)
    unanimous_correct = ((full >= .5) == y[:, None]).all(1)
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as saved:
        additional = np.ones(len(y), bool)
        additional[saved['development']] = False

    features = passenger_vectors(raw)
    expert_view = logit(half[:, :5].clip(.02, .98)) / 2
    views = {'attributes': features, 'attributes_and_experts': np.column_stack([features, expert_view])}
    results = {}
    for view, x in views.items():
        # In each fold, donor and query scores come from the same fitted experts.
        # All fitting data for those experts belong to the opposite panel.
        votes, error = vote(x, np.column_stack([y, half[:, 4]]), group, fold, panel=fold % 2)
        estimates = {}
        for matching in ['individual', 'transport']:
            local_y, local_expected = ((48 * votes[matching] + 1) / 50).T
            residual = logit(local_y) - logit(local_expected)
            learned = np.zeros(len(y))
            for f in np.unique(fold):
                outer = np.flatnonzero(fold == f)
                splits = GroupKFold(4, shuffle=True, random_state=462)
                for a, b in splits.split(outer, groups=group[outer]):
                    train, test = outer[a], outer[b]
                    assert not np.isin(group[train], group[test]).any()
                    weight = learn(residual[train, None], y[train], anchor[train], np.zeros(1))
                    learned[test] = expit(anchor[test] + weight[0] * residual[test])
            policies = {
                'direct': local_y, 'half_residual': expit(anchor + .5 * residual),
                'full_residual': expit(anchor + residual), 'learned_residual': learned,
            }
            for policy, score in policies.items():
                assert np.isfinite(score).all()
                delta = ((score >= .5) == y).astype(int) - (reference == y)
                estimates[matching + '_' + policy] = dict(
                    accuracy=float(((score >= .5) == y).mean()),
                    fixed=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
                    common_fixed=int((delta[shared] == 1).sum()),
                    unanimous_harmed=int((delta[unanimous_correct] == -1).sum()),
                    additional_net=int(delta[additional].sum()),
                    fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],
                    ci_pp=interval(delta, group),
                )
        results[view] = dict(marginal_error=error, estimates=estimates)
        for k, v in estimates.items():
            print(view, k, v['fixed'], v['harmed'], v['common_fixed'], flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/donor_residual.json'
    path.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
