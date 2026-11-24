"""Sum over all possible outcomes for a booking to infer each passenger's probability."""
import itertools
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities


def booking_states(raw, group, probabilities, y):
    sleep = raw.CryoSleep
    cabin = raw.Cabin
    score = logit(probabilities.clip(.005, .995))
    _, inverse = np.unique(group, return_inverse=True)
    size = np.bincount(inverse)
    bookings = np.split(np.argsort(inverse, kind='stable'), np.cumsum(size)[:-1])
    batches = []
    for n in np.unique(size[size > 1]):
        rows = np.stack([rows for rows in bookings if len(rows) == n])
        states = np.array(list(itertools.product([0, 1], repeat=n)))
        spin = 2 * states - 1
        i, j = np.triu_indices(n, 1)
        def same_known(column):
            a = column.fillna('__unknown__').to_numpy()[rows]
            return (a[:, i] == a[:, j]) & (a[:, i] != '__unknown__')
        sides = score[rows] >= 0
        features = np.stack([np.ones((len(rows), len(i))), same_known(sleep), same_known(cabin),
                             sides[:, i] == sides[:, j]], axis=-1)
        pair_term = np.einsum('se,gek->gsk', spin[:, i]*spin[:, j], features)/(n-1)
        unary = np.einsum('sn,gn->gs', states, score[rows])
        terms = np.concatenate([np.broadcast_to(states.sum(1)[None, :, None], (*unary.shape, 1)),
                                unary[:, :, None], pair_term], axis=2)
        observed = y[rows].astype(int) @ (2**np.arange(n-1, -1, -1))
        np.testing.assert_array_equal(states[observed], y[rows])
        batches.append((rows, states, unary, terms, observed))
    return batches, size[inverse] > 1


def fit(batches, training, width):
    def objective(w):
        loss, gradient = 5*(w @ w), 10*w
        for rows, states, unary, terms, obs in batches:
            take = training[rows[:, 0]]
            if not take.any():
                continue
            x = terms[take, :, :width]
            energy = unary[take] + x @ w
            norm = logsumexp(energy, axis=1)
            weight = np.exp(energy - norm[:, None])
            idx = np.arange(take.sum())
            loss += (norm-energy[idx, obs[take]]).sum()
            gradient += np.einsum('bs,bsj->j', weight, x) - x[idx, obs[take]].sum(0)
        return loss, gradient
    sol = minimize(objective, np.zeros(width), jac=True, method='L-BFGS-B')
    assert sol.success, sol.message
    return sol.x


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    baseline = latest_reference(y, group, fold, names, half, full)
    gate, _ = gate_probabilities(names, half, full, y, group, fold)
    np.testing.assert_array_equal(gate >= .5, baseline)
    batches, multiple = booking_states(raw, group, gate, y)
    shared = ((full >= .5) != y[:, None]).all(1)
    unanimous_correct = ((full >= .5) == y[:, None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as archive:
        additional = np.ones(len(y), bool)
        additional[archive['development']] = False
    report = dict(multiple=int(multiple.sum()), multi_errors=int((multiple & (baseline != y)).sum()),
                  multi_common_errors=int((multiple & shared).sum()), methods={})
    for key, width in [('unary', 2), ('shared_interaction', 3), ('context_interaction', 6)]:
        prediction, with_known_peers = gate.copy(), gate.copy()
        assigned = np.zeros(len(y), int)
        joint_gain = 0.
        for f in np.unique(fold):
            outer = np.flatnonzero((fold == f) & multiple)
            for t, q in GroupKFold(4, shuffle=True, random_state=462).split(outer, groups=group[outer]):
                training = np.zeros(len(y), bool)
                training[outer[t]] = True
                query = np.zeros(len(y), bool)
                query[outer[q]] = True
                assert not np.isin(group[training], group[query]).any()
                coef = fit(batches, training, width)
                for rows, states, unary, terms, obs in batches:
                    use = query[rows[:, 0]]
                    if not use.any():
                        continue
                    ids = rows[use]
                    assert query[ids].all()
                    energy = unary[use] + terms[use, :, :width] @ coef
                    logprob = energy-logsumexp(energy, axis=1, keepdims=True)
                    marg = np.exp(logprob) @ states
                    prediction[ids] = marg
                    assigned[ids] += 1
                    # Compare with a distribution with the exact same fitted marginals.
                    idx = np.arange(len(ids))
                    independent = np.where(y[ids], marg, 1-marg)
                    joint_gain += (logprob[idx, obs[use]] - np.log(independent).sum(1)).sum()
                    # Retrospective diagnostic: hold every other outcome at its true value.
                    for k in range(ids.shape[1]):
                        bit = 1 << (ids.shape[1]-k-1)
                        off = obs[use] & ~bit
                        high, low = energy[idx, off | bit], energy[idx, off]
                        with_known_peers[ids[:, k]] = np.exp(high - np.logaddexp(high, low))
        np.testing.assert_array_equal(assigned, multiple.astype(int))
        def evaluate(p):
            delta = ((p >= .5) == y).astype(int) - (baseline == y)
            return dict(accuracy=float(((p >= .5) == y).mean()), fixed=int((delta == 1).sum()),
                        harmed=int((delta == -1).sum()), common_fixed=int((delta[shared] == 1).sum()),
                        unanimous_harmed=int((delta[unanimous_correct] == -1).sum()),
                        additional_net=int(delta[additional].sum()),
                        fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta,group))
        if key == 'unary':
            np.testing.assert_allclose(with_known_peers, prediction, atol=1e-10)
            assert abs(joint_gain) < 1e-8
        report['methods'][key] = dict(marginal=evaluate(prediction),
                                      diagnostic_true_peers=evaluate(with_known_peers),
                                      joint_log_gain_per_multi_passenger=joint_gain/multiple.sum())
        print(key, report['methods'][key], flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/order_partition.json'
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
