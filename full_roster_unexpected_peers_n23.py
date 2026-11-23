"""Learn incremental evidence from companions after accounting for the traveler's attributes."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from broad_booking_policy_replay_n23 import interval
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_direct_reference_risk_n23 import choose_reference
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference


def signals(raw, scores, group):
    sleep = raw.CryoSleep.fillna(-1).astype(float).to_numpy()
    levels = pd.get_dummies(raw[['HomePlanet', 'Destination']], dummy_na=True, dtype=float)
    fees = raw[BILLS].fillna(0).to_numpy()
    own = np.column_stack([
        levels, sleep == 1, sleep == 0, raw.Age.fillna(-1), raw.Age.isna(), raw.VIP.eq(True),
        np.log1p(fees), fees > 0, raw[BILLS].isna(), scores[:, :5], scores[:, 5:].mean(axis=1)
    ])
    _, ids, size = np.unique(group, return_inverse=True, return_counts=True)
    totals = np.zeros((len(size), own.shape[1]))
    np.add.at(totals, ids, own)
    peer = (totals[ids] - own) / np.maximum(size[ids, None] - 1, 1)
    peer[size[ids] == 1] = 0
    # Separate real zero spending from a missing cabin or an unavailable companion.
    cabin = pd.get_dummies(raw.Cabin.str.split('/').str[0], dummy_na=True, dtype=float)
    x = np.column_stack([own, cabin, np.log1p(size[ids])])
    return x, peer, size[ids] > 1


def residuals(a, b, peers_a, peers_b, t, q, fold):
    estimated = np.zeros_like(peers_a)
    def predict(train, query_x):
        scale = StandardScaler().fit(a[train])
        model = Ridge(alpha=20).fit(scale.transform(a[train]), peers_a[train])
        return model.predict(scale.transform(query_x))
    for h in np.unique(fold[t]):
        held = t & (fold == h)
        estimated[held] = predict(t & ~held, a[held])
    return peers_a[t] - estimated[t], peers_b[q] - predict(t, b[q])


def correction(x, y, p):
    offset = logit(p.clip(.001, .999))
    def loss(w):
        z = offset + x @ w
        return (np.logaddexp(0, z).sum() - y @ z + 10 * (w @ w),
                x.T @ (expit(z) - y) + 20 * w)
    fit = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B')
    assert fit.success, fit.message
    return fit.x


def reference_score(scores, ref, alternative):
    """Replay exported threshold scores, including the prior-balanced expert on overrides.

    For outcome probabilities on conflicts use gate_probabilities from
    full_roster_gate_probability_n23 instead.
    """
    baseline = scores[:, 4]
    p = np.where((baseline >= .5) == ref, baseline, scores[:, alternative])
    np.testing.assert_array_equal(p >= .5, ref)
    return p


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    alt = names.index('balanced_full')
    full_p = reference_score(full, ref, alt)
    a, peer_a, multi = signals(raw, half, group)
    b, peer_b, _ = signals(raw, full, group)
    earth = raw.HomePlanet.eq('Earth').to_numpy() & raw.CryoSleep.eq(True).to_numpy()
    eligible = multi & earth
    keys = ['raw_peers', 'unexpected_peers', 'unexpected_with_sleep_interaction', 'matched_exchange']
    preds = {k: ref.copy() for k in keys}
    inds = [names.index(k) for k in EVIDENCE]
    proposal, design = proposal_design(half[:, :5], half[:, inds])
    for f in np.unique(fold):
        t = (fold != f) & (fold % 2 == f % 2) & multi
        q = (fold == f) & multi
        # The reference's inner calibrator uses all other calibration bookings, including solo.
        all_t = (fold != f) & (fold % 2 == f % 2)
        tr_ref = np.zeros(len(y), bool)
        for h in np.unique(fold[t]):
            held = t & (fold == h)
            train = all_t & (fold != h)
            assert not np.isin(group[train], group[held | q]).any()
            tr_ref[held] = choose_reference(half, design, proposal, y, train, held)
        p = reference_score(half[t], tr_ref[t], alt)
        surprise, surprise_q = residuals(a, b, peer_a, peer_b, t, q, fold)
        idx = np.flatnonzero(q)
        strata = pd.DataFrame(dict(planet=raw.HomePlanet.fillna('?').to_numpy()[q],
                                   sleep=raw.CryoSleep.astype('string').fillna('?').to_numpy()[q]))
        rng = np.random.default_rng(319 + int(f))
        control = surprise_q.copy()
        for bucket in strata.groupby(list(strata)).indices.values():
            for i in bucket:
                pool = bucket[group[idx[bucket]] != group[idx[i]]]
                control[i] = surprise_q[rng.choice(pool)] if len(pool) else 0.
        for key in keys:
            if key == 'raw_peers':
                x, z = peer_a[t], peer_b[q]
            else:
                x, z = surprise, control if key == 'matched_exchange' else surprise_q
            scale = StandardScaler().fit(x)
            x, z = scale.transform(x), scale.transform(z)
            if key == 'unexpected_with_sleep_interaction':
                x = np.column_stack([x, x * earth[t, None]])
                z = np.column_stack([z, z * earth[q, None]])
            w = correction(x, y[t], p)
            preds[key][q] = logit(full_p[q].clip(.001, .999)) + z @ w >= 0
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    persistent = np.zeros(len(y), bool)
    persistent[dev] = hard_cohorts(dev, y[dev])[2]
    shared = ((full >= .5) != y[:, None]).all(1)
    result = {}
    for key, pred in preds.items():
        delta = (pred == y).astype(int) - (ref == y)
        assert np.all(delta[~multi] == 0)
        result[key] = dict(accuracy=float((pred == y).mean()), fixed=int((delta == 1).sum()),
                           harmed=int((delta == -1).sum()), earth_fixed=int((delta[earth] == 1).sum()),
                           earth_harmed=int((delta[earth] == -1).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
                           common_fixed=int(((delta == 1) & shared).sum()), persistent_fixed=int(((delta == 1) & persistent).sum()),
                           folds=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta, group))
    profile = dict(multi=int(multi.sum()), multi_errors=int((ref[multi] != y[multi]).sum()),
                   earth_sleep_multi=int(eligible.sum()), earth_sleep_multi_errors=int((ref[eligible] != y[eligible]).sum()))
    out = dict(profile=profile, policies=result)
    (ROOT / 'reports/nov23_matched_learning_curve/unexpected_peers.json').write_text(json.dumps(out, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    run()
