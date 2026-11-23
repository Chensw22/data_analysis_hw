"""Calibrate on independent bookings predicted by the same fitted base models."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_direct_reference_risk_n23 import choose_reference
from full_roster_unexpected_peers_n23 import reference_score


def design(scores, decisions, names, method):
    alt = names.index('balanced_full')
    p = reference_score(scores, decisions, alt)
    offset = logit(p.clip(.001, .999))
    change = decisions != (scores[:, 4] >= .5)
    values = [np.ones(len(scores)), offset, change.astype(float), offset * change]
    if method == 'expert_scores':
        selected = [0, 1, 2, 3, 4, *(names.index(k) for k in EVIDENCE)]
        values.extend(logit(scores[:, selected].clip(.001, .999)).T)
    return np.column_stack(values), offset


def learn(x, y, offset, center):
    def objective(w):
        z = offset + x @ w
        deviation = w - center
        return (np.logaddexp(0, z).sum() - y @ z + 10*(deviation @ deviation),
                x.T @ (expit(z) - y) + 20*deviation)
    fit = minimize(objective, center, jac=True, method='L-BFGS-B', options={'maxiter':500})
    assert fit.success, fit.message
    return fit.x


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    proposal, previous = proposal_design(half[:, :5], half[:, [names.index(k) for k in EVIDENCE]])
    shared = ((full >= .5) != y[:, None]).all(1)
    earth = raw.HomePlanet.eq('Earth').to_numpy() & raw.CryoSleep.eq(True).to_numpy()
    experiments = {}
    for method in ['branch', 'expert_scores']:
        xfull, basefull = design(full, ref, names, method)
        probabilities = {key: np.full(len(y), np.nan) for key in
                         ['small_models', 'same_models', 'small_then_same']}
        for f in np.unique(fold):
            outer = np.flatnonzero(fold == f)
            t = (fold != f) & (fold % 2 == f % 2)
            rt = np.zeros(len(y), bool)
            for h in np.unique(fold[t]):
                train, held = t & (fold != h), t & (fold == h)
                assert not np.isin(group[train], group[held | (fold == f)]).any()
                rt[held] = choose_reference(half, previous, proposal, y, train, held)
            xhalf, basehalf = design(half[t], rt[t], names, method)
            zero = np.zeros(xhalf.shape[1])
            small = learn(xhalf, y[t], basehalf, zero)
            probabilities['small_models'][outer] = expit(basefull[outer]+xfull[outer]@small)
            # All four partitions come from the same withheld fold. Its labels
            # were excluded from the base models and from its original 124 gate.
            split = GroupKFold(4, shuffle=True, random_state=462)
            for a, b in split.split(outer, groups=group[outer]):
                donors, q = outer[a], outer[b]
                assert not np.isin(group[donors], group[q]).any()
                for mode, prior in [('same_models', zero), ('small_then_same', small)]:
                    w = learn(xfull[donors], y[donors], basefull[donors], prior)
                    probabilities[mode][q] = expit(basefull[q] + xfull[q] @ w)
        with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
            dev = np.sort(z['development'])
        for mode, p in probabilities.items():
            assert np.isfinite(p).all()
            delta = ((p >= .5) == y).astype(int) - (ref == y)
            experiments[method+'_'+mode] = dict(accuracy=float(((p >= .5) == y).mean()),
                fixed=int((delta == 1).sum()), harmed=int((delta == -1).sum()),
                loss=float(np.logaddexp(0, (1-2*y.astype(int))*logit(p.clip(1e-10, 1-1e-10))).mean()),
                earth_fixed=int((delta[earth] == 1).sum()), earth_harmed=int((delta[earth] == -1).sum()),
                common_fixed=int((delta[shared] == 1).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
                fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)], ci_pp=interval(delta, group))
    (ROOT / 'reports/nov23_matched_learning_curve/same_model_calibration.json').write_text(
        json.dumps(experiments, indent=2)+'\n')
    for k,v in experiments.items():
        print(k, round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['fold_net'],v['additional_net'])


if __name__ == '__main__':
    run()
