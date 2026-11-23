"""Match calibration bookings and labels exactly when comparing half and seven-fold scores."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_direct_reference_risk_n23 import choose_reference
from full_roster_same_model_calibration_n23 import learn
from full_roster_unexpected_peers_n23 import reference_score


def design(prob, route, earth, mode):
    z = logit(prob.clip(.001, .999))
    x = np.column_stack([np.ones(len(z)), z])
    if mode == 'global':
        return x
    region = np.column_stack([route, route * earth[:, None]])
    features = [x, region, region*z[:,None]]
    if mode == 'low_score':
        features.append(region*np.maximum(0, logit(.3)-z)[:,None])
    return np.column_stack(features)


def run():
    raw, y, group, fold = data()
    names, half, full = banks(raw, y, fold)
    ref = latest_reference(y, group, fold, names, half, full)
    p = reference_score(full, ref, names.index('balanced_full'))
    evidence = half[:, [names.index(k) for k in EVIDENCE]]
    proposal, prior_design = proposal_design(half[:, :5], evidence)
    half_p = np.zeros(len(y))
    for f in np.unique(fold):
        q = fold == f
        t = (fold != f) & (fold % 2 == f % 2)
        assert not np.isin(group[t], group[q]).any()
        chosen = choose_reference(half, prior_design, proposal, y, t, q)
        half_p[q] = reference_score(half[q], chosen, names.index('balanced_full'))
    side = raw.Cabin.str.split('/').str[2].fillna('?')
    route = raw.Destination.fillna('?') + '/' + side
    route_matrix = pd.get_dummies(route, dtype=float).to_numpy()
    earth = (raw.HomePlanet.eq('Earth') & raw.CryoSleep.eq(True)).to_numpy()
    focus = earth & raw.Destination.eq('TRAPPIST-1e').to_numpy()
    report = {'observed': {}, 'results': {}}
    for s in ('P', 'S'):
        selected = focus & side.eq(s).to_numpy()
        lo = selected & (p <= .3)
        report['observed'][s] = dict(rows=int(selected.sum()), wrong=int((ref[selected] != y[selected]).sum()),
                                      low_rows=int(lo.sum()), low_transported=int(y[lo].sum()),
                                      low_expected=float(p[lo].sum()))
    for mode in ['global','route_side','low_score']:
        a = design(half_p, route_matrix, earth, mode)
        b = design(p, route_matrix, earth, mode)
        outputs = {k: np.full(len(y), np.nan) for k in ('smaller','same')}
        for f in np.unique(fold):
            outer = np.flatnonzero(fold == f)
            split = GroupKFold(4, shuffle=True, random_state=462)
            for first, second in split.split(outer,groups=group[outer]):
                donors, q = outer[first], outer[second]
                assert not np.isin(group[donors], group[q]).any()
                for k, x, score in [('smaller', a, half_p), ('same', b, p)]:
                    w = learn(x[donors], y[donors], logit(score[donors].clip(.001,.999)), np.zeros(x.shape[1]))
                    outputs[k][q] = expit(logit(p[q].clip(.001,.999)) + b[q] @ w)
        for k, prob in outputs.items():
            assert np.isfinite(prob).all()
            delta = ((prob >= .5) == y).astype(int) - (ref == y)
            brier = (prob-y)**2 - (p-y)**2
            report['results'][mode+'_'+k] = dict(accuracy=float(((prob >= .5) == y).mean()), fixed=int((delta==1).sum()),
                harmed=int((delta==-1).sum()), brier_delta=float(brier.mean()), focus_brier_delta=float(brier[focus].mean()),
                focus_fixed=int((delta[focus]==1).sum()), focus_harmed=int((delta[focus]==-1).sum()),
                low_focus_net=int(delta[focus & (p <= .3)].sum()), fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
                ci_pp=interval(delta, group))
    (ROOT/'reports/nov23_matched_learning_curve/route_side_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    run()
