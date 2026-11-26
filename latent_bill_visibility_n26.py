"""Learn the chance of relying on each facility from all 32 masked-bill predictions on held-out bookings."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from full_roster_bill_subset_route_n24 import BILLS, MASK, predict
from full_roster_booking_validation_n23 import ROOT, data
from full_roster_two_view_reliability_n24 import design
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART / 'nov26_latent_bill_visibility'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/latent_bill_visibility.json'


def manifest(frame, groups):
    v = frame[BILLS]
    values = np.log1p(v.fillna(0).to_numpy()) / 8
    common = np.column_stack([np.ones(len(v)), frame.CryoSleep.eq(True), frame.CryoSleep.isna(),
                              np.log1p(v.sum(axis=1))/8, v.gt(0).sum(axis=1)/5,
                              (frame.Age.fillna(30).to_numpy()-30)/30,
                              pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy() > 1])
    return np.dstack([np.column_stack([common, values[:, j], v.iloc[:, j].isna(), v.iloc[:, j].gt(0)])
                      for j in range(5)])


def mixture(weights, x, curve, relative_offset):
    d = x.shape[1]
    theta = weights.reshape(d, 5)
    gates = np.einsum('ndb,db->nb', x, theta)
    mass = softmax(gates @ MASK.T, axis=1)
    p = np.sum(mass * curve, axis=1).clip(1e-6, 1-1e-6)
    if relative_offset is None:
        return p, mass, p
    return expit(relative_offset + logit(p) - logit(curve[:, -1].clip(.001,.999))), mass, p


def learn(x, curve, y, offset):
    # Start with equal inclusion odds. A quadratic penalty shrinks conditional changes to this baseline.
    regularizer = np.full((x.shape[1], 5), 2.)
    regularizer[0] = .3
    def objective(weights):
        p, mass, marginal = mixture(weights, x, curve, offset)
        centered = mass * (curve - marginal[:, None])
        derivative = (centered @ MASK) * ((p-y) / (marginal*(1-marginal)))[:, None]
        grad = np.einsum('ndb,nb->db', x, derivative) + regularizer * weights.reshape(regularizer.shape)
        loss = -(y*np.log(p.clip(1e-12,1))+(1-y)*np.log((1-p).clip(1e-12,1))).sum()
        return loss + .5*np.sum(regularizer.ravel()*weights**2), grad.ravel()
    result = minimize(objective, np.zeros(regularizer.size), method='L-BFGS-B', jac=True,
                      options=dict(maxiter=250, ftol=1e-10))
    assert result.success, result.message
    return result.x


def run():
    raw, y, groups, fold = data()
    base, blind, known, fresh = cohort(raw, y, fold)
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns)
    frame = pd.concat([raw, test], ignore_index=True)
    all_groups = frame.PassengerId.str.split('_').str[0].to_numpy()
    n = len(y)
    train, submit = np.arange(n), np.arange(n,len(frame))
    assert not np.isin(groups, all_groups[submit]).any()
    folds = np.unique(fold)
    curves = np.empty((len(frame), 32))
    inputs = (design(frame, all_groups)[0], np.log1p(frame[BILLS]).fillna(-1).to_numpy())
    for k in folds:
        t, q = train[fold != k], train[fold == k]
        with np.load(ART/f'nov24_bill_subset_route/full_{k}.npz') as archive:
            for key, expected in [('q',q),('t',t),('truth',y[t])]:
                np.testing.assert_array_equal(archive[key],expected)
            curves[q] = archive['p']
            if k == 0:
                # Refit with the test roster present to check that training features and scores remain unchanged.
                np.testing.assert_allclose(predict(inputs,y,all_groups,t,q),curves[q],rtol=0,atol=1e-8)
    curves[submit] = predict(inputs, y, all_groups, train, submit)
    assert np.isfinite(curves).all()
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'], frame.PassengerId)
        np.testing.assert_array_equal(z['anchor'][:n], base)
        np.testing.assert_array_equal(z['names'][-3:], ['Mitra','TabICL fare','Nori fare'])
        anchor = z['anchor']
        confidence = abs(logit(z['scores'][:,-3:].mean(1).clip(.01,.99)))
        offset = np.where(anchor, confidence, -confidence)
    x = manifest(frame, all_groups)
    modes = ('global','personal','anchored_global','anchored_personal')
    scores = {key:np.full(len(frame), np.nan) for key in modes}
    test_scores = {key:[] for key in modes}
    inner = partition(groups,'fee_visibility_held',4)
    for k in folds:
        for split in [0,1,2,3,None]:
            t = train[(fold==k) & known & ((inner!=split) if split is not None else True)]
            q = train[(fold==k) & (inner==split)] if split is not None else submit
            assert not np.isin(all_groups[t],all_groups[q]).any()
            for key in modes:
                a = x if 'personal' in key else x[:, :1]
                off = offset if key.startswith('anchored') else None
                beta = learn(a[t], curves[t], y[t], None if off is None else off[t])
                p = mixture(beta, a[q], curves[q], None if off is None else off[q])[0]
                if split is None:
                    test_scores[key].append(p)
                else:
                    scores[key][q] = p
    for key in modes:
        scores[key][submit] = np.mean(test_scores[key], axis=0)
    scores['complete'] = curves[:, -1]
    scores['equal_masks'] = curves.mean(1)
    flags = {k:(p >= .5) != anchor for k,p in scores.items()}
    flags['personal_only'] = flags['anchored_personal'] & ~flags['anchored_global']
    flags['confirmed'] = flags['anchored_personal'] & flags['personal'] & ~flags['complete']
    report = dict(accuracy={k:float(((p[:n]>=.5)==y).mean()) for k,p in scores.items()},rules={})
    for k, edit in flags.items():
        delta = edit[:n] * np.where(base!=y,1,-1)
        report['rules'][k] = dict(test_changes=int(edit[n:].sum()),
            by_fold=[int(delta[known & (fold==j)].sum()) for j in folds],
            scopes={scope:count(edit[:n],m,base!=y) for scope,m in dict(all=known,blind=blind,fresh=known&fresh).items()})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=fold,curves=curves,**scores)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'],[(k,v['scopes']['all'],v['scopes']['blind']) for k,v in report['rules'].items()])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
