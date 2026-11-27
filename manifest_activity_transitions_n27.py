"""Estimate whether spending shifts both registration outcomes in the same or opposite directions."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import inputs

REPORT = ROOT / 'reports/nov23_matched_learning_curve/manifest_activity_transitions.json'


def basis(series, knots, scale):
    values = series.to_numpy(float)
    v = np.nan_to_num(values) / scale
    return [np.isnan(values), v, *[np.maximum(v - knot / scale, 0.) for knot in knots]]


def design(frame):
    cabin = frame.Cabin.str.split('/', expand=True)
    nominal = frame[['HomePlanet', 'Destination', 'VIP']].astype('string').fillna('?').copy()
    nominal['deck_side'] = (cabin[0] + cabin[2]).fillna('?')
    booking = frame.PassengerId.str.split('_').str[0]
    size = booking.map(booking.value_counts())
    position = basis(pd.to_numeric(cabin[1]), [250, 650, 1100, 1600], 1000)
    reg = np.column_stack([np.ones(len(frame)), pd.get_dummies(nominal).to_numpy(float),
                           *basis(frame.Age, [8, 18, 35, 55], 30), np.log(size), *position,
                           *[v * cabin[2].eq('P').to_numpy() for v in position]])
    amounts = frame[FEES]
    active = pd.get_dummies(frame.CryoSleep.astype('string').fillna('?')).to_numpy(float)
    consumption = [np.ones(len(frame)), active, amounts.gt(0).sum(axis=1),
                   *basis(np.log1p(amounts.sum(axis=1, min_count=5)), [5, 7, 8], 6)]
    for col in FEES:
        consumption.extend([amounts[col].gt(0), *basis(np.log1p(amounts[col]), [3, 5, 7, 8], 6)])
    return reg.astype(float), np.column_stack(consumption).astype(float)


def fit(reg, act, y, train, kind):
    a, b, labels = reg[train], act[train], y[train]
    width = reg.shape[1]
    symmetric = kind == 'symmetric'
    blocks = 2 if kind == 'asymmetric' else 1
    ridge = 3.
    def forward(x, z, w):
        u = expit(x @ w[:width])
        if kind == 'additive':
            p = expit(x @ w[:width] + z @ w[width:])
            return p, [p*(1-p), p*(1-p)]
        lo = expit(z @ w[width:width+b.shape[1]])
        hi = 1-lo if symmetric else expit(z @ w[width+b.shape[1]:])
        p = lo*(1-u) + hi*u
        du = u*(1-u)*(hi-lo)
        if symmetric:
            return p, [du, lo*(1-lo)*(1-2*u)]
        return p, [du, lo*(1-lo)*(1-u), hi*(1-hi)*u]
    def objective(w):
        p, derivatives = forward(a, b, w)
        p = p.clip(1e-12, 1-1e-12)
        d = (p-labels) / (p*(1-p))
        grad = np.concatenate([arr.T @ (d*deriv) for arr,deriv in zip([a]+[b]*blocks, derivatives)])
        return -np.where(labels,np.log(p),np.log1p(-p)).sum() + ridge*(w@w)/2, grad + ridge*w
    initial = np.zeros(width + blocks*b.shape[1])
    if kind != 'additive':
        initial[width] = -2.
        if not symmetric:
            initial[width+b.shape[1]] = 2.
    res = minimize(objective, initial, method='L-BFGS-B', jac=True, options={'maxiter':600,'ftol':1e-10})
    if not res.success:
        raise RuntimeError(res.message)
    return lambda q:forward(reg[q],act[q],res.x)[0]


def run():
    raw,frame,y,groups,fold,first,random,ordered,wide,_ = inputs()
    n = len(y)
    _,_,known,fresh = cohort(raw,y,fold)
    reg,act = design(frame)
    all_groups = frame.PassengerId.str.split('_').str[0].to_numpy()
    contiguous = np.zeros(n,int)
    for k,orders in enumerate(np.array_split(np.unique(groups),8)):
        contiguous[np.isin(groups,orders)] = k
    # Anchor and agreement cohort stay fixed; only the new classifiers are independently retrained.
    anchor = first ^ random
    agree = (wide[:n] >= .5).astype(bool) == anchor[:n,None]
    unanimous = known & agree.all(1)
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    scopes = dict(all=known, new_orders=known&fresh, unanimous=unanimous,
                  awake_paid=unanimous&raw.CryoSleep.eq(False).to_numpy()&paid,
                  sleep_zero=unanimous&raw.CryoSleep.eq(True).to_numpy()&~paid)
    wrong = anchor[:n] != y
    assert int((unanimous & wrong).sum()) == 862
    report, saved = {}, {}
    odds = lambda p: logit(p.clip(.003,.997))
    margin = np.where(anchor,1.,-1.) * abs(odds(np.median(wide[:,1:4],axis=1)))
    for tag,external in [('random',fold),('ordered',contiguous)]:
        predictions = {}
        for kind in ('additive','symmetric','asymmetric'):
            scores = np.empty(len(frame))
            for k in range(9):
                train = np.flatnonzero(external!=k)
                q = np.flatnonzero(external==k) if k<8 else np.arange(n,len(frame))
                assert not np.isin(all_groups[train],all_groups[q]).any()
                scores[q] = fit(reg,act,y,train,kind)(q)
            predictions[kind] = scores
            saved[tag+'_'+kind] = scores
        propose = {k:(p>=.5)!=anchor for k,p in predictions.items()}
        fixed = propose['asymmetric'] & ~propose['additive']
        excess = odds(predictions['asymmetric']) - odds(predictions['additive'])
        shift = ((margin+excess>=0)!=anchor) & propose['asymmetric']
        rules = dict(asymmetric=propose['asymmetric'],new_vs_additive=fixed,
                     new_vs_both=fixed&~propose['symmetric'],incremental=shift)
        report[tag] = dict(standalone={k:dict(accuracy=float(np.mean((p[:n]>=.5)==y)),log_loss=float(log_loss(y,p[:n])))
                                            for k,p in predictions.items()},rules={})
        for k,flag in rules.items():
            delta = np.where(wrong,1,-1)*flag[:n]
            report[tag]['rules'][k] = dict(scopes={key:count(flag[:n],mask,wrong) for key,mask in scopes.items()},
                fold_net=[int(delta[known&(external==j)].sum()) for j in range(8)],test_changes=int(flag[n:].sum()))
            saved[tag+'_'+k+'_edit'] = flag
    # Both protocols use the same full train for their test outputs.
    for k in predictions:
        np.testing.assert_allclose(saved['ordered_'+k][n:], saved['random_'+k][n:])
    report['repeat'] = {}
    for k in rules:
        flag=saved['random_'+k+'_edit'] & saved['ordered_'+k+'_edit']
        report['repeat'][k]={key:count(flag[:n],mask,wrong) for key,mask in scopes.items()}
    out=ART/'nov27_manifest_activity_transitions'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=fold,ordered_fold=contiguous,**saved)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['random']['standalone'],report['ordered']['standalone'],report['repeat'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
