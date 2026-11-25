"""Estimate outcome prevalence from other unlabeled bookings, then test the resulting decisions."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs, summary
from full_roster_booking_validation_n23 import OUT, ROOT, data
from price_implied_deck_n26 import FEES


def rates(source_rate, calibrated, group):
    # For a calibrated posterior s, the two class-conditional density ratios
    # against the source marginal are s/pi and (1-s)/(1-pi).
    grid = expit(logit(source_rate)+np.arange(-200,201)*.015)
    ll = np.log((calibrated[:,None]/source_rate)*grid +
                ((1-calibrated[:,None])/(1-source_rate))*(1-grid))
    # Fifteen source-distribution pseudo-observations regularize small target batches.
    penalty = 15*(source_rate*np.log(grid)+(1-source_rate)*np.log1p(-grid))
    unique, ids = np.unique(group,return_inverse=True)
    by_order = np.zeros((len(unique),len(grid)))
    np.add.at(by_order,ids,ll)
    scores = ll.sum(0)+penalty-by_order
    rate = grid[np.argmax(scores,axis=1)]
    if len(unique)==1:
        np.testing.assert_allclose(rate,source_rate)
    return rate[ids]


def crossfit(matrix,y,g,fold,cell,geographic):
    n = len(y)
    individual = np.full((n,3),np.nan)
    shifted = individual.copy()
    diagnostics = []
    for k in np.unique(fold):
        rows = np.flatnonzero(fold==k)
        if geographic:
            ordered = np.array_split(np.unique(g[rows]),4)
            held = [rows[np.isin(g[rows],batch)] for batch in ordered]
        else:
            held = [rows[q] for _,q in GroupKFold(4,shuffle=True,random_state=6215).split(rows,groups=g[rows])]
        for q in held:
            t = np.setdiff1d(rows,q)
            assert len(q)>0 and not np.isin(g[q],g[t]).any()
            for c in np.unique(cell[q]):
                ask = q[cell[q]==c]
                bank = t[cell[t]==c]
                if len(bank)<30 or len(np.unique(y[bank]))<2:
                    bank = t
                source_rate = y[bank].mean()
                for j in range(matrix.shape[1]):
                    col = matrix[:,j:j+1]
                    cal = LogisticRegression(C=.5,max_iter=300).fit(col[bank],y[bank])
                    p = cal.predict_proba(col[ask])[:,1]
                    local = rates(source_rate,p,g[ask])
                    individual[ask,j] = p
                    shifted[ask,j] = expit(logit(p)+logit(local)-logit(source_rate))
                    if j==0:
                        diagnostics.append((len(ask),y[ask].mean(),source_rate,p.mean(),local.mean()))
    assert np.isfinite(shifted).all() and np.isfinite(individual).all()
    weight,truth,prior,mean,updated = np.array(diagnostics).T
    return individual,shifted, {key:float(np.average(abs(value-truth),weights=weight))
                                for key,value in [('source_rate',prior),('individual_mean',mean),
                                                  ('other_unlabelled_orders',updated)]}


def run():
    raw,y,g,f = data()
    scores = []
    for model in ('mitra96','tabicl','cat'):
        with np.load(OUT/(model+'.npz')) as z:
            for key,expected in [('y',y),('fold',f),('passenger',raw.PassengerId.to_numpy(str))]:
                np.testing.assert_array_equal(z[key],expected)
            scores.append(logit(z['p'].clip(.003,.997)))
    matrix = np.column_stack(scores)
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    cell = 2*raw.CryoSleep.eq(True).to_numpy().astype(int)+paid
    _, cur, _ = inputs()
    parent = cur['parent']
    report = {}
    for kind,ordered in [('random_bookings',False),('contiguous_booking_ids',True)]:
        before,after,err = crossfit(matrix,y,g,f,cell,ordered)
        old = before>=.5
        new = after>=.5
        opposite = new!=parent[:,None]
        changes = opposite & (old!=new)
        actions = dict(individual=old[:,0]!=parent,adjusted=opposite[:,0],mixture_change=changes[:,0],
                       two_family_changes=changes.sum(1)>=2,
                       newly_unanimous=opposite.all(1)&~(old!=parent[:,None]).all(1))
        report[kind] = dict(prevalence_mean_absolute_error=err,scopes={})
        for name,mask in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
            report[kind]['scopes'][name] = {n:summary(v,parent!=y,mask,f) for n,v in actions.items()}
        print(kind,err,{m:r['net'] for m,r in report[kind]['scopes']['all'].items()},flush=True)
    report['scope'] = ('Eight base folds hold out full bookings. Secondary calibration labels use only other orders '
        'inside the same base fold. Query prevalence estimates only use other unlabeled bookings in the same '
        'query batch and cryosleep/positive-bill cell. Geographic batches are contiguous booking identifiers.')
    (ROOT/'reports/nov23_matched_learning_curve/unlabeled_order_prior_shift.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
