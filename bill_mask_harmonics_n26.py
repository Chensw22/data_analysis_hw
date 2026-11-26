"""Test which signed interactions between facility visibility flags transfer to unseen bookings."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from full_roster_bill_subset_route_n24 import BILLS, MASK
from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART / 'nov26_bill_mask_harmonics'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/bill_mask_harmonics.json'


def features(curves):
    # Each mask is one vertex of a five-dimensional cube. This basis separates the five
    # main effects from all 26 interactions, averaging over all backgrounds of other bills.
    parity = (MASK.astype(int) @ MASK.astype(int).T) % 2
    transform = 1 - 2 * parity
    odds = logit(curves.clip(.002, .998))
    coeff = odds @ transform / 32
    np.testing.assert_allclose(coeff @ transform, odds, atol=1e-12)
    orders = MASK.sum(1)
    base = np.column_stack([coeff[:, orders <= 1], odds[:, 0], odds[:, -1]])
    ordered = [coeff[:, orders == order] for order in range(2, 6)]
    anonymous = [np.sort(block, axis=1) for block in ordered]
    return dict(main=base, magnitude=np.column_stack([base,*anonymous]), named=np.column_stack([base,*ordered]))


def fit(x, base, y, t, q):
    mean, scale = x[t].mean(0), x[t].std(0).clip(.06)
    design = np.column_stack([np.ones(len(t)), (x[t]-mean)/scale])
    strength = np.full(design.shape[1], 24.)
    strength[0] = 5.
    def objective(w):
        z = base[t] + design @ w
        loss = np.logaddexp(0,z).sum() - y[t] @ z + .5*(w*strength)@w
        grad = design.T @ (expit(z)-y[t]) + strength*w
        return loss, grad
    opt = minimize(objective, np.zeros(len(strength)), jac=True, method='L-BFGS-B', options=dict(maxiter=250,ftol=1e-10))
    assert opt.success, opt.message
    return opt.x[0] + ((x[q]-mean)/scale) @ opt.x[1:]


def run():
    raw,y,g,fold = data()
    base,blind,known,fresh = cohort(raw,y,fold)
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    ids = np.r_[raw.PassengerId.to_numpy(str),test.PassengerId.to_numpy(str)]
    n = len(y)
    with np.load(ART/'nov26_latent_bill_visibility/aligned.npz') as cache:
        for k,v in [('passenger',ids),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(cache[k],v)
        curves = cache['curves']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as cache:
        np.testing.assert_array_equal(cache['passenger'],ids)
        anchor = cache['anchor']
        np.testing.assert_array_equal(base,anchor[:n])
        np.testing.assert_array_equal(cache['names'][-3:],['Mitra','TabICL fare','Nori fare'])
        native = logit(cache['scores'][:,-3:].mean(1).clip(.01,.99))
    opposite = ((curves-.5)*np.where(anchor,-1.,1.)[:,None] > 0).any(1)
    assert (opposite[:n]&blind&(base!=y)).sum()==94
    x = features(curves)
    groups = np.char.partition(ids,'_')[:,0]
    inner = partition(g,'mask_harmonics',4)
    scores = {}
    for name, design in x.items():
        correction = np.zeros(len(ids))
        for fold_id in np.unique(fold):
            for split in [0,1,2,3,None]:
                t = np.flatnonzero((fold==fold_id)&(inner!=split)) if split is not None else np.flatnonzero(fold==fold_id)
                q = np.flatnonzero((fold==fold_id)&(inner==split)) if split is not None else np.arange(n,len(ids))
                assert not np.isin(groups[t],groups[q]).any()
                effect = fit(design,native,y,t,q)
                correction[q] += effect if split is not None else effect/len(np.unique(fold))
        scores[name] = expit(native+correction)
    signed_offset = np.where(anchor,1.,-1.)*abs(native)
    flags = {name:(p>=.5)!=anchor for name,p in scores.items()}
    for before in ('main','magnitude'):
        increment = logit(scores['named']) - logit(scores[before])
        key = 'relative_to_'+before
        scores[key] = expit(signed_offset+increment)
        flags[key] = (scores[key]>=.5)!=anchor
    flags['independent'] = flags['relative_to_main']&flags['relative_to_magnitude']&flags['named']&opposite
    flags['adds_to_both'] = flags['named']&~flags['magnitude']&~flags['main']&opposite
    scopes = dict(known=known, fresh=known&fresh, common31=blind, common31_with_mask_dissent=blind&opposite[:n])
    report = dict(method='Eight grouped first-stage folds, within-fold four held booking splits; membership in the common-error subset and anchor direction are excluded from fitting.',
                  standalone={k:dict(accuracy=float(np.mean((v[:n]>=.5)==y)),logloss=float(log_loss(y,v[:n]))) for k,v in scores.items() if k in x},
                  scopes={k:dict(rows=int(m.sum()), errors=int((m&(base!=y)).sum())) for k,m in scopes.items()},changes={})
    for k,flag in flags.items():
        delta = np.where(base!=y,1,-1)*flag[:n]
        report['changes'][k] = dict(test=int(flag[n:].sum()), known_fold_net=[int(delta[known&(fold==f)].sum()) for f in np.unique(fold)],
                                    scopes={s:count(flag[:n],m,base!=y) for s,m in scopes.items()})
    # For each passenger, count the fewest facility identities that must be hidden
    # before the underlying subset model even disagrees with the incumbent.
    dissent = (curves[:n]-.5)*np.where(base,-1.,1.)[:,None] > 0
    removed = 5-MASK.sum(1)
    first = np.where(dissent,removed[None,:],6).min(1)
    report['fewest_hidden_facilities'] = {
        str(k):count(first==k,blind,base!=y) for k in range(6)}
    assert sum(v['fixed'] for v in report['fewest_hidden_facilities'].values()) == 94
    report['single_hidden_facility'] = {
        facility:count((first==1)&dissent[:,31-(1<<j)],blind,base!=y)
        for j,facility in enumerate(BILLS)}
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',passenger=ids,y=y,fold=fold,**scores)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['standalone'])
    for name,v in report['changes'].items():
        print(name,v['test'],v['scopes'])


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
