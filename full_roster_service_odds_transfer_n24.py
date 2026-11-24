"""Replay the identical bill decoder on the historical and grouped incumbent probabilities."""
import json

import numpy as np
import torch
from scipy.special import logit

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_service_conditional_odds_n24 import fit
from full_roster_zero_truncated_bills_n24 import inputs


def run():
    _, oy, og, rows, ofold, projected, champ, _, _, _, _, old, _ = aligned()
    raw, y, group, fold = data()
    names, half, full = banks(raw,y,fold)
    ref = latest_reference(y,group,fold,names,half,full)
    k = names.index('mitra96')
    base, historical = full[:,k], old[:,0]
    x, bills, _ = inputs(raw)
    # Paired views differ only in query base probability. No appended row enters fitting.
    xx = np.vstack([x,x[rows]])
    bb = np.vstack([bills,bills[rows]])
    yy = np.r_[y,oy]
    seeds = [860,861,862]
    scored = np.full((len(yy),len(seeds)),np.nan)
    dest = ROOT/'artifacts/nov24_service_odds_transfer'
    dest.mkdir(exist_ok=True)
    for j,seed in enumerate(seeds):
        for f in np.unique(fold):
            t = np.flatnonzero((fold%2 == f%2) & (fold != f))
            q = np.flatnonzero(fold == f)
            oq = np.flatnonzero(projected == f)
            paired = np.r_[q, len(y)+oq]
            priors = np.r_[half[:,k],historical]
            priors[q] = base[q]
            assert not np.isin(group[t],group[q]).any() and not np.isin(group[t],og[oq]).any()
            path = dest/f'{seed}_{f}.npz'
            if not path.exists():
                ll = fit(xx,bb,yy,t,paired,priors,'conditional_hybrid',seed)
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=t,q=paired,y=yy[t],priors=priors[np.r_[t,paired]],ll=ll)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,value in [('t',t),('q',paired),('y',yy[t]),('priors',priors[np.r_[t,paired]])]:
                    np.testing.assert_array_equal(z[key],value)
                ll = z['ll']
            if j == 0:
                with np.load(ROOT/f'artifacts/nov24_service_conditional_odds/conditional_hybrid_{f}.npz') as z:
                    np.testing.assert_allclose(ll[:len(q)], z['density'], atol=1e-5, rtol=1e-6)
            scored[paired,j] = ll[:,1]-ll[:,0]
        print('seed',seed,flush=True)
    assert np.isfinite(scored).all()
    shared = ((full >= .5) != y[:,None]).all(1)
    hard = shared[rows] & (champ != oy)
    unseen = ~np.isin(group,og)
    report = {}
    for key,d in zip([*map(str,seeds),'mean'],[*scored.T,scored.mean(1)],strict=True):
        current = np.where((base >= .5) == ref, logit(base.clip(.0001,.9999))+.25*d[:len(y)] >= 0,ref)
        previous = np.where((historical >= .5) == champ,logit(historical.clip(.0001,.9999))+.25*d[len(y):] >= 0,champ)
        report[key] = dict(full=stats(current,ref,y,group,fold),
            historical=stats(previous,champ,oy,og,ofold),
            unseen=stats(current[unseen],ref[unseen],y[unseen],group[unseen],fold[unseen]),
            common_correct=int((shared & (current == y)).sum()),
            persistent_both=int((hard & (previous == oy) & (current[rows] == oy)).sum()))
        print(key,[report[key][s]['net'] for s in ('historical','full','unseen')],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/service_odds_transfer.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    run()
