"""Check whether independent frozen voter support transfers the same bill update between baselines."""
import json

import numpy as np
from scipy.special import logit

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats


def rules(base, anchor, rivals, delta):
    challenger = logit(base.clip(.0001,.9999))+.25*delta >= 0
    eligible = ((base >= .5) == anchor) & (challenger != anchor)
    support = ((rivals >= .5) == challenger[:,None]).sum(1)
    return {name: anchor ^ (eligible & allowed) for name,allowed in [
        ('everyone', support >= 0), ('one_or_more', support >= 1), ('both', support == 2),
        ('no_support', support == 0)]}, eligible


def run():
    _, oy, og, rows, ofold, mapped, champ, _, cold, _, _, old, _ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    indices = [names.index(name) for name in ('mitra96','v44_without_name','tabicl')]
    scored = np.full((len(y)+len(oy),3),np.nan)
    for j,seed in enumerate((860,861,862)):
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            oq = np.flatnonzero(mapped == f)
            paired = np.r_[q,len(y)+oq]
            t = np.flatnonzero((fold%2 == f%2) & (fold != f))
            with np.load(ROOT/f'artifacts/nov24_service_odds_transfer/{seed}_{f}.npz') as z:
                for k,v in [('t',t),('q',paired),('y',y[t]),('priors',np.r_[half[t,indices[0]],full[q,indices[0]],old[oq,0]])]:
                    np.testing.assert_array_equal(z[k],v)
                scored[paired,j] = z['ll'][:,1]-z['ll'][:,0]
    assert np.isfinite(scored).all()
    before, old_changed = rules(old[:,0],champ,old[:,1:],scored[len(y):].mean(1))
    current, new_changed = rules(full[:,indices[0]],ref,full[:,indices[1:]],scored[:len(y)].mean(1))
    unseen = ~np.isin(g,og)
    common = ((full >= .5) != y[:,None]).all(1)
    report = dict(paired=dict(both=int((old_changed & new_changed[rows]).sum()),
                              historical_only=int((old_changed & ~new_changed[rows]).sum()),
                              grouped_only=int((~old_changed & new_changed[rows]).sum())),
                  rules={})
    for k,p in current.items():
        report['rules'][k] = dict(historical=stats(before[k],champ,oy,og,ofold),
                                  full=stats(p,ref,y,g,fold),
                                  historical_no_training_peer=stats(before[k][cold],champ[cold],oy[cold],og[cold],ofold[cold]),
                                  unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                                  common_fixed=int((common & (p == y)).sum()))
        print(k,[report['rules'][k][s]['net'] for s in ['historical','full','unseen']])
    path = ROOT/'reports/nov23_matched_learning_curve/service_odds_confirmation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
