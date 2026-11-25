"""Learn a correction by maximizing expected accuracy under uncertain decision margins."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_direct_edit_utility_n24 import features
from full_roster_edit_resampling_n24 import inputs, test_inputs
from full_roster_disagreement_calibration_n24 import stats
from full_roster_conditional_pool_n23 import ROOT


METHODS = ('log_loss', 'fixed_width', 'opinion_width')


def estimates(x, p, y, t, query):
    mean, scale = x[t].mean(0), x[t].std(0).clip(.15)
    train, ask = [np.column_stack([np.ones(len(v)), ((v-mean)/scale).clip(-5, 5)])
                  for v in (x[t], query)]
    anchor = logit(p[t].clip(.003, .997))
    sign = 2*y[t].astype(float)-1
    # First two original features are other-model logits minus the anchor logit.
    width = np.sqrt(.35**2 + np.mean(x[t, :2]**2, axis=1)).clip(.35, 2.)
    result = []
    for method in METHODS:
        def objective(w):
            margin = anchor + train@w
            if method == 'log_loss':
                loss = np.logaddexp(0, -sign*margin)
                derivative = -sign*expit(-sign*margin)
            else:
                bandwidth = width if method == 'opinion_width' else .7
                loss = expit(-sign*margin/bandwidth)
                derivative = -sign*loss*(1-loss)/bandwidth
            return loss.mean()+.01*(w@w), train.T@derivative/len(t)+.02*w
        opt = minimize(objective, np.zeros(train.shape[1]), jac=True, method='L-BFGS-B')
        if not opt.success:
            raise RuntimeError(opt.message)
        result.append(ask@opt.x)
    return np.column_stack(result)


def decide(p, shift, ref):
    return np.where((p >= .5) == ref, logit(p.clip(.003,.997))+shift >= 0, ref)


def run():
    raw,y,g,f,ref,p,ask,sets = inputs()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    test,query,pt,_,official,_ = test_inputs(g)
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    hx = features(raw.iloc[rows],size[rows],old)
    full = np.full((len(y),len(sets),len(METHODS)), np.nan)
    test_shifts = []
    for j,(panel,prob,x) in enumerate(sets):
        for level in np.unique(f):
            q = np.flatnonzero(f == level)
            t = np.flatnonzero((panel == panel[q[0]]) & (f != level))
            assert not np.isin(g[t],g[q]).any()
            shifts = estimates(x,prob,y,t,np.vstack([ask[q],query]))
            full[q,j] = shifts[:len(q)]
            test_shifts.append(shifts[len(q):])
    hist = np.full((len(oy),len(METHODS)), np.nan)
    for level in np.unique(of):
        ids = np.flatnonzero(of == level)
        for train, test_ix in GroupKFold(4,shuffle=True,random_state=633).split(ids,groups=og[ids]):
            t,q = ids[train],ids[test_ix]
            t = t[(old[t,0] >= .5) == champ[t]]
            assert not np.isin(og[t],og[q]).any()
            hist[q] = estimates(hx,old[:,0],oy,t,hx[q])
    assert np.isfinite(hist).all() and np.isfinite(full).all()
    newcomer = ~np.isin(g,og)
    shared = (ref[rows]!=oy) & (champ!=oy)
    report = {}
    for j,k in enumerate(METHODS):
        current = decide(p,full.mean(1)[:,j],ref)
        past = decide(old[:,0],hist[:,j],champ)
        query_vote = decide(pt,np.mean(test_shifts,axis=0)[:,j],official)
        values = dict(full=stats(current,ref,y,g,f),historical=stats(past,champ,oy,og,of),
                      new_orders=stats(current[newcomer],ref[newcomer],y[newcomer],g[newcomer],f[newcomer]),
                      repeated_fixed=int((shared & (current[rows]==oy) & (past==oy)).sum()),
                      repeated_errors=int(shared.sum()),test_edits=int((query_vote!=official).sum()),
                      half_panel_nets=[int((decide(p,full[:,s,j],ref)==y).sum()-(ref==y).sum())
                                       for s in range(len(sets))])
        report[k] = values
        print(k,[(n,values[n]['corrected'],values[n]['harmed']) for n in ('full','historical','new_orders')],
              values['half_panel_nets'],values['test_edits'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/uncertain_decisions.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
