"""Refit the direct-edit utility on a second set of fully retrained half-roster base scores."""
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_direct_edit_utility_n24 import features,fit
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data


def margins(x,query,p,y,g,t,q):
    mean,sd = x[t].mean(0),x[t].std(0).clip(.15)
    train,ask = [np.column_stack([np.ones(len(v)),((v-mean)/sd).clip(-5,5)]) for v in (x[t],query[q])]
    offset = logit(p[t].clip(.003,.997))
    return np.column_stack([ask@fit(train,offset,y[t],g[t],robust) for robust in (False,True)])


def run():
    raw,y,g,fold,names,h,full,ref,_,cols,rotated = panel_data()
    _,_,og,*_ = aligned()
    indices = [names.index(k) for k in ('mitra96','v44_without_name','tabicl')]
    half,rot,large = h[:,indices],rotated[:,[cols.index(i) for i in indices]],full[:,indices]
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    a,b,qx = [features(raw,size,s) for s in (half,rot,large)]
    shifts = []
    for panel,prob,x in [(fold%2,half,a),((fold//2)%2,rot,b)]:
        scores = np.full((len(y),2),np.nan)
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((panel==panel[q[0]]) & (fold!=f))
            assert not np.isin(g[t],g[q]).any() and np.all(panel[q]==panel[q[0]])
            scores[q] = margins(x,qx,prob[:,0],y,g,t,q)
        assert np.isfinite(scores).all()
        shifts.append(scores)
    before = json.loads((ROOT/'reports/nov23_matched_learning_curve/direct_edit_utility.json').read_text())
    p = large[:,0]
    anchor = logit(p.clip(.003,.997))
    free = (p>=.5)==ref
    methods = {k:np.where(free[:,None],(anchor[:,None]+s)>=0,ref[:,None]) for k,s in
               [('original',shifts[0]),('rotated',shifts[1]),('averaged',np.mean(shifts,axis=0))]}
    methods['both_flip'] = np.where(methods['original']==methods['rotated'],methods['original'],ref[:,None])
    result = {}
    new = ~np.isin(g,og)
    for kind,z in methods.items():
        result[kind] = {}
        for j,name in enumerate(('net_edits','min_panel_net_edits')):
            pred = z[:,j]
            val = stats(pred,ref,y,g,fold)
            if kind=='original':
                assert val['corrected']==before[name]['full']['corrected'] and val['harmed']==before[name]['full']['harmed']
            result[kind][name] = dict(full=val,unseen=stats(pred[new],ref[new],y[new],g[new],fold[new]))
            print(kind,name,val['corrected'],val['harmed'],result[kind][name]['unseen']['net'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/edit_panel_transfer.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
