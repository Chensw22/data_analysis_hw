"""Fit recursive kernel regressions to signed probability errors from two separate honest panels."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_disagreement_calibration_n24 import stats
from full_roster_mistake_influence_n24 import design
from full_roster_recursive_metric_n24 import learned_metric, train_at_metric
from full_roster_repartition_gate_n24 import panel_data, ROOT


def train(x, y, prior, t, q):
    z = ((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-7,7)
    rng = np.random.default_rng(547)
    centers = z[rng.choice(t,min(768,len(t)),replace=False)]
    probe = rng.choice(t,min(1536,len(t)),replace=False)
    pairs = rng.choice(t,(2,3072))
    eye = np.eye(z.shape[1])
    target = y-prior
    def fit(target, m):
        return train_at_metric(z,target,t,q,centers,probe,m,pairs)
    first, grad = fit(target,eye)
    _, ordinary = fit(y,eye)
    second,_ = fit(target,learned_metric(grad,False))
    placebo,_ = fit(target,learned_metric(ordinary,False))
    return np.column_stack([first,second,placebo])


def run():
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold,names,half,full,ref,_,_,rotated = panel_data()
    k = names.index('mitra96')
    assert k == 4
    x = design(raw)
    base = full[:,k]
    sources = [half[:,k],rotated[:,4]]
    panels = [fold%2,(fold//2)%2]
    scores = {key:np.full((len(y),3),np.nan) for key in ('original','rotated','union')}
    for f in np.unique(fold):
        q = np.flatnonzero(fold==f)
        masks = [(panel==panel[q[0]]) & (fold!=f) for panel in panels]
        for panel in panels:
            assert np.all(panel[q] == panel[q[0]]) and not np.isin(g[panel!=panel[q[0]]],g[q]).any()
        for key, use in [('original',[0]),('rotated',[1]),('union',[0,1])]:
            mask = np.column_stack([masks[i] for i in use])
            count = mask.sum(1)
            t = np.flatnonzero(count>0)
            # For a shared donor both teachers excluded this query panel and this donor.
            # One target per donor gives an order the same weight under either construction.
            prior = np.zeros(len(y))
            prior[t] = (mask[t]*np.column_stack([sources[i][t] for i in use])).sum(1)/count[t]
            assert not np.isin(g[t],g[q]).any()
            scores[key][q] = train(x,y,prior,t,q)
        print('fold',int(f),flush=True)
    for val in scores.values():
        assert np.isfinite(val).all()
    scores['mean_of_panels'] = (scores['original']+scores['rotated'])/2
    common = ((full>=.5)!=y[:,None]).all(1)
    hard = common[rows] & (champ!=oy)
    unseen = ~np.isin(g,og)
    report = {}
    for key,values in scores.items():
        report[key] = {}
        for j,metric in enumerate(('fixed','residual_metric','outcome_metric')):
            d = values[:,j]
            report[key][metric] = {}
            for w in (1.,.25):
                p = np.where((base>=.5)==ref,base+w*d>=.5,ref)
                a = np.where((old[:,0]>=.5)==champ,old[:,0]+w*d[rows]>=.5,champ)
                v = dict(full=stats(p,ref,y,g,fold),historical=stats(a,champ,oy,og,ofold),
                         unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                         common_fixed=int((common & (p==y)).sum()),persistent_fixed=int((hard & (a==oy) & (p[rows]==oy)).sum()))
                report[key][metric][str(w)] = v
            print(key,metric,[(v['historical']['net'],v['full']['net'],v['unseen']['net'],v['common_fixed'])
                              for v in report[key][metric].values()],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/residual_metric_panels.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
