"""Nyström Laplace kernel classification with a metric learned from prediction gradients."""
import json

import numpy as np
from scipy.linalg import eigh, solve
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_mistake_influence_n24 import design

MODES = ('isotropic', 'diagonal_one', 'diagonal_two', 'full_one', 'full_two')


def distance(x, c, metric):
    xm, cm = x@metric, c@metric
    squared = (xm*x).sum(1)[:,None]+(cm*c).sum(1)-2*xm@c.T
    return np.sqrt(np.maximum(squared,0.))


def kernel(x,c,m,width):
    return np.exp(-distance(x,c,m)/width)


def derivative(x,c,m,width,alpha):
    d = distance(x,c,m)
    coeff = np.divide(np.exp(-d/width)*alpha[None,:], width*d,
                      where=d>1e-6, out=np.zeros_like(d))
    return (coeff@c-x*coeff.sum(1)[:,None])@m


def learned_metric(grad, diagonal):
    covariance = grad.T@grad/len(grad)
    if diagonal:
        covariance = np.diag(np.diag(covariance))
    val, vec = eigh(covariance)
    # A square root regularizes the estimated metric; a small isotropic part keeps it nonsingular.
    smooth = np.sqrt(np.maximum(val,0))
    norm = smooth.mean()
    if norm <= 1e-12:
        return np.eye(len(val))
    return (vec*(.02+.98*smooth/norm))@vec.T


def train_at_metric(x,y,t,q,c,probe,m,pairs):
    displacement = x[pairs[0]]-x[pairs[1]]
    width = max(np.median(np.sqrt(np.maximum(np.sum((displacement@m)*displacement, axis=1),0.))),.01)
    k = kernel(x[t],c,m,width)
    cent = kernel(c,c,m,width)
    normal = k.T@k+1.5*cent+1e-5*np.eye(len(c))
    mean = y[t].mean()
    alpha = solve(normal,k.T@(y[t]-mean),assume_a='pos')
    score = mean+kernel(x[q],c,m,width)@alpha
    grad = derivative(x[probe],c,m,width,alpha)
    return score,grad


def score_fold(raw_features,y,group,t,q):
    x = ((raw_features-raw_features[t].mean(0))/raw_features[t].std(0).clip(.2)).clip(-7,7)
    rng = np.random.default_rng(547)
    center_ids = rng.choice(t,min(len(t),768),replace=False)
    probe = rng.choice(t,min(len(t),1536),replace=False)
    pairs = rng.choice(t,(2,3072))
    eye = np.eye(x.shape[1])
    def one(m):
        return train_at_metric(x,y,t,q,x[center_ids],probe,m,pairs)
    initial, gradient = one(eye)
    scores = [initial]
    for diagonal in (True,False):
        grad = gradient
        for _ in range(2):
            value, grad = one(learned_metric(grad,diagonal))
            scores.append(value)
    return np.column_stack(scores).clip(0,1)


def protocol(x,y,g,rows,fold,tag,scorer=score_fold,modes=MODES,experiment='recursive_metric'):
    out = ROOT/'artifacts'/f'nov24_{experiment}'
    out.mkdir(exist_ok=True)
    p = np.full((len(rows),len(modes)),np.nan)
    for f in np.unique(fold):
        held = fold == f
        q = rows[held]
        t = rows[~held & ~np.isin(g[rows],g[q])]
        assert not np.isin(g[t],g[q]).any()
        path = out/f'{tag}_{f}.npz'
        if not path.exists():
            pred = scorer(x,y,g,t,q)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,train=t,query=q,truth=y[t],prob=pred)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,v in [('train',t),('query',q),('truth',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            p[held] = z['prob']
        print(tag,int(f),flush=True)
    assert np.isfinite(p).all()
    return p


def run(scorer=score_fold,modes=MODES,experiment='recursive_metric'):
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    x = design(raw)
    a = protocol(x,y,g,rows,ofold,'historical',scorer,modes,experiment)
    b = protocol(x,y,g,np.arange(len(y)),fold,'full',scorer,modes,experiment)
    common = ((full>=.5)!=y[:,None]).all(1)
    hard = common[rows] & (champ!=oy)
    unseen = ~np.isin(g,og)
    _,inv,count = np.unique(g,return_inverse=True,return_counts=True)
    solo = count[inv]==1
    report = {}
    for j,name in enumerate(modes):
        report[name] = {}
        for w in (1.,.25):
            p = (1-w)*gate+w*b[:,j]>=.5
            z = (1-w)*old[:,0]+w*a[:,j]>=.5
            if w<1:
                z = np.where((old[:,0]>=.5)==champ,z,champ)
            v = dict(historical=stats(z,champ,oy,og,ofold),full=stats(p,ref,y,g,fold),
                    new_orders=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                    common_fixed=int((common & (p==y)).sum()),persistent_fixed=int((hard & (z==oy) & (p[rows]==oy)).sum()),
                    solo_misses_fixed=int((common & solo & y & p).sum()))
            report[name][str(w)] = v
            print(name,w,[v[k]['net'] for k in ('historical','full','new_orders')],v['common_fixed'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{experiment}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
