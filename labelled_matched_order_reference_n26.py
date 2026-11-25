"""Predict a held passenger by conditioning fitted pair probabilities on labeled reference passengers."""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from frozen_queue_shared_errors_n26 import paired
from localized_opposite_order_pairs_n26 import coordinates
from manifest_sleep_contrast_n26 import design
from matched_case_ordering_n26 import forecasts
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, ticker


def neighbors(x, source, dest, groups):
    d = cdist(x[dest],x[source],'sqeuclidean')
    d[groups[dest,None]==groups[source]] = np.inf
    idx = np.argsort(d,axis=1,kind='stable')[:,:8]
    gap = np.take_along_axis(d,idx,axis=1)
    assert np.isfinite(gap).all()
    return np.repeat(dest,len(idx[0])),source[idx].ravel(),gap.ravel()


def conditional_system(p, x, a, b, gap):
    pa, pb = p[a],p[b]
    prior = np.column_stack([(1-pa)*(1-pb), pa*(1-pb)+(1-pa)*pb, pa*pb])
    features = np.column_stack([np.ones(len(a)),pa+pb-1,np.abs(pa-pb),gap/(1+gap)])
    return np.log(prior),features,x[a]-x[b]


def fit_system(p, x, bank, labels, edges):
    a,b,gap = edges
    assert np.isin(a,bank).all() and np.isin(b,bank).all()
    prior, features, difference = conditional_system(p,x,a,b,gap)
    count = labels[a].astype(int)+labels[b].astype(int)
    def objective(params):
        beta = params.reshape(features.shape[1],2)
        logits = prior.copy()
        logits[:,1:] += features @ beta
        loss = np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(a)),count]) + .05*(params@params)
        gradient = features.T @ (softmax(logits,axis=1)[:,1:]-np.eye(3)[count,1:])/len(a)
        return loss, gradient.ravel()+.1*params
    res = minimize(objective,np.zeros(features.shape[1]*2),method='L-BFGS-B',jac=True)
    assert res.success
    # For discordant labels, predict which passenger is transported; reversing the pair must reverse its odds.
    keep = count==1
    assert keep.any()
    dx = difference[keep]
    truth = labels[a[keep]]
    initial = logit(p[a[keep]])-logit(p[b[keep]])
    def order_objective(w):
        logits = initial + dx@w
        loss = np.mean(np.logaddexp(0,logits)-truth*logits)+.05*(w@w)
        return loss,dx.T@(expit(logits)-truth)/len(logits)+.1*w
    rank = minimize(order_objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True)
    assert rank.success
    return res.x.reshape(features.shape[1],2),rank.x


def evaluate(p, x, edges, model, labels, sham, n):
    a,b,gap = edges
    prior, features, dx = conditional_system(p,x,a,b,gap)
    beta, w = model
    prior[:,1:] += features@beta
    zero,one,two = softmax(prior,axis=1).T
    first = expit(logit(p[a])-logit(p[b])+dx@w)
    parts = [two+one*first]
    for yref in (labels[b],sham[b]):
        parts.append(np.where(yref,two/(two+one*(1-first)),one*first/(zero+one*first)))
    weight = (1/(1+gap)).reshape(n,-1)
    return np.column_stack([np.average(a.reshape(n,-1),weights=weight,axis=1) for a in parts])


def run():
    raw,y,g,f,known,versions = paired()
    scores = forecasts(raw,y,g,f)
    p = expit(np.median(scores[:,:3],axis=1)).clip(.01,.99)
    geom = np.column_stack([design(raw,g), np.log1p(raw[FEES].to_numpy(float))])
    cohorts = dict(sleep=raw.CryoSleep.eq(True).to_numpy() & ~raw[FEES].gt(0).any(axis=1).to_numpy(),
                   awake=raw.CryoSleep.eq(False).to_numpy() & raw[FEES].gt(0).any(axis=1).to_numpy())
    report = {}
    for name,eligible in cohorts.items():
        estimates = np.zeros((len(y),3))
        seen = np.zeros(len(y),int)
        for k in np.unique(f):
            # All scores were fitted without labels from this entire base fold.
            ids = np.flatnonzero((f==k) & eligible)
            parts = [ids[q] for _,q in GroupKFold(4,shuffle=True,random_state=9137).split(ids,groups=g[ids])]
            for j,q in enumerate(parts):
                for shift in (1,2,3):
                    anchors = parts[(j+shift)%4]
                    t = np.concatenate([z for i,z in enumerate(parts) if i not in (j,(j+shift)%4)])
                    assert not np.isin(g[t],g[q]).any() and not np.isin(g[anchors],np.r_[g[t],g[q]]).any()
                    scene = coordinates(geom,t)
                    train_edges = neighbors(scene,t,t,g)
                    model = fit_system(p,scores,t,y,train_edges)
                    test_edges = neighbors(scene,anchors,q,g)
                    placebo = np.zeros(len(y),bool)
                    rng = np.random.default_rng(263+j+shift)
                    bins = (p[anchors]*10).astype(int)
                    for b in np.unique(bins):
                        rows = anchors[bins==b]
                        placebo[rows] = rng.permutation(y[rows])
                    estimates[q] += evaluate(p,scores,test_edges,model,y,placebo,len(q))
                    seen[q] += 1
        assert (seen[eligible]==3).all()
        pred = estimates/3 >= .5
        parent = versions[:,-1]
        unanimous = known & eligible & (versions==parent[:,None]).all(1)
        usual = unanimous & (parent==(name=='sleep'))
        rules = dict(marginal=pred[:,0]!=parent, labeled=pred[:,1]!=parent, shuffled=pred[:,2]!=parent,
                     newly_flipped=(pred[:,1]!=parent)&(pred[:,0]==parent),
                     only_true_anchors=(pred[:,1]!=parent)&(pred[:,2]==parent))
        report[name] = {n:{cohort:ticker(m,y!=parent,rule) for cohort,m in [('all_common',unanimous),('usual',usual)]}
                        for n,rule in rules.items()}
        print(name,{n:s['usual'] for n,s in report[name].items()},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/labelled_matched_order_reference.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
