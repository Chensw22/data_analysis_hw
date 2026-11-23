"""Fit global and passenger-specific mixtures of 19 independently evaluated expert probabilities."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS
from broad_tree_field_dividends_n23 import hard_cohorts
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_neighborhood_disputes_n23 import DEST, NAMES
from full_roster_sleep_bill_factorial_n23 import VIEWS
from half_roster_disagreement_eval_n23 import DEST as METRIC, NEIGHBOR


def banks(raw,y,fold):
    base = [*FAMILIES, 'mitra96']
    half = half_bank(raw,y,fold)
    full = np.column_stack([load_scores(raw,y,fold,OUT/f'{k}.npz',['p'])[:,0] for k in base])
    with np.load(METRIC/'others.npz') as z:
        for k,v in [('passenger',raw.PassengerId), ('y',y), ('panel',fold%2)]:
            np.testing.assert_array_equal(z[k],v)
        hn = np.column_stack([z[k] for k in NEIGHBOR])
    fn = load_scores(raw,y,fold,ROOT/'artifacts/nov23_full_roster_metric_neighbors/oof.npz',NEIGHBOR)
    factorial = ROOT/'artifacts/nov23_sleep_bill_factorial/oof.npz'
    prior = ROOT/'artifacts/nov23_bill_prior_balance'
    names = [*base, *NEIGHBOR, *NAMES, *VIEWS, *EVIDENCE, 'restored_full']
    out = []
    for key,s,b,n in [('half',fold%2,half,hn),('full',fold,full,fn)]:
        geo = load_scores(raw,y,s,DEST/f'{key}.npz',NAMES)
        view = load_scores(raw,y,fold,factorial,[key])
        extra = load_scores(raw,y,s,prior/f'{key}.npz',[*EVIDENCE,'restored_full'])
        scores = np.column_stack([b,n,geo,view,extra])
        assert scores.shape == (len(y),len(names)) and np.isfinite(scores).all()
        out.append(scores)
    return names, *out


def context(raw, groups, p):
    fees = raw[BILLS].fillna(0)
    _, code, size = np.unique(groups,return_inverse=True,return_counts=True)
    return np.column_stack([raw.CryoSleep.eq(True),raw.CryoSleep.isna(),raw.Age.fillna(-1)/40,
        np.log1p(size[code]),np.log1p(fees),raw[BILLS].isna().sum(axis=1),p[:,:5].mean(1),
        p[:,:5].std(1),p.std(1)])


def weights_train(xtrain, p, y):
    n, width = xtrain.shape
    params = np.zeros((width, p.shape[1]))
    penalty = np.full((width,1), 5.)
    penalty[0] = .25
    def objective(theta):
        coef = theta.reshape(params.shape)
        weights = softmax(xtrain @ coef,axis=1)
        prob = np.sum(weights*p,axis=1).clip(1e-8,1-1e-8)
        loss = -np.sum(np.where(y,np.log(prob),np.log1p(-prob)))
        gradient = ((prob-y)/(prob*(1-prob)))[:,None]*weights*(p-prob[:,None])
        return loss + .5*np.sum(penalty*coef**2), (xtrain.T @ gradient + penalty*coef).ravel()
    solution = minimize(objective,params.ravel(),method='L-BFGS-B',jac=True,options={'maxiter':400,'ftol':1e-10})
    assert solution.success, solution.message
    return solution.x.reshape(params.shape)


def run():
    raw,y,g,fold = data()
    names, h, p = banks(raw,y,fold)
    idx = [names.index(k) for k in EVIDENCE]
    hdiff, hx = proposal_design(h[:,:5],h[:,idx])
    fdiff, fx = proposal_design(p[:,:5],p[:,idx])
    base = p[:,4] >= .5
    gate = base.copy()
    a, b = context(raw,g,h), context(raw,g,p)
    policies = {k:np.empty(len(y)) for k in ['fixed_pool','context_pool','unconstrained','context_one_expert']}
    for f in np.unique(fold):
        t = np.flatnonzero((fold%2 == f%2) & (fold != f))
        q = np.flatnonzero(fold == f)
        assert not np.isin(g[q],g[t]).any()
        tt, qq = t[hdiff[t]], q[fdiff[q]]
        learner = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
        learner.fit(hx[tt],y[tt] != (h[tt,4] >= .5))
        flips = qq[learner.predict_proba(fx[qq])[:,1] >= .5]
        gate[flips] = ~base[flips]
        scaler = StandardScaler().fit(a[t])
        train = np.column_stack([np.ones(len(t)),scaler.transform(a[t]).clip(-4,4)])
        query = np.column_stack([np.ones(len(q)),scaler.transform(b[q]).clip(-4,4)])
        for key, width in [('fixed_pool',1),('context_pool',train.shape[1])]:
            coefficients = weights_train(train[:,:width],h[t],y[t])
            weights = softmax(query[:,:width] @ coefficients,axis=1)
            policies[key][q] = (weights*p[q]).sum(1)
            if key == 'context_pool':
                policies['context_one_expert'][q] = p[q,np.argmax(weights,axis=1)]
        learner.fit(logit(h[t].clip(.001,.999)),y[t])
        policies['unconstrained'][q] = learner.predict_proba(logit(p[q].clip(.001,.999)))[:,1]
    assert (gate == y).sum() == 7118
    agree = ((p[:,:5] >= .5) == base[:,None]).all(1)
    seven = agree & ((p[:,[names.index('without_near'),names.index('only_near')]] >= .5) == base[:,None]).all(1)
    remaining = seven & (base != y)
    ids = [names.index(k) for k in [*VIEWS,'balanced_full']]
    previous_missed = remaining & ((p[:,ids] >= .5) != y[:,None]).all(1)
    assert remaining.sum() == 819 and previous_missed.sum() == 329
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    hard = np.zeros(len(y),bool)
    hard[dev] = hard_cohorts(dev,y[dev])[2]
    def summary(pick):
        delta = (pick == y).astype(int)-(gate == y)
        return dict(accuracy=float((pick == y).mean()),correct=int((pick == y).sum()),
            fixed_vs_124=int((delta == 1).sum()),harmed_vs_124=int((delta == -1).sum()),
            original_common_correct=int(((pick == y) & agree & (base != y)).sum()),
            of_819=int(((pick == y) & remaining).sum()), of_329=int(((pick == y) & previous_missed).sum()),
            historical_correct=int(((pick == y) & hard).sum()),additional_net=int(delta.sum()-delta[dev].sum()),
            fold_net=[int(delta[fold == f].sum()) for f in np.unique(fold)],ci_pp=interval(delta,g))
    report = dict(experts=names,reference=summary(gate),evaluations={k:summary(v >= .5) for k,v in policies.items()})
    report['all_experts_wrong'] = int(((p >= .5) != y[:,None]).all(1).sum())
    path = ROOT/'reports/nov23_matched_learning_curve/conditional_pool.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
