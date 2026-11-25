"""Train passenger error scores directly on whether any member of a similar pair was wrong."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conflict_pair_trigger_n26 import inputs_for_trigger
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,data


def make_edges(a,b,order,t,n,side,g):
    eligible = np.zeros(n,bool)
    eligible[t] = True
    degree = np.zeros(n,int)
    first,second = [],[]
    for e in order[eligible[a[order]]&eligible[b[order]]]:
        i,j = a[e],b[e]
        if degree[i]<8 and degree[j]<8:
            degree[i]+=1
            degree[j]+=1
            first.append(i)
            second.append(j)
    i,j = np.array(first),np.array(second)
    other = j.copy()
    rng = np.random.default_rng(77531)
    for direction in (False,True):
        ix = np.flatnonzero(side[i]==direction)
        other[ix] = rng.permutation(j[ix])
        for pos in ix[g[i[ix]]==g[other[ix]]]:
            valid = ix[(g[i[ix]]!=g[other[pos]])&(g[other[ix]]!=g[i[pos]])]
            alt = rng.choice(valid)
            other[pos],other[alt] = other[alt],other[pos]
    assert (g[i]!=g[other]).all() and np.array_equal(side[i],side[other])
    np.testing.assert_array_equal(np.sort(other),np.sort(j))
    return [np.searchsorted(t,e) for e in (i,j,other)]


def existential(z,target,a,b):
    # For each pair the model predicts P(any error) = 1 - (1-p_i)(1-p_j).
    p = expit(z)
    log_none = -np.logaddexp(0,z[a])-np.logaddexp(0,z[b])
    prob_any = -np.expm1(log_none)
    observed = target[a]|target[b]
    loss = np.where(observed,-np.log(prob_any.clip(1e-14)),-log_none)
    coeff = np.where(observed,-np.exp(log_none)/prob_any.clip(1e-14),1.)
    gradient = np.bincount(a,weights=coeff*p[a],minlength=len(z))
    gradient += np.bincount(b,weights=coeff*p[b],minlength=len(z))
    return loss.mean(),gradient/len(a)


def fit(x,y,a,b,mode):
    def objective(w):
        z = x@w
        loss = np.mean(np.logaddexp(0,z)-y*z)
        derivative = (expit(z)-y)/len(z)
        if mode!='individual':
            value,grad = existential(z,y,a,b)
            if mode=='two_events':
                second,reverse = existential(-z,~y,a,b)
                value,grad = (value+second)/2,(grad-reverse)/2
            loss,derivative = (loss+value)/2,(derivative+grad)/2
        ridge = w.copy()
        ridge[0] = 0.
        return loss+.006*(ridge@ridge),x.T@derivative+.012*ridge
    fit = minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'gtol':1e-8,'ftol':1e-12,'maxiter':300})
    assert fit.success,fit.message
    return fit.x


def validate_gradient():
    z = np.array([-.7,.32,.4,-2.1,1.62])
    y = np.array([0,1,0,0,1],bool)
    a,b = np.array([0,0,1,3]),np.array([1,2,4,4])
    _,g = existential(z,y,a,b)
    for k,step in enumerate(1e-6*np.eye(len(z))):
        diff = (existential(z+step,y,a,b)[0]-existential(z-step,y,a,b)[0])/(2e-6)
        np.testing.assert_allclose(diff,g[k],atol=1e-8)


def run():
    validate_gradient()
    raw,y,g,f = data()
    _,cur,_ = inputs()
    side,wrong,x,a,b,d,gap = inputs_for_trigger(raw,y,g,f)
    order = np.lexsort((b,a,d))
    modes = ['individual','any_error','two_events','shuffled_any_error']
    records = np.full((len(y),len(modes)),np.nan)
    edge_count = 0
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        for u,v in GroupKFold(3,shuffle=True,random_state=153).split(rows,groups=g[rows]):
            t,q = rows[u],rows[v]
            assert np.all(t[:-1]<t[1:]) and not np.isin(g[t],g[q]).any()
            i,j,other = make_edges(a,b,order,t,len(y),side,g)
            edge_count += len(i)
            standard = StandardScaler().fit(x[t])
            fields = np.column_stack([np.ones(len(x)),standard.transform(x).clip(-6,6)])
            for m,method in enumerate(modes):
                coef = fit(fields[t],wrong[t],i,other if method=='shuffled_any_error' else j,method)
                records[q,m] = expit(fields[q]@coef)
    assert np.isfinite(records).all()
    valid = (side==cur['parent'])&cur['known']
    votes = records>.5
    report = dict(training_pair_uses_across_fits=edge_count,scopes={})
    for s,mask in [('all',valid),('common',valid&cur['common']),('new',valid&cur['new'])]:
        r = {}
        report['scopes'][s] = r
        for m,method in enumerate(modes):
            prediction = votes[:,m]
            r[method] = dict(challenges=summary(prediction,wrong,mask,f),
                additions=summary(prediction&~votes[:,0],wrong,mask,f),
                removed=summary(votes[:,0]&~prediction,~wrong,mask,f),
                brier=float(np.mean((records[mask,m]-wrong[mask])**2)))
        print(s,[(k,v['challenges']['fixed'],v['challenges']['harmed']) for k,v in r.items()],flush=True)
    report['scope'] = ('All outputs are passenger error probabilities. Logistic point loss and pair loss carry equal '
      'weight with identical features and regularization; the control uses point loss alone. The shuffled '
      'edges preserve endpoint degrees and predicted class. Pair relationships never depend on outcomes.')
    (ROOT/'reports/nov23_matched_learning_curve/train_existential_error_bags.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
