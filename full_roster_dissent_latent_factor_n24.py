"""Infer error odds from a panel of dependent model-family votes on held-out bookings."""
import json

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets


def likelihood(theta,x,v,nodes,log_weights,derivative=False):
    width = x.shape[1]
    coef = theta.reshape(width+1,v.shape[1])
    logits = (x@coef[:width])[:,None,:]+nodes[None,:,None]*coef[-1]
    terms = (v[:,None,:]*logits-np.logaddexp(0,logits)).sum(2)+log_weights
    total = logsumexp(terms,axis=1)
    if not derivative:
        return total
    residual = np.exp(terms-total[:,None])[:,:,None]*(expit(logits)-v[:,None,:])
    grad = np.vstack([x.T@residual.sum(1),(residual*nodes[None,:,None]).sum(axis=(0,1))])
    return -total.mean()+.006*(coef**2).sum(),(grad/len(x)+.012*coef).ravel()


def evidence(x,v,error,t,q,factor):
    if factor:
        points,w = hermgauss(7)
        nodes,log_w = points*np.sqrt(2),np.log(w/np.sqrt(np.pi))
    else:
        nodes,log_w = np.zeros(1),np.zeros(1)
    ll = []
    for side in (False,True):
        s = t[error[t]==side]
        start = np.zeros((x.shape[1]+1,v.shape[1]))
        start[0] = logit((v[s].sum(0)+1)/(len(s)+2))
        if factor:
            start[-1] = .8
        result = minimize(likelihood,start.ravel(),args=(x[s],v[s],nodes,log_w,True),jac=True,
                          method='L-BFGS-B',options=dict(maxiter=150))
        assert result.success
        ll.append(likelihood(result.x,x[q],v[q],nodes,log_w))
    return ll[1]-ll[0]


def run():
    raw,y,g,fold = data()
    _,_,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    columns,panels = prediction_sets(raw,y,g,fold,rows,of)
    family = np.array([k.split('/')[0] for k in columns])
    report = {}
    for (tag,ix,f,base,p),(six,archive) in zip([('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
             ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        error = base!=y[ix]
        offered = ((np.c_[six,archive]>=.5)!=base[:,None]).any(1)
        candidates = [six[:,:3],six[:,3:],*[archive[:,family==key] for key in np.unique(family)]]
        v = np.column_stack([(np.median(table,axis=1)>=.5)!=base for table in candidates]).astype(float)
        risk = logit(np.where(base,1-p,p).clip(.003,.997))
        x = np.column_stack([np.ones(len(ix)),risk/3,base])
        methods = ('prior','linear_votes','independent','factor','scaled_independent','scaled_factor')
        pred = {k:np.zeros(len(ix),bool) for k in methods}
        for level in np.unique(f):
            ids = np.flatnonzero(f==level)
            chunks = [ids[q] for _,q in GroupKFold(4,shuffle=True,random_state=629).split(ids,groups=g[ix][ids])]
            for j,query in enumerate(chunks):
                t = np.r_[chunks[(j+1)%4],chunks[(j+2)%4]]
                cal = chunks[(j+3)%4]
                t,cal,q = [s[offered[s]] for s in (t,cal,query)]
                assert not np.isin(g[ix][q],g[ix][np.r_[t,cal]]).any()
                both = np.r_[cal,q]
                prior = LogisticRegression(C=.3,max_iter=300).fit(x[t,1:],error[t]).decision_function(x[both,1:])
                pred['prior'][q] = prior[len(cal):]>0
                xx = np.column_stack([x[:,1:],v])
                pred['linear_votes'][q] = LogisticRegression(C=.15,max_iter=300).fit(xx[t],error[t]).predict(xx[q])
                for factor,name in [(False,'independent'),(True,'factor')]:
                    signal = evidence(x,v,error,t,both,factor)
                    pred[name][q] = (prior+signal)[len(cal):]>0
                    def objective(w):
                        l = prior[:len(cal)]+w[0]*signal[:len(cal)]
                        return (np.logaddexp(0,l)-error[cal]*l).mean()+.005*w[0]**2
                    w = minimize(objective,[.3],bounds=[(0,2)],method='L-BFGS-B')
                    assert w.success
                    pred['scaled_'+name][q] = (prior+w.x[0]*signal)[len(cal):]>0
        result = {}
        for name,flip in pred.items():
            outcome = base^flip
            r = stats(outcome,base,y[ix],g[ix],f)
            if tag=='full':
                new = ~np.isin(g[ix],og)
                r['new_orders_net'] = int((outcome[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
            result[name] = r
        report[tag] = dict(disputed_rows=int(offered.sum()),disputed_errors=int((offered&error).sum()),families=v.shape[1],methods=result)
        print(tag,report[tag]['disputed_errors'],[(n,r['corrected'],r['harmed']) for n,r in result.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/dissent_latent_factor.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
