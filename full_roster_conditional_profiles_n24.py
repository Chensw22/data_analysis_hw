"""Fit discriminative and joint objectives starting from exactly the same generative profiles."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import log_softmax, logsumexp, softmax
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_dependency_tree_n24 import encode
from full_roster_disagreement_calibration_n24 import stats
from full_roster_latent_profiles_n24 import fit_components
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_solo_coverage_n24 import cached

MODES = ('initial','conditional','hybrid')


def design(x,dims):
    offsets = np.r_[0,dims.cumsum()]
    rows = np.repeat(np.arange(len(x)),x.shape[1])
    cols = (x+offsets[:-1]).ravel()
    return csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(x),offsets[-1]))


def normalized(params,dims):
    bias = log_softmax(params[0])
    fields = np.split(params[1:],np.cumsum(dims)[:-1])
    return bias,np.vstack([log_softmax(f,axis=0) for f in fields])


def objective(flat,start,data,labels,dims,joint_weight):
    params = flat.reshape(start.shape)
    weights,tables = normalized(params,dims)
    scores = data@tables+weights
    normalizer = logsumexp(scores,axis=1)
    side = np.arange(8)//4
    selected = np.where(side[None,:]==labels[:,None],scores,-np.inf)
    class_ll = logsumexp(selected,axis=1)
    gap = params-start
    penalty = .00025
    loss = ((1-joint_weight)*normalizer-class_ll).mean()+penalty*.5*(gap**2).sum()
    d = ((1-joint_weight)*softmax(scores,axis=1)-np.exp(selected-class_ll[:,None]))/len(labels)
    mass = d.sum(0)
    grad = np.vstack([mass-np.exp(weights)*mass.sum(),data.T@d-np.exp(tables)*mass])
    return loss,(grad+penalty*gap).ravel()


def check_gradient():
    x = np.array([[0,1],[1,2],[0,0],[1,1]])
    dims = x.max(0)+1
    start = np.random.default_rng(138).normal(size=(1+dims.sum(),8))
    labels = np.array([False,True,True,False])
    a = design(x,dims)
    for w in [0.,.3]:
        theta = start.ravel()+.11
        _,grad = objective(theta,start,a,labels,dims,w)
        for j in [2,9,17,len(theta)-1]:
            d = np.zeros_like(theta)
            d[j] = 1e-5
            empirical = (objective(theta+d,start,a,labels,dims,w)[0]-objective(theta-d,start,a,labels,dims,w)[0])/2e-5
            np.testing.assert_allclose(grad[j],empirical,atol=1e-8)


def predict(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    dims = x.max(0)+1
    parts = []
    for c in (False,True):
        members = t[y[t]==c]
        w,fields,_ = fit_components(x,y,dims,members,4,False,True)
        prior = np.log((len(members)+1)/(len(t)+2))
        parts.append(np.vstack([w+prior,*fields]))
    start = np.column_stack(parts)
    train,ask = design(x[t],dims),design(x[q],dims)
    pred = []
    for fraction in (None,0.,.3):
        params = start
        if fraction is not None:
            args = (start,train,y[t],dims,fraction)
            initial_loss = objective(start.ravel(),*args)[0]
            fit = minimize(objective,start.ravel(),args=args,method='L-BFGS-B',jac=True,
                           options=dict(maxiter=700,ftol=1e-8,gtol=1e-5))
            assert fit.success and fit.fun<=initial_loss, fit.message
            params = fit.x.reshape(start.shape)
        w,fields = normalized(params,dims)
        pred.append(softmax(ask@fields+w,axis=1)[:,4:].sum(1))
    return np.column_stack(pred)


def run():
    check_gradient()
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,panel = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    result = {}
    for (tag,ix,f,base,p),(six,archive) in zip([
         ('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
         ('historical',rows,of,champ,old[:,0])],panel,strict=True):
        v = protocol(encode(raw,g),y,g,ix,f,tag,predict,MODES,'conditional_profiles')
        original = cached('latent_profiles',tag,ix,f,g,y)[:,1]
        np.testing.assert_allclose(v[:,0],original,atol=2e-13,rtol=0.)
        missed = (base!=y[ix]) & ((six>=.5)==base[:,None]).all(1)
        overlooked = missed & ((archive>=.5)==base[:,None]).all(1)
        original_rescues = overlooked & ((original>=.5)==y[ix])
        result[tag] = {}
        for j,key in enumerate(MODES):
            for w in (1.,.25):
                pred = v[:,j]>=.5 if w==1 else np.where((p>=.5)==base,(1-w)*p+w*v[:,j]>=.5,base)
                fixed = (base!=y[ix]) & (pred==y[ix])
                r = stats(pred,base,y[ix],g[ix],f)
                r.update(six_fixed=int((fixed & missed).sum()),archive_fixed=int((fixed & overlooked).sum()),
                         prior_rescues_retained=int((fixed & original_rescues).sum()),hard_fixed=int((fixed & hard[ix]).sum()))
                if tag=='full':
                    new = ~np.isin(g[ix],og)
                    r['new_net'] = int((pred[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
                result[tag][f'{key}_{w}'] = r
                print(tag,key,w,r['corrected'],r['harmed'],r['archive_fixed'],r['prior_rescues_retained'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/conditional_profiles.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
