"""Fit population decision boundaries to a smoothed count of mistakes and compare with logistic loss."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_refit_dissent_n23 import features
from broad_weak_disagreement_n23 import ART, OUT, REPORT, SEEDS, archive
from relational_error_corrector import write


def basis(frame, t, q):
    raw = features(frame)[0]
    raw = np.column_stack([raw, frame[BILLS].isna(), frame[BILLS].gt(0), np.log1p(frame[BILLS].sum(axis=1)),
                           frame[BILLS].fillna(0).div(1+frame[BILLS].sum(axis=1),axis=0)])
    mean, scale = raw[t].mean(0), raw[t].std(0).clip(.3)
    a,b = ((raw[ix]-mean)/scale for ix in (t,q))
    a,b = a.clip(-6,6), b.clip(-6,6)
    # Threshold bases on continuous coordinates and multivariate nonlinear
    # interactions via random Fourier features, identical for both losses.
    cont = [j for j in range(a.shape[1]) if np.unique(a[:,j]).size > 8]
    knots = np.quantile(a[:,cont],[.2,.5,.8],axis=0)
    rng = np.random.default_rng(731)
    omega = rng.normal(size=(a.shape[1],192))/np.sqrt(a.shape[1])
    phases = rng.uniform(0,2*np.pi,size=omega.shape[1])
    def transform(x):
        hinge = np.maximum(x[:,None,cont]-knots,0).reshape(len(x),-1)
        return np.column_stack([np.ones(len(x)),x,.5*hinge,np.cos(x@omega+phases)])
    return transform(a),transform(b)


def fit(x, z, y, offset, offset_q):
    sign = 2*y.astype(float)-1
    ridge = np.full(x.shape[1],10.)
    ridge[0] = 1.
    params = np.zeros(x.shape[1])
    result = {}
    for tau in (None, 1., .45):
        def objective(v):
            margin = sign*(offset+x@v)
            if tau is None:
                loss = np.logaddexp(0,-margin).sum()
                deriv = -expit(-margin)
            else:
                # Scaling keeps the derivative at the boundary equal to that
                # of logistic loss. Far-away errors have bounded influence.
                u = expit(-margin/tau)
                loss = 2*tau*u.sum()
                deriv = -2*u*(1-u)
            return loss + .5*ridge@(v*v), x.T@(sign*deriv) + ridge*v
        m = minimize(objective,params,jac=True,method="L-BFGS-B",options={"maxiter":600,"ftol":1e-10})
        if not m.success:
            raise RuntimeError(m.message)
        params = m.x
        if tau is None or tau == .45:
            result["logistic" if tau is None else "smooth_count"] = expit(offset_q+z@params)
    return result


def run():
    row,y,frame,g,_=load()
    old,old_y,bases,_,masks,_,_=archive()
    np.testing.assert_array_equal(row,old)
    np.testing.assert_array_equal(y,old_y)
    plans=[]
    for i,s in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{s}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            plans.append((str(s),z["fold"],bases[i],bases[i]>=.5))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        np.testing.assert_array_equal(z["row"],row)
        plans.append(("v131",z["fold"],z["v108"],z["v131"].astype(bool)))
    result,historical={},{}
    for name,fold,base,ref in plans:
        scores={}
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            x,z=basis(frame,t,q)
            direct=fit(x,z,y[t],np.zeros(len(t)),np.zeros(len(q)))
            for method,p in direct.items():
                scores.setdefault("direct_"+method,np.full(len(y),np.nan))[q]=p
            # Learn a correction on held-out predictions using other bookings in
            # this same outer fold, so no upstream fit used the query labels.
            for a,b in GroupKFold(3,shuffle=True,random_state=667).split(q,groups=g[q]):
                assert not np.intersect1d(g[q[a]],g[q[b]]).size
                x,z=basis(frame,q[a],q[b])
                prior=logit(base[q].clip(.001,.999))
                for method,p in fit(x,z,y[q[a]],prior[a],prior[b]).items():
                    scores.setdefault("adjust_"+method,np.full(len(y),np.nan))[q[b]]=p
        result[name]={}
        for method,p in scores.items():
            assert np.isfinite(p).all()
            # Direct classifiers are standalone; transfers retain existing v131 corrections.
            vote = np.where((ref!=(base>=.5)) & method.startswith("adjust_"),ref,p>=.5)
            hit=vote==y
            gain=hit.astype(int)-(ref==y)
            result[name][method]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                      cohort_fixes={k:int(((gain>0)&v).sum()) for k,v in masks.items()},
                      cohort_harms={k:int(((gain<0)&v).sum()) for k,v in masks.items()})
            if name!="v131":
                historical.setdefault(method,[]).append((hit,gain))
        print(name,[(k,v["fixed"],v["harmed"]) for k,v in result[name].items()],flush=True)
        np.savez_compressed(OUT/f"objective_{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        write(REPORT/"objective.json",dict(results=result))
    stable={}
    for key,trials in historical.items():
        hits=np.logical_and.reduce([h for h,_ in trials])
        stable[key]={k:int((hits&m).sum()) for k,m in masks.items()}
        stable[key]["paired_ci_pp"]=interval(np.mean([v for _,v in trials],0),g)
    write(REPORT/"objective.json",dict(results=result,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
