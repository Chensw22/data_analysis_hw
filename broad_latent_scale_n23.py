"""Fit a heteroskedastic binary probit and test whether its learned boundary transfers."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr, ndtri
from threadpoolctl import threadpool_limits
from broad_direct_accuracy_n23 import basis
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, BILLS, historical, evaluate, interval, load, teachers, write

OUT = ART / "nov23_broad_latent_scale"
REPORT = ROOT / "reports/nov23_broad_latent_scale"


def scale_fields(frame):
    b = frame[BILLS]
    a = b.fillna(0).to_numpy(float)
    total = a.sum(1)
    shares = a / (1 + total[:,None])
    return np.column_stack([b.isna(), b.gt(0), (a>0).sum(1), np.log1p(total),
                            (shares**2).sum(1), np.log(frame.BookingSize),
                            frame.CryoSleep.astype(str).eq("True"), frame.Age.isna(),
                            frame.Age.lt(13), frame.CabinNumber.isna()])


def fit_location_scale(x, z, target, offset, xq, zq, offset_q, initial):
    # No constant in log-scale. Ridge regularization and bounded scale factors
    # distinguish a global rescaling from genuine heterogeneity.
    n = x.shape[1]
    penalty = np.r_[.2, np.full(n-1, 12.), np.full(z.shape[1], 40.)]
    sign = 2*target.astype(float)-1

    def linear(v, design, scales, intercept):
        loc = intercept + design@v[:n]
        u = np.tanh((scales@v[n:])/1.5)
        inv_scale = np.exp(-1.5*u)
        return loc*inv_scale, inv_scale, u

    def objective(v):
        value, inv, u = linear(v,x,z,offset)
        margin = sign*value
        loss = log_ndtr(margin)
        slope = -sign*np.exp(-.5*margin**2 - .5*np.log(2*np.pi) - loss)
        grad = np.r_[x.T@(slope*inv), z.T@(-slope*value*(1-u*u))]
        return -loss.sum()+.5*(penalty*v*v).sum(),grad+penalty*v

    opt = minimize(objective,np.r_[initial,np.zeros(z.shape[1])],jac=True,method="L-BFGS-B",
                   options={"maxiter":700,"ftol":1e-10})
    if not opt.success:
        raise RuntimeError(opt.message)
    value,inv,_ = linear(opt.x,xq,zq,offset_q)
    return opt.x[:n],ndtr(value).clip(.0001,.9999),1/inv


def run():
    row,y,frame,g,_ = load()
    old,old_y,masks = historical()
    np.testing.assert_array_equal(old,row)
    np.testing.assert_array_equal(old_y,y)
    scales = scale_fields(frame)
    OUT.mkdir(exist_ok=True)
    results,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as src:
            for key,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(src[key],v)
            base,ref,fold=src["base"],src["ref"],src["fold"]
        scores={}
        ranges=[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            x,xq=basis(frame,t,q)
            z=(scales-scales[t].mean(0))/scales[t].std(0).clip(.3)
            z=z.clip(-5,5)
            pt,_=teachers(frame,y,g,t,q,1171+int(f))
            for mode,loc,lq in [("native",np.zeros(len(t)),np.zeros(len(q))),
                                 ("update",ndtri(pt.clip(.002,.998)),ndtri(base[q].clip(.002,.998)))]:
                xt=x if mode=="native" else x*.35
                xquery=xq if mode=="native" else xq*.35
                theta,p,_=fit_location_scale(xt,z[t,:0],y[t],loc,xquery,z[q,:0],lq,np.zeros(x.shape[1]))
                scores.setdefault(mode+"_constant",np.full(len(y),np.nan))[q]=p
                rng=np.random.default_rng(843)
                tq,qq=rng.permutation(t),rng.permutation(q)
                for kind,train_z,query_z in (("real",z[t],z[q]),("shuffled",z[tq],z[qq])):
                    _,p,scale=fit_location_scale(xt,train_z,y[t],loc,xquery,query_z,lq,theta)
                    scores.setdefault(mode+"_"+kind,np.full(len(y),np.nan))[q]=p
                    if kind=="real":
                        ranges.append(dict(fold=int(f),mode=mode,scale_quantiles=np.quantile(scale,[.1,.5,.9]).tolist()))
            print(name,f,flush=True)
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks)
        results[name]=dict(methods=result,scales=ranges)
        if name!="v131":
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        write(REPORT/"results.json",dict(plans=results))
    stable={}
    for k,values in history.items():
        hit=np.logical_and.reduce([h for h,_ in values])
        stable[k]={n:int((hit&m).sum()) for n,m in masks.items()}
        stable[k]["gain_ci_pp"]=interval(np.mean([d for _,d in values],0),g)
    write(REPORT/"results.json",dict(plans=results,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
