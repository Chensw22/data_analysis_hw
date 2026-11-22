"""Learn smooth monotone min-max combinations on independent calibration bookings."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_diverse_rescue_n23 import ACTIONS,MORE,predictions
from broad_ordered_ensemble_n23 import ART,OUT,REPORT,SEEDS,load,historical,interval,write


def score(x,w,b,kind,tau=.45):
    planes=(x@w+b).reshape(len(x),3,3)
    if kind=="linear":
        return planes.mean((1,2)),None
    lower=-tau*(logsumexp(-planes/tau,axis=2)-np.log(3))
    out=tau*(logsumexp(lower/tau,axis=1)-np.log(3))
    derivative=softmax(lower/tau,axis=1)[:,:,None]*softmax(-planes/tau,axis=2)
    return out,derivative.reshape(len(x),9)


def fit(x,y,q,kind):
    k=x.shape[1]
    prior=np.zeros((k,9))
    prior[0]=1.
    rng=np.random.default_rng(427)
    w0=.65*prior
    # Distinct initial planes avoid a symmetric stationary configuration.
    for j in range(9):
        w0[rng.integers(k),j]+=.35
    b0=rng.normal(scale=.25,size=9)
    def unpack(params):
        return params[:-9].reshape(k,9),params[-9:]
    def objective(params):
        w,b=unpack(params)
        z,d=score(x,w,b,kind)
        r=expit(z)-y
        sensitivity=r[:,None]*(np.ones((1,9))/9 if d is None else d)
        ridge=(w-prior)
        loss=np.sum(np.logaddexp(0.,z)-y*z)+.5*(ridge**2).sum()+.5*.4*(b@b)
        return loss,np.r_[(x.T@sensitivity+ridge).ravel(),sensitivity.sum(0)+.4*b]
    opt=minimize(objective,np.r_[w0.ravel(),b0],jac=True,method="L-BFGS-B",
                 bounds=[(0,None)]*(k*9)+[(None,None)]*9,options={"maxiter":550,"ftol":1e-8})
    if not opt.success:
        raise RuntimeError(opt.message)
    w,b=unpack(opt.x)
    assert (w>=0).all()
    return expit(score(q,w,b,kind)[0])


def control():
    rng=np.random.default_rng(892)
    x=rng.normal(size=(7500,4))
    prob=expit(4*np.maximum(np.minimum(x[:,0],x[:,1]),np.minimum(x[:,2],x[:,3])))
    y=rng.random(len(x))<prob
    result={}
    for mode in ("linear","minmax"):
        q=fit(x[:4500],y[:4500],x[4500:],mode)
        result[mode]=dict(accuracy=float(((q>=.5)==y[4500:]).mean()),probability_mse=float(((q-prob[4500:])**2).mean()))
    return result


def run():
    row,y,_,g,_=load()
    orig,target,masks=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(target,y)
    report,hist={},{}
    check=control()
    for name in [*map(str,SEEDS),"v131"]:
        cols=[]
        for folder,field in (("broad_passenger_differences/balanced","native_pair"),
                             ("broad_count_supervision","native_similar"),("broad_latent_scale","native_real")):
            with np.load(ART/f"nov23_{folder}/{name}.npz") as z:
                for k,v in (("row",row),("y",y)):
                    np.testing.assert_array_equal(z[k],v)
                if cols:
                    for k,v in (("base",base),("ref",ref),("fold",fold)):
                        np.testing.assert_array_equal(z[k],v)
                base,ref,fold=z["base"],z["ref"],z["fold"]
                cols.append(z[field])
        plans={"small":np.column_stack([base,*cols])}
        if name!="v131":
            anchor,members,folds=predictions(row,y,int(name),{**ACTIONS,**MORE})
            np.testing.assert_array_equal(anchor,base)
            np.testing.assert_array_equal(folds,fold)
            plans["wide"]=np.column_stack([base,members])
        locked=ref!=(base>=.5)
        report[name]={}
        scores={}
        for bank,p in plans.items():
            x=logit(p.clip(.001,.999))
            for method in ("linear","minmax"):
                s=bank+"_"+method
                prob=base.copy()
                for f in np.unique(fold):
                    ids=np.flatnonzero((fold==f)&~locked)
                    for a,b in GroupKFold(3,shuffle=True,random_state=823).split(ids,groups=g[ids]):
                        t,q=ids[a],ids[b]
                        assert not np.intersect1d(g[t],g[q]).size
                        prob[q]=fit(x[t],y[t],x[q],method)
                scores[s]=prob
                hit=np.where(locked,ref,prob>=.5)==y
                gain=hit.astype(int)-(ref==y)
                report[name][s]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()))
                if name!="v131":
                    hist.setdefault(s,[]).append((hit,gain))
        np.savez_compressed(OUT/f"smooth_{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        write(REPORT/"smooth.json",dict(plans=report,control=check))
        print(name,report[name],flush=True)
    stable={}
    for k,values in hist.items():
        hit=np.logical_and.reduce([h for h,_ in values])
        stable[k]={n:int((hit&m).sum()) for n,m in masks.items()}
        stable[k]["gain_ci95_pp"]=interval(np.mean([d for _,d in values],0),g)
    write(REPORT/"smooth.json",dict(plans=report,control=check,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
