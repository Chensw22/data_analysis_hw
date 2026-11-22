"""Check identifiable scale signal on a known probit population and compare matched real-model changes."""
import numpy as np
from scipy.special import ndtr
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from broad_diverse_rescue_n23 import MORE,predictions
from broad_latent_scale_n23 import OUT, REPORT, SEEDS, fit_location_scale, historical, load, interval, write


def run():
    rng=np.random.default_rng(119)
    sample=rng.normal(size=(45000,3))
    x=np.column_stack([np.ones(len(sample)),sample[:,:2]])
    z=np.where(sample[:,2:]>=0,1.,-1.)
    location=x@np.array([.2,1.2,-.6])
    p=ndtr(location/np.exp(.85*z[:,0]))
    labels=rng.random(len(p))<p
    t,q=np.arange(5000),np.arange(5000,len(p))
    offset=np.zeros(len(p))
    coef,fixed,_=fit_location_scale(x[t],z[t,:0],labels[t],offset[t],x[q],z[q,:0],offset[q],np.zeros(x.shape[1]))
    _,learned,scale=fit_location_scale(x[t],z[t],labels[t],offset[t],x[q],z[q],offset[q],coef)
    synthetic=dict(train=len(t),test=len(q),constant_log_loss=float(log_loss(labels[q],fixed)),
                   real_log_loss=float(log_loss(labels[q],learned)),oracle_log_loss=float(log_loss(labels[q],p[q])),
                   true_scales=np.exp([-.85,.85]).tolist(),fitted_scales=[float(scale[z[q,0]==v].mean()) for v in [-1,1]])
    row,y,_,groups,_=load()
    _,_,masks=historical()
    results={}
    for mode in ["native","update"]:
        diff=[]
        revised=[]
        for seed in SEEDS:
            with np.load(OUT/f"{seed}.npz") as z:
                np.testing.assert_array_equal(row,z["row"])
                np.testing.assert_array_equal(y,z["y"])
                old=z[mode+"_constant"]
                new=z[mode+"_real"]
            before=(old>=.5)==y
            good=(new>=.5)==y
            gain=good.astype(int)-before
            diff.append(gain)
            revised.append(dict(fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                                log_loss_change=float(log_loss(y,new)-log_loss(y,old))))
        results[mode]=dict(changes=revised,paired_ci_pp=interval(np.mean(diff,0),groups))
    added=np.stack([predictions(row,y,s,MORE)[1] for s in SEEDS])
    still_missing=masks["unresolved"] & ~((added>=.5)==y[None,:,None]).all(0).any(1)
    new_hits=[]
    for s in SEEDS:
        with np.load(OUT/f"{s}.npz") as z:
            new_hits.append((z["native_real"]>=.5)==y)
    extra=dict(remaining_after_65=int(still_missing.sum()),
               new_native=int((still_missing&np.logical_and.reduce(new_hits)).sum()))
    write(REPORT/"checks.json",dict(synthetic=synthetic,matched=results,extra=extra))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
