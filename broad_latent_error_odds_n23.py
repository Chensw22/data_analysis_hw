"""Estimate error odds by integrating uncertain unlabeled passenger factors."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp,softmax
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from broad_bill_exception_n23 import ART,ROOT,SEEDS,historical,load,strata,evaluate,write

REPORT=ROOT/"reports/nov23_error_tail_ranking"


def effects(h,center,w,mode):
    if mode=="linear":
        return (h-center)@w,h-center
    # Log E[exp(w'H)] for independent latent Bernoulli states conditional on each
    # passenger; center at a common reference and absorb its level in the intercept.
    d=np.expm1(w)
    denom=1+h*d
    ref=1+center*d
    shift=np.log1p(h*d).sum(1)-np.log(ref).sum()
    deriv=(h/denom-center/ref)*np.exp(w)
    return shift,deriv


def push(h,center,offset,mean,theta,n,mode):
    weights=theta[2:].reshape(2*n,h.shape[1])
    terms=[];derivatives=[]
    for w in weights:
        a,d=effects(h,center,w,mode);terms.append(a);derivatives.append(d)
    a,b=np.split(np.stack(terms,1),2,axis=1)
    tilt=logsumexp(a,axis=1)-logsumexp(b,axis=1)
    z=offset+theta[0]+theta[1]*(offset-mean)+tilt
    mix=np.column_stack([softmax(a,axis=1),-softmax(b,axis=1)])
    return z,mix,derivatives


def learn(h,y,offset,t,q,mode,n):
    center=h[t].mean(0)
    mean=offset[t].mean()
    penalty=3./n
    def objective(v):
        z,mix,deriv=push(h[t],center,offset[t],mean,v,n,mode)
        d=expit(z)-y[t]
        grad=np.concatenate([(d*mix[:,j])@a for j,a in enumerate(deriv)])
        prior=np.r_[1.,4.,np.full(len(grad),penalty)]
        loss=np.logaddexp(0,z).sum()-y[t]@z+.5*((v*v)@prior)
        return loss,np.r_[d.sum(),d@(offset[t]-mean),grad]+prior*v
    rng=np.random.default_rng(492)
    start=np.r_[0.,0.,rng.normal(0,.18,2*n*h.shape[1])]
    res=minimize(objective,start,jac=True,method="L-BFGS-B",
          bounds=[(-5,5)]*len(start),options={"maxiter":320,"ftol":1e-9})
    assert res.success,res.message
    z,_,_=push(h[q],center,offset[q],mean,res.x,n,mode)
    return expit(z)


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig);np.testing.assert_array_equal(y,labels)
    with np.load(ART/"nov23_manifest_energy/representations.npz") as z:
        np.testing.assert_array_equal(z["row"],row)
        real,sham=[z[k][:,:24].astype(float) for k in ("real","sham")]
    scopes=dict(**masks,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy(),
                repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    records,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_manifest_energy/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,ref,fold=z["base"],z["ref"],z["fold"]
        wrong=(ref!=y).astype(float)
        eligible=ref==(base>=.5)
        risk=np.minimum(base,1-base).clip(.001,.4999)
        offset=logit(risk)
        scores={}
        for f in np.unique(fold):
            ids=np.flatnonzero((fold==f)&eligible)
            for a,b in GroupKFold(3,shuffle=True,random_state=438).split(ids,groups=g[ids]):
                for direction in (False,True):
                    t=ids[a][ref[ids[a]]==direction]
                    q=ids[b][ref[ids[b]]==direction]
                    assert not np.intersect1d(g[t],g[q]).size
                    for k,h,mode,n in (("base",np.zeros((len(y),1)),"linear",1),
                                  ("linear",real,"linear",1),("integrated",real,"exact",1),
                                  ("mixture",real,"exact",3),("sham",sham,"exact",3)):
                        p=learn(h,wrong,offset,t,q,mode,n)
                        scores.setdefault("update_"+k,base.copy())[q]=np.where(ref[q],1-p,p).clip(1e-9,1-1e-9)
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        if name!="v131":
            for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        records[name]=result
        print(name,result["update_mixture"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in history.items()}
    write(REPORT/"latent_error_odds.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
