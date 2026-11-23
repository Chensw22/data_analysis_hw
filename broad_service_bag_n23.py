"""Treat a passenger's recorded paid services, cabin and incomplete basket as alternative explanations."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,softmax
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_fixed_vote_totals_n23 import REPORT


def design(frame):
    fees=frame[BILLS].to_numpy(float)
    observed=np.nan_to_num(fees)
    total=observed.sum(1)
    n=len(frame)
    missing=np.isnan(fees).sum(1)
    # A cabin explanation is available to everyone; a missing-bill explanation
    # is available only when at least one service has an unknown amount.
    available=np.column_stack([np.ones(n,bool),observed>0,missing>0])
    amt=np.column_stack([np.zeros(n),observed,np.zeros(n)])
    a=np.log1p(amt)/7
    share=amt/np.maximum(total[:,None],1)
    age=frame.Age.fillna(27).to_numpy()/40
    account=np.log1p(total)/7
    cabin=frame.CabinNumber.fillna(800).to_numpy()/900
    fields=['CryoSleep','Deck','Side','HomePlanet','VIP','Destination']
    cats=pd.get_dummies(frame[fields],dtype=float).to_numpy()
    person=np.column_stack([np.ones(n),cats,age,np.maximum(0,age-.4),np.maximum(0,age-1.),
                            frame.Age.isna(),cabin,np.maximum(0,cabin-.8),frame.CabinNumber.isna(),
                            np.log(frame.BookingSize),account,(observed>0).sum(1)/4,missing/5])
    # Common person features with service-specific coefficients. The two service-only
    # channels represent the fee and its share of the observed basket.
    x=np.concatenate([np.broadcast_to(person[:,None,:],(n,7,person.shape[1])),a[:,:,None],share[:,:,None]],2)
    asleep=frame.CryoSleep.eq('True').to_numpy()
    context=np.column_stack([np.ones(n),asleep,age,account,missing/5])
    gate=np.concatenate([np.broadcast_to(context[:,None,:],(n,7,context.shape[1])),a[:,:,None],share[:,:,None]],2)
    # Doubling service expenditure changes its prior weight gradually; zeros cannot
    # activate an observed service. Never assign unavailable alternatives any mass.
    prior=np.where(available,np.log1p(amt)/7,-np.inf)
    return x,gate,prior


def fit(x,g,prior,y,t,q,mode,return_parameters=False):
    risk_shape=x.shape[1:]
    split=np.prod(risk_shape)
    gating=mode=='learned'
    size=split+(np.prod(g.shape[1:]) if gating else 0)
    def forward(w,xx,gg,lp):
        risk=np.einsum('nsd,sd->ns',xx,w[:split].reshape(risk_shape))
        gate=lp+np.einsum('nsd,sd->ns',gg,w[split:].reshape(gg.shape[1:])) if gating else lp
        weights=softmax(gate,axis=1)
        if mode=='pooled_logit':
            p=expit((weights*risk).sum(1))
            deriv=p[:,None]*(1-p[:,None])*weights
            return p,deriv,None
        state=expit(risk)
        p=(weights*state).sum(1)
        return p,weights*state*(1-state),weights*(state-p[:,None])
    xx,gg,lp=x[t],g[t],prior[t]
    def objective(w):
        p,dr,da=forward(w,xx,gg,lp)
        p=p.clip(1e-10,1-1e-10)
        r=(p-y[t])/(p*(1-p))
        gradient=np.einsum('nsd,ns,n->sd',xx,dr,r).ravel()
        if gating:
            gradient=np.r_[gradient,np.einsum('nsd,ns,n->sd',gg,da,r).ravel()]
        penalty=np.r_[np.full(split,2.),np.full(size-split,8.)]
        value=-np.where(y[t],np.log(p),np.log1p(-p)).sum()+.5*(penalty*w)@w
        return value,gradient+penalty*w
    result=minimize(objective,np.zeros(size),method='L-BFGS-B',jac=True,options={'maxiter':450,'ftol':1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    predicted=forward(result.x,x[q],g[q],prior[q])[0]
    return (predicted,result.x) if return_parameters else predicted


def run_fold(frame,groups,y,t,q):
    x,g,prior=design(frame)
    out={k:fit(x,g,prior,y,t,q,k) for k in ('pooled_logit','fixed','learned')}
    return out,dict(unique_explanation=int((np.isfinite(prior[q]).sum(1)==1).sum()),passengers=len(q))


if __name__=='__main__':
    with threadpool_limits(1):
        evaluate('service_bag',run_fold,REPORT)
