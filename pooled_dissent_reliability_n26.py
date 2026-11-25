"""Learn a single passenger error likelihood that pools contributions from opposing experts."""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from held_expert_portfolios_n26 import sources,outcome_scores,panel_predictions
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import order_splits

REPORT=RESULT.parent/'pooled_dissent_reliability.json'


def evidence(activ,probs,parent):
    row,expert=np.nonzero(activ)
    count=activ.sum(1)
    assert (count>0).all()
    signed=np.where(parent[row],-1,1)*logit(probs[row,expert].clip(.001,.999))
    assert signed.min()>-1.e-8
    return row,expert,signed,count


def score(theta,x,packet,kind):
    row,j,s,count=packet
    n,d=x.shape
    logit_x=x@theta[:d]
    if kind=='passenger_only':
        return logit_x,None
    ids=theta[d:-1]
    z=ids[j]+s*theta[-1]
    if kind=='mean':
        contrib=np.bincount(row,weights=z,minlength=n)/count
        slope=1/count[row]
    else:
        maximum=np.full(n,-np.inf)
        np.maximum.at(maximum,row,z)
        mass=np.exp(z-maximum[row])
        total=np.bincount(row,weights=mass,minlength=n)
        contrib=maximum+np.log(total/count)
        slope=mass/total[row]
    return logit_x+contrib,slope


def loss(theta,x,packet,y,kind):
    # Every passenger contributes one observation irrespective of how many models dissent.
    pred,slope=score(theta,x,packet,kind)
    residual=expit(pred)-y
    penalty=np.full(len(theta),6.)
    penalty[0]=0.
    val=(np.logaddexp(0,pred)-y*pred).sum()+.5*(penalty*theta)@theta
    grad=x.T@residual
    if slope is not None:
        row,j,s,_=packet
        a=residual[row]*slope
        grad=np.r_[grad,np.bincount(j,weights=a,minlength=len(theta)-x.shape[1]-1),a@s]
    return val,grad+penalty*theta


def run():
    panels=sources()[-1]
    probs={k:outcome_scores(v,panel_predictions(v,k)) for k,v in panels.items()}
    outcomes={}
    for b,origin,it,iq,tr_scope,scope,fold,xt,xq in order_splits():
        vt,vq=xt[:,1:],xq[:,1:]
        t=tr_scope&vt.any(1)
        q=scope&vq.any(1)
        predictions={name:np.zeros(len(scope),bool) for name in ('passenger_only','mean','log_mean_exp')}
        if q.any() and t.any():
            old=b.train_cell[t]%2!=0
            now=b.query_cell[q]%2!=0
            train=evidence(vt[t],probs[origin][it[t]],old)
            query=evidence(vq[q],probs['full'][iq[q]],now)
            scaler=StandardScaler().fit(b.train_fields[t])
            def inputs(z):
                return np.column_stack([np.ones(len(z)),scaler.transform(z).clip(-5,5)])
            x=inputs(b.train_fields[t])
            h=inputs(b.query_fields[q])
            y=b.train_error[t]
            for kind,pred in predictions.items():
                length=x.shape[1]+(0 if kind=='passenger_only' else vt.shape[1]+1)
                start=np.zeros(length)
                start[0]=logit((y.sum()+1)/(len(y)+2))
                opt=minimize(loss,start,args=(x,train,y,kind),jac=True,method='L-BFGS-B',
                             options={'maxiter':300,'ftol':1.e-11})
                if not opt.success:
                    raise RuntimeError(opt.message)
                pred[q]=score(opt.x,h,query,kind)[0]>0
        outcomes.setdefault(b.tag,[]).append((scope,b.query_error,fold,predictions,q))
    report={}
    for tag,items in outcomes.items():
        scope,wrong,fold=(np.concatenate([r[k] for r in items]) for k in range(3))
        active=np.concatenate([r[4] for r in items])
        report[tag]=dict(offered_dissent=int(active.sum()),errors_with_dissent=int((active&wrong).sum()))
        for name in items[0][3]:
            actions=np.concatenate([r[3][name] for r in items])
            r=ticker(scope,wrong,actions)
            r['net_by_fold']=[int((np.where(wrong,1,-1)*actions*(fold==k)).sum()) for k in range(8)]
            report[tag][name]=r
        print(tag,report[tag],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
