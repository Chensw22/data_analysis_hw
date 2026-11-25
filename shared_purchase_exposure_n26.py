"""Compound-Poisson receipt likelihoods with a shared or service-specific latent opportunity to spend."""
import json
import numpy as np
from scipy.special import expit,logit,logsumexp,ive
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from distinct_challenger_error_reach_n26 import population,ROOT,FEES
from exaone_validation_data import compact_features
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker

EXPOSURE=np.array([.2,.5,1.,1.8,3.])
EXPOSURE/=EXPOSURE.mean()


def moments(values,target=None):
    n=np.isfinite(values).sum(0)
    z=(values==0).sum(0)
    s=np.nansum(values,axis=0)
    old0,oldmean=(np.full(5,.65),np.full(5,220.)) if target is None else target
    return (z+25*old0)/(n+25),(s+25*oldmean)/(n+25)


def rate_scale(zero,mean):
    # E exp(-exposure * rate) equals the observed zero mass; E exposure is one.
    lo=np.zeros(5);hi=np.full(5,60.)
    for _ in range(42):
        mid=(lo+hi)/2
        high=np.exp(-mid[:,None]*EXPOSURE).mean(1)<zero
        hi=np.where(high,mid,hi);lo=np.where(high,lo,mid)
    rate=(lo+hi)/2
    return rate,(mean/rate).clip(.1)


def evidence(x,params):
    rate,scale=rate_scale(*params)
    lam=rate[None,:,None]*EXPOSURE
    dollars=np.maximum(np.nan_to_num(x),1)[:,:,None]
    theta=scale[None,:,None]
    bessel_arg=2*np.sqrt(lam*dollars/theta)
    positive=-lam-dollars/theta+.5*np.log(lam/(dollars*theta))+bessel_arg+np.log(ive(1,bessel_arg))
    density=np.where(x[:,:,None]>0,positive,-lam)
    occurrence=np.where(x[:,:,None]>0,np.log(-np.expm1(-lam)),-lam)
    def likelihood(block):
        block=np.where(np.isfinite(x[:,:,None]),block,0.)
        independent=(logsumexp(block,axis=2)-np.log(len(EXPOSURE))).sum(1)
        joint=logsumexp(block.sum(1),axis=1)-np.log(len(EXPOSURE))
        return independent,joint
    return np.column_stack([*likelihood(occurrence),*likelihood(density)])


def fitted_rates(x,t,q,y,coarse,fine):
    out=np.zeros((len(q),4,2))
    for label in (False,True):
        bank=t[y[t]==label]
        global_prior=moments(x[bank])
        for c in np.unique(coarse[q]):
            m=coarse[q]==c
            prior=moments(x[bank[coarse[bank]==c]],global_prior)
            for f in np.unique(fine[q[m]]):
                sub=m&(fine[q]==f)
                close=bank[(coarse[bank]==c)&(fine[bank]==f)]
                out[sub,:,int(label)]=evidence(x[q[sub]],moments(x[close],prior))
    return out


def run():
    raw,y,g,fold,parent,same,count,_,_,wrong,solo,paid=population()
    from arbitration_against_scored_nine_n26 import champions
    valid=champions()[1]
    full=compact_features(raw)
    full['size']=raw.groupby(g).PassengerId.transform('size')
    reg=full.drop(columns=FEES)
    x=raw[FEES].to_numpy()
    coarse=(full.CryoSleep+'|'+full.HomePlanet).to_numpy()
    fine=(full.Deck+'|'+full.Destination).to_numpy()
    post=np.zeros((len(y),6))
    density=np.zeros((len(y),4,2))
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        density[q]=fitted_rates(x,t,q,y,coarse,fine)
        for j,frame in enumerate([full,reg]):
            post[q,j]=fit_model(frame.iloc[t],y[t]).predict_proba(frame.iloc[q])[:,1]
        post[q,2:]=expit(logit(post[q,1].clip(.001,.999))[:,None]+density[q,:,1]-density[q,:,0])
        print(k,flush=True)
    names=['raw_tree','registration','independent_presence','shared_presence','independent_amount','shared_amount']
    predictions=dict(zip(names,post.T,strict=True))
    # Learn the value of shared exposure after conditioning on both the complete ordinary
    # classifier and occurrence likelihoods. All meta-training/calibration bookings are in
    # the same foundation fold and remain distinct from meta-test bookings.
    odds=(density[:,:,1]-density[:,:,0]).clip(-24,24)
    base=np.column_stack([logit(post[:,:2].clip(.001,.999)),odds[:,[0,2]]])
    mask_shift=odds[:,1]-odds[:,0]
    all_shift=odds[:,3]-odds[:,2]
    designs={'marginal':base,'occurrence':np.column_stack([base,mask_shift]),
             'full_exposure':np.column_stack([base,mask_shift,all_shift-mask_shift])}
    cal={n:np.zeros(len(y)) for n in designs}
    for k in np.unique(fold):
        here=np.flatnonzero(fold==k)
        for a,b in GroupKFold(3,shuffle=True,random_state=671).split(here,groups=g[here]):
            t,q=here[a],here[b]
            assert not np.isin(g[t],g[q]).any()
            for n,xx in designs.items():
                scaler=StandardScaler().fit(xx[t])
                model=LogisticRegression(C=.3,max_iter=500).fit(scaler.transform(xx[t]),y[t])
                cal[n][q]=model.predict_proba(scaler.transform(xx[q]))[:,1]
    predictions.update(cal)
    awake=raw.CryoSleep.eq(False).to_numpy()&paid
    blind=same&(count==0)
    scopes=dict(all=valid,common=same,awake_paid=valid&awake,awake_paid_solo=valid&awake&solo,
                blind=blind,blind_awake_paid_negative=blind&awake&~parent)
    assert (blind&wrong).sum()==172 and (scopes['blind_awake_paid_negative']&wrong).sum()==127
    actions={n:(p>=.5)!=parent for n,p in predictions.items()}
    actions['joint_amount_extra']=actions['shared_amount']&~actions['independent_amount']
    actions['calibrated_extra']=actions['full_exposure']&~actions['occurrence']
    report={n:{s:ticker(m,wrong,act) for s,m in scopes.items()} for n,act in actions.items()}
    report['observed_class_mean_log_likelihood']={k:density[m,:,y[m].astype(int)].mean(0).tolist() for k,m in scopes.items()}
    from sklearn.metrics import roc_auc_score,log_loss
    report['awake_paid_metrics']={n:[float(roc_auc_score(y[awake&valid],p[awake&valid])),float(log_loss(y[awake&valid],p[awake&valid]))] for n,p in predictions.items()}
    path=ROOT/'reports/nov23_matched_learning_curve/shared_purchase_exposure.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    folder=ROOT/'artifacts/nov26_shared_purchase_exposure'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'scores.npz',ids=raw.PassengerId.to_numpy(str),fold=fold,y=y,**predictions)
    print({n:[(report[n][s]['fixed'],report[n][s]['harmed']) for s in scopes] for n in actions},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
