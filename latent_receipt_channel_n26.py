"""Model billed amounts as draws from shared basket templates passed through a label-dependent recording channel."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier

from exaone_validation_data import compact_features, CATEGORICAL
from disjoint_tree_fields_n25 import ART,DEST
from full_roster_booking_validation_n23 import data
from family_identity_transfer_n26 import fit,panels_and_reference
from outstanding_error_families_n26 import observed

BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
K,B=12,5
OUT=ART/'nov26_latent_receipt_channel'


def bin_amounts(frame,t):
    z=np.zeros((len(frame),len(BILLS)),int)
    for j,col in enumerate(BILLS):
        v=frame[col].to_numpy()
        cuts=np.quantile(v[t][v[t]>0],[.25,.50,.75])
        z[:,j]=np.where(np.isnan(v),-1,np.where(v==0,0,1+np.searchsorted(cuts,v)))
    return z


def template_logp(theta,z):
    result=np.zeros((len(z),len(theta)))
    for j,col in enumerate(z.T):
        ok=col>=0
        result[ok]+=np.log(theta[:,j,:][:,col[ok]].T)
    return result


def templates(z):
    rng=np.random.default_rng(6432)
    # Start near actual baskets, including missing values as uniform emissions.
    init=z[rng.choice(len(z),K,replace=False)]
    theta=np.full((K,5,B),.3)
    for i,j in zip(*np.where(init>=0)):
        theta[i,j,init[i,j]]+=1
    theta/=theta.sum(2,keepdims=True)
    weights=np.full(K,1/K)
    for _ in range(28):
        r=softmax(template_logp(theta,z)+np.log(weights),axis=1)
        weights=(1+r.sum(0))/(len(z)+K)
        for j in range(5):
            theta[:,j,:]=.5+r.T@(z[:,j,None]==np.arange(B))
        theta/=theta.sum(2,keepdims=True)
    return theta,softmax(template_logp(theta,z)+np.log(weights),axis=1)


def channels(theta):
    # Conditional on the recording mode, each specified service is independently
    # erased or shrunk by one positive bin with probability 0.65. Missing is unobserved.
    out=[theta]
    for subset in [[j] for j in range(5)]+[[0,3,4],[1,2],list(range(5))]:
        for erase in (True,False):
            new=theta.copy()
            for j in subset:
                new[:,j,:]=.35*theta[:,j,:]
                for v in range(B):
                    dst=0 if erase else max(0,v-1)
                    new[:,j,dst]+=.65*theta[:,j,v]
            out.append(new)
    return out


def mixture_weights(log_values):
    n=log_values.shape[1]
    weights=np.full(n,1/n)
    for _ in range(55):
        count=softmax(log_values+np.log(weights),axis=1).sum(0)
        weights=(count+3)/(len(count)*3+len(log_values))
    return weights


def fits(frame,x,t,q):
    z=bin_amounts(frame,t)
    # Learn shared templates and registration-dependent prevalence from half the orders.
    # The other half estimates supervised channel frequencies, avoiding in-sample priors.
    theta,r=templates(z[t])
    forest=ExtraTreesRegressor(n_estimators=72,min_samples_leaf=24,max_features=.9,random_state=523,n_jobs=2)
    forest.fit(x[t],r)
    registration=forest.predict(x[q]).clip(1e-7,1)
    registration/=registration.sum(1,keepdims=True)
    ll=np.stack([template_logp(param,z[q]) for param in channels(theta)],axis=2)
    return logsumexp(np.log(registration)[:,:,None]+ll,axis=1), ll[:,:,0]


def run():
    raw,y,g,f=data()
    frame=compact_features(raw)
    meta=frame.drop(columns=BILLS)
    x=pd.get_dummies(meta,dtype=float).fillna(-1).to_numpy()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z: hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    curr=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    OUT.mkdir(exist_ok=True)
    report={}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,curr)]:
        clean=~np.isin(g[ids],g[rows])
        flags={}
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k)
            q=ids[loc]
            t=ids[(fold!=k)&~np.isin(g[ids],g[q])]
            if tag=='historical': clean[loc]=~np.isin(g[q],g[ids[fold!=k]])
            assert not np.isin(g[q],g[t]).any()
            split=list(GroupKFold(2,shuffle=True,random_state=4372).split(t,groups=g[t]))
            scores={s:np.zeros(len(q)) for s in ('channel','placebo','template')}
            for a,b in split:
                a,b=t[a],t[b]
                assert not np.isin(g[a],g[b]).any()
                lc,lt=fits(frame,x,a,np.r_[b,q])
                label=y[b]
                placebo=np.random.default_rng(612+k).permutation(label)
                for s,values,target in [('channel',lc,label),('placebo',lc,placebo),('template',lt,label)]:
                    likelihood=[]
                    for state in (False,True):
                        weights=mixture_weights(values[:len(b)][target==state])
                        vals=logsumexp(values[len(b):]+np.log(weights),axis=1)
                        likelihood.append(vals)
                    scores[s]+=(likelihood[1]-likelihood[0])/2
            prior_model=CatBoostClassifier(iterations=330,depth=5,verbose=False,thread_count=2,random_seed=612,allow_writing_files=False)
            prior_model.fit(meta.iloc[t],y[t],cat_features=CATEGORICAL)
            prior=logit(prior_model.predict_proba(meta.iloc[q])[:,1].clip(.002,.998))
            offset=logit(base[loc].clip(.002,.998))
            core=np.column_stack([np.ones(len(q)),offset,prior-offset])
            for s,value in [('prior',np.zeros(len(q))),*scores.items()]:
                flags.setdefault(s+'_direct',np.zeros(len(ids),bool))[loc]=(prior+value>=0)!=(offset>=0)
                for a,b in GroupKFold(4,shuffle=True,random_state=2782).split(q,groups=g[q]):
                    xx=np.column_stack([core,value])
                    coef=fit(xx[a],offset[a],y[q[a]])
                    flags.setdefault(s+'_calibrated',np.zeros(len(ids),bool))[loc[b]]=(offset[b]+xx[b]@coef>=0)!=(offset[b]>=0)
            np.savez_compressed(OUT/f'{tag}_{k}.npz',q=q,t=t,train_y=y[t],prior=prior,**scores)
            print(tag,k,flush=True)
        hard=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        results={}
        for s,flag in flags.items():
            counts={}
            for name,mask in [('all',view['known']),('unseen',view['known']&clean),('family_agreement',hard),('unseen_agreement',hard&clean)]:
                edit=flag&mask&(view['anchor']==(base>=.5))
                wrong=view['anchor']!=y[ids]
                counts[name]=dict(rows=int(mask.sum()),fixed=int((edit&wrong).sum()),harmed=int((edit&~wrong).sum()))
            results[s]=counts
        report[tag]=results
        print(tag,{s:r['all'] for s,r in results.items()},flush=True)
    DEST.with_name('latent_receipt_channel.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
