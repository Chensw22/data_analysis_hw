"""Learn the change in transport odds associated with a bill's departure from matched registration peers."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import compact_features
from family_identity_transfer_n26 import fit, panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
OUT=ART/'nov26_conditional_bill_deviations'


def design(raw):
    frame=compact_features(raw)
    g=raw.PassengerId.str[:4]
    context=frame.drop(columns=BILLS).copy()
    context['party_size']=np.log(g.map(g.value_counts()))
    context['Age']=context.Age.fillna(-10)/30
    context['CabinNumber']=context.CabinNumber.fillna(-100)/600
    z=pd.get_dummies(context,dtype=float).to_numpy()
    amount=frame[BILLS].to_numpy(float)
    known=np.nan_to_num(amount)
    sorted_amount=np.sort(known,axis=1)
    amounts=np.column_stack([known,known.sum(1), sorted_amount[:,-1],
                             (known[:,0]+known[:,3]+known[:,4]), known[:,1]+known[:,2]])
    log_amount=np.log1p(amounts)
    bills=[np.isnan(amount),amounts>0,log_amount/6,known/(1+known.sum(1)[:,None])]
    for cut in [75,400,1000,2400]:
        bills.append(np.maximum(log_amount-np.log1p(cut),0)/2)
    for j in range(5):
        for k in range(j):
            bills.append((np.minimum(log_amount[:,j],log_amount[:,k])/6)[:,None])
    phi=np.column_stack(bills)
    # These strata allow a bill to have a different association for a sleeping
    # passenger or on a different home planet. Coefficients have a shared component.
    strata=pd.get_dummies(frame[['HomePlanet','CryoSleep']],dtype=float).to_numpy()
    strata=np.column_stack([(g.map(g.value_counts()).to_numpy()>1),strata])
    return z,phi,strata


def expected(z,phi,y,g,t,q,seed):
    assert not np.isin(g[t],g[q]).any()
    model=CatBoostClassifier(iterations=280,depth=5,learning_rate=.065,l2_leaf_reg=6,
                             random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    model.fit(z[t],y[t])
    prior=logit(model.predict_proba(z[q])[:,1].clip(.002,.998))
    distance,index=cKDTree(z[t]).query(z[q],k=min(64,len(t)))
    weights=np.exp(-2*(distance/distance[:,-1,None].clip(.001))**2)
    weights/=weights.sum(axis=1,keepdims=True)
    donors=t[index]
    mean=np.einsum('ij,ijk->ik',weights,phi[donors],optimize=True)
    # A local replacement uses only one independent registration neighbour's
    # entire bill, preserving service co-occurrence and missing fields.
    rng=np.random.default_rng(seed+3101)
    draw=(rng.random(len(q))[:,None]>np.cumsum(weights,axis=1)).sum(1).clip(max=weights.shape[1]-1)
    replacement=phi[donors[np.arange(len(q)),draw]]
    return prior,mean,replacement


def interact(bill,strata):
    return np.column_stack([bill,*[bill*col[:,None] for col in strata.T]])


def fit_fold(z,phi,strata,y,g,t,q,seed):
    ordered=np.r_[t,q]
    prior=np.empty(len(ordered))
    mean=np.empty_like(phi[ordered])
    swap=mean.copy()
    for tr,va in GroupKFold(3,shuffle=True,random_state=seed).split(t,groups=g[t]):
        prior[va],mean[va],swap[va]=expected(z,phi,y,g,t[tr],t[va],seed)
    test=np.arange(len(t),len(ordered))
    prior[test],mean[test],swap[test]=expected(z,phi,y,g,t,q,seed)
    basic=phi[ordered]
    predictions={'prior':prior[test]}
    for name,part in [('raw',basic),('deviation',basic-mean),('local_sham',swap-mean)]:
        feature=np.column_stack([np.ones(len(ordered)),prior,interact(part,strata[ordered])])
        beta=fit(feature[:len(t)],prior[:len(t)],y[t])
        predictions[name]=prior[test]+feature[test]@beta
        if name=='deviation':
            displaced=np.column_stack([np.ones(len(test)),prior[test],interact(swap[test]-mean[test],strata[q])])
            predictions['same_fit_other_bill']=prior[test]+displaced@beta
    return predictions


def run():
    raw,y,g,f=data()
    z,phi,strata=design(raw)
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as file:
        np.testing.assert_array_equal(file['original_prior'],ref[0])
        ref[0]=file['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as file:
        hf,old,par=file['fold'],file['nine'],file['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    OUT.mkdir(exist_ok=True)
    result={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(raw)),f,native,current)]:
        clean=~np.isin(g[ids],g[rows])
        flags={}
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k)
            q=ids[loc]
            others=ids[fold!=k]
            t=others[~np.isin(g[others],g[q])]
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[others])
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                probs=fit_fold(z,phi,strata,y,g,t,q,4329+int(k))
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle,t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),**probs)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as file:
                for key,val in [('t',t),('q',q),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(file[key],val)
                probs={key:file[key] for key in ['prior','raw','deviation','local_sham','same_fit_other_bill']}
            off=logit(nat[loc].clip(.002,.998))
            for name,estimate in probs.items():
                for w in [1.,.25]:
                    flip=((1-w)*nat[loc]+w*expit(estimate)>=.5)!=(off>=0)
                    flags.setdefault(f'{name}_{w}',np.zeros(len(ids),bool))[loc]=flip
            basic=np.column_stack([np.ones(len(q)),off,probs['prior']-off,probs['raw']-off])
            schemes={'raw':basic,'deviation':np.column_stack([basic,probs['deviation']-probs['raw']]),
                     'sham':np.column_stack([basic,probs['local_sham']-probs['prior']])}
            for protocol in ['all','new']:
                if tag=='historical' and protocol=='new':
                    continue
                batches=([(np.flatnonzero(~clean[loc]),np.flatnonzero(clean[loc]))] if protocol=='new' else
                         GroupKFold(4,shuffle=True,random_state=7615).split(q,groups=g[q]))
                for tr,te in batches:
                    assert not np.isin(g[q[tr]],g[q[te]]).any()
                    for name,features in schemes.items():
                        beta=fit(features[tr],off[tr],y[q[tr]])
                        estimate=off[te]+features[te]@beta
                        flags.setdefault(protocol+'_'+name,np.zeros(len(ids),bool))[loc[te]]=(estimate>=0)!=(off[te]>=0)
            print(tag,k,flush=True)
        wrong=view['anchor']!=y[ids]
        family=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        agreement=(nat>=.5)==view['anchor']
        summary={}
        for name,flag in flags.items():
            scores={}
            for scope,mask in [('all',view['known']),('new',view['known']&clean),('families',family),('new_families',family&clean)]:
                if name.startswith('new_') and not scope.startswith('new'):
                    continue
                edit=flag&agreement&mask
                scores[scope]=dict(rows=int(mask.sum()),fixed=int((edit&wrong).sum()),harmed=int((edit&~wrong).sum()),
                    net_by_fold=[int((edit&wrong&(fold==k)).sum())-int((edit&~wrong&(fold==k)).sum()) for k in np.unique(fold)])
            summary[name]=scores
        result[tag]=summary
        print([(key,s.get('all',s['new'])['fixed'],s.get('all',s['new'])['harmed']) for key,s in summary.items()],flush=True)
    DEST.with_name('conditional_bill_deviations.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
