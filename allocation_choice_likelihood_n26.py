"""Condition on the exact sorted receipt and active services, then model which service gets each charge."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit,logsumexp
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import projection
from exaone_validation_data import compact_features
from price_implied_deck_n26 import FEES
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker
from full_roster_booking_validation_n23 import ROOT

def records(raw,shuffled):
    amounts=raw[FEES].to_numpy()
    rng=np.random.default_rng(4862)
    registry=pd.get_dummies(raw[['HomePlanet','Destination']].astype('string').fillna('?'),dtype=float)
    deck=pd.get_dummies(raw.Cabin.str.split('/').str[0].fillna('?'),dtype=float)
    context=np.column_stack([np.ones(len(raw)),registry,deck,raw.Age.fillna(28)/35,raw.CryoSleep.eq(True).fillna(False)])
    rows=[]; options=[]; chosen=[]; attrs=[]
    for i,values in enumerate(amounts):
        active=np.flatnonzero(values>0)
        if len(active)<2 or not np.isfinite(values).all():
            continue
        ordered=active[np.argsort(-values[active],kind='stable')]
        magnitudes=values[ordered].copy()
        if shuffled:
            ordered=rng.permutation(ordered)
        avail=np.zeros(5,bool);avail[active]=True
        for j,amount in enumerate(magnitudes[:-1]):
            if (j>0 and amount==magnitudes[j-1]) or amount==magnitudes[j+1]:
                avail[ordered[j]]=False
                continue
            rows.append(i);options.append(avail.copy());chosen.append(ordered[j])
            attrs.append(np.r_[context[i],np.log1p(magnitudes.sum())/8,amount/magnitudes.sum(),
                               np.log((1+amount)/(1+magnitudes[j+1])),j/4,len(active)/5])
            avail[ordered[j]]=False
    return np.array(rows),np.array(options),np.array(chosen),np.array(attrs,float)


def log_probability(data,y,t,q):
    rows,options,chosen,attrs=data
    ask=np.flatnonzero(np.isin(rows,q))
    pick=chosen[ask]
    obs=[]
    for label in (False,True):
        fit=np.isin(rows,t)&(y[rows]==label)
        x,available,target=attrs[fit],options[fit],chosen[fit]
        def loss(parameters):
            w=parameters.reshape(x.shape[1],5)
            value=np.where(available,x@w,-np.inf)
            normal=logsumexp(value,axis=1)
            error=np.exp(value-normal[:,None])
            error[np.arange(len(x)),target]-=1.
            obj=(normal-value[np.arange(len(x)),target]).sum()+2.*(w*w).sum()
            return obj,(x.T@error+4*w).ravel()
        sol=minimize(loss,np.zeros(x.shape[1]*5),jac=True,method='L-BFGS-B',options={'maxiter':400})
        assert sol.success
        logits=np.where(options[ask],attrs[ask]@sol.x.reshape(-1,5),-np.inf)
        prob=logits[np.arange(len(ask)),pick]-logsumexp(logits,axis=1)
        obs.append(np.bincount(rows[ask],weights=prob,minlength=len(y))[q])
    return np.column_stack(obs)


def run():
    raw,y,g,fold,views,_=projection(include_full=True)
    nine,known,_,_=champions()
    actual,sham=[records(raw,flag) for flag in [False,True]]
    np.testing.assert_array_equal(actual[0],sham[0])
    x=compact_features(raw)
    x['party_size']=raw.groupby(g).PassengerId.transform('size')
    blind=x.copy()
    ranked=np.sort(raw[FEES].fillna(-1).to_numpy(),axis=1)
    for j,col in enumerate(FEES):
        blind[col]=np.sign(raw[col].fillna(-1))
        blind[f'sorted_charge_{j}']=ranked[:,j]
    scores=np.zeros((len(y),4))
    likelihood=np.zeros((len(y),2))
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        base=fit_model(blind.iloc[t],y[t]).predict_proba(blind.iloc[q])[:,1]
        direct=fit_model(x.iloc[t],y[t]).predict_proba(x.iloc[q])[:,1]
        scores[q,:2]=np.column_stack([base,direct])
        for j,data in enumerate([actual,sham]):
            logs=log_probability(data,y,t,q)
            scores[q,j+2]=expit(logit(base.clip(.001,.999))+logs[:,1]-logs[:,0])
            likelihood[q,j]=logs[np.arange(len(q)),y[q].astype(int)]
        print(k,flush=True)
    folder=ROOT/'artifacts/nov26_allocation_choice_likelihood'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'scores.npz',passenger=raw.PassengerId.to_numpy(str),fold=fold,y=y,
                        scores=scores,assignment_log_probability=likelihood)
    report={}
    for tag in ('full','bookingmix','historical_to_new'):
        mask,parent=views[tag]
        assert known[mask].all()
        same=parent==nine
        wrong=parent!=y
        oppose=(scores>=.5)!=parent[:,None]
        actions={**dict(zip(['unordered','ordinary','allocation','permuted'],oppose.T,strict=True)),
            'allocation_extra':oppose[:,2]&~oppose[:,0],
            'permuted_extra':oppose[:,3]&~oppose[:,0],
            'exclusive':oppose[:,2]&~oppose[:,0]&~oppose[:,3],
            'confirmed_extra':oppose[:,2]&~oppose[:,0]&oppose[:,1]}
        scopes={'all':mask,'both_agree':mask&same,'both_negative':mask&same&~parent,'both_positive':mask&same&parent}
        result={name:{label:ticker(m,wrong,action) for label,m in scopes.items()} for name,action in actions.items()}
        evaluated=int(np.isin(actual[0],np.flatnonzero(mask)).sum())
        result['choice_steps']=evaluated
        informative=mask&np.isin(np.arange(len(y)),actual[0])
        result['eligible_rows']=int(informative.sum())
        result['eligible_shared_errors']=int((informative&same&wrong).sum())
        result['choice_nll']=( -likelihood[mask].sum(0)/evaluated).tolist()
        result['outcome_nll']=[float(log_loss(y[mask],p[mask])) for p in scores.T]
        report[tag]=result
        print(tag,[(n,r['both_agree']['fixed'],r['both_agree']['harmed']) for n,r in result.items() if isinstance(r,dict)],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/allocation_choice_likelihood.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
