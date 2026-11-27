"""Fit partial-membership receipt profiles and score each service using the other four."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL,compact_features
from matched_state_context_n26 import ROOT, inputs, FEES
from repeated_archive_misses_n26 import ART,cohort,count

OUT=ART/'nov27_solo_mixed_receipt_topics'
REPORT=ROOT/'reports/nov23_matched_learning_curve/solo_mixed_receipt_topics.json'
K=5
LEVELS=7


def encode(b,t):
    codes=np.zeros(b.shape,int)
    for j in range(b.shape[1]):
        positive=b[t,j];positive=positive[positive>0]
        cuts=np.quantile(positive,[1/6,2/6,3/6,4/6,5/6])
        codes[:,j]=np.where(b[:,j]>0,1+np.searchsorted(cuts,b[:,j]),0)
        codes[~np.isfinite(b[:,j]),j]=-1
    return codes


def strengths(beta,codes,excluded=None,initial=None,steps=8):
    theta=np.full((len(codes),K),1/K) if initial is None else initial.copy()
    probs=np.stack([beta[j,:,codes[:,j].clip(0)] for j in range(5)],axis=1)
    seen=(codes>=0).astype(float)
    if excluded is not None:seen[:,excluded]=0.
    for _ in range(steps):
        joint=theta[:,None,:]*probs
        responsibilities=joint/joint.sum(2,keepdims=True)
        counts=(responsibilities*seen[:,:,None]).sum(1)+.45
        theta=counts/counts.sum(1,keepdims=True)
    return theta,responsibilities


def profiles(codes,seed):
    rng=np.random.default_rng(seed)
    base=np.stack([np.bincount(codes[codes[:,j]>=0,j],minlength=LEVELS)+1 for j in range(5)])
    prior=base/base.sum(1,keepdims=True)
    beta=prior[:,None,:]*rng.lognormal(0,.6,(5,K,LEVELS))
    beta/=beta.sum(2,keepdims=True)
    theta=None
    for _ in range(38):
        theta,r=strengths(beta,codes,initial=theta,steps=2)
        mass=np.empty_like(beta)
        for j in range(5):
            seen=codes[:,j]>=0
            for k in range(K):
                mass[j,k]=np.bincount(codes[seen,j],weights=r[seen,j,k],minlength=LEVELS)+8*prior[j]
        beta=mass/mass.sum(2,keepdims=True)
    return beta,prior


def measured(beta,prior,codes):
    features=[];losses=[]
    for j in range(5):
        theta,_=strengths(beta,codes,excluded=j)
        p=theta@beta[j]
        index=codes[:,j].clip(0)
        observed=p[np.arange(len(p)),index]
        ordinary=prior[j,index]
        previous=p.cumsum(1)[np.arange(len(p)),index]-observed
        terms=np.column_stack([np.log(observed/ordinary),previous+.5*observed,1-p[:,0]])
        terms[codes[:,j]<0,:2]=np.nan
        features.append(terms)
        losses.append(np.where(codes[:,j]>=0,-np.log(observed),np.nan))
    return np.column_stack(features),np.column_stack(losses)


def fit_classifier(x,y,t,q):
    model=HistGradientBoostingClassifier(max_iter=155,learning_rate=.065,max_leaf_nodes=20,l2_regularization=5.,
                                        min_samples_leaf=24,early_stopping=False,random_state=494)
    return model.fit(x[t],y[t]).predict_proba(x[q])[:,1]


def run():
    frame,y,g,fold,states=inputs(); n=len(y); ids=frame.PassengerId.to_numpy(str)
    _,blind,known,fresh=cohort(frame.iloc[:n],y,fold)
    income=frame[FEES].to_numpy(float)
    amount=np.nansum(income,axis=1); paid=amount>0; count_used=(income>0).sum(1)
    group_size=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    solo=group_size==1
    cats=compact_features(frame)
    x=np.column_stack([pd.get_dummies(cats,columns=CATEGORICAL,dtype=float).to_numpy(),group_size,np.log1p(amount),count_used])
    with np.load(ART/'nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        np.testing.assert_array_equal(z['y'],y)
        anchor=z['parent'];opin=z['opinions'];ordered=z['ordered_fold']
    agree=((opin>=.5)==anchor[:,None]).all(1)
    native=opin[:,4]
    odds=lambda v:logit(np.clip(v,.002,.998))
    scopes=dict(all=known,solo=known&solo[:n],paid_solo=known&solo[:n]&paid[:n],
                common_solo=known&solo[:n]&agree[:n],fresh=known&fresh,fresh_solo=known&fresh&solo[:n],harder=blind)
    wrong=anchor[:n]!=y
    assert (scopes['common_solo']&wrong).sum()==525 and (scopes['common_solo']&wrong&paid[:n]).sum()==329
    OUT.mkdir(exist_ok=True)
    report={'coverage':{k:dict(rows=int(m.sum()),errors=int((m&wrong).sum())) for k,m in scopes.items()},
            'method':'Five mixed-membership service profiles per passenger, holding each facility out when scoring its registered value; six positive-amount quantiles and zero per service. No outcome or numerical identifier enters the profile learning.',
            'control':'Shuffle each service independently within training origin, payment count, total spending stratum and solo status, preserving its within-stratum marginal distribution.',
            'previous':['full_roster_latent_profiles_n24.py','solo_selection_likelihood_n26.py','conditional_basket_energy_n26.py','solo_conditional_splines_n25.py'],
            'protocols':{}}
    saved={}
    for scheme,f in [('random',fold),('contiguous',ordered)]:
        p=np.full((len(ids),3),np.nan)
        nll=np.full((len(ids),2),np.nan)
        for j in range(9):
            t=np.flatnonzero((f!=j)&paid[:n]); q=np.flatnonzero((f==j)&paid[:n]) if j<8 else n+np.flatnonzero(paid[n:])
            assert not np.isin(g[t],g[q]).any()
            path=OUT/(f'{scheme}_{j}.npz' if j<8 else 'test.npz')
            if not path.exists():
                codes=encode(income,t)
                loose=codes[t].copy()
                level=np.digitize(amount[t],np.quantile(amount[t],[.25,.5,.75]))
                state=pd.DataFrame({'level':level,'used':count_used[t],'origin':frame.HomePlanet.fillna('?').iloc[t].to_numpy(),'solo':solo[t]})
                rng=np.random.default_rng(895)
                for idx in state.groupby(list(state),sort=False).indices.values():
                    for col in range(5):loose[idx,col]=rng.permutation(loose[idx,col])
                rows=np.r_[t,q]
                result=[fit_classifier(x,y,t,q)]
                scores=[]
                for patterns in [codes[t],loose]:
                    prof,prior=profiles(patterns,776)
                    mat,loss=measured(prof,prior,codes[rows])
                    features=np.column_stack([x[rows],mat])
                    # Reindex the same true training/query passengers for both profile fits.
                    result.append(fit_classifier(features,np.r_[y[t],np.zeros(len(q),bool)],np.arange(len(t)),len(t)+np.arange(len(q))))
                    scores.append(np.nanmean(loss[len(t):],axis=1))
                np.savez_compressed(path,t=t,q=q,labels=y[t],ids=ids[q],p=np.column_stack(result),nll=np.column_stack(scores))
            with np.load(path) as z:
                for field,val in [('t',t),('q',q),('labels',y[t]),('ids',ids[q])]:
                    np.testing.assert_array_equal(z[field],val)
                p[q]=z['p'];nll[q]=z['nll']
            print(scheme,j,flush=True)
        np.testing.assert_array_equal(np.isfinite(p[:,0]),paid)
        baseline=(p[:,0]>=.5)!=anchor; true=(p[:,1]>=.5)!=anchor; sham=(p[:,2]>=.5)!=anchor
        change=odds(p[:,1])-odds(p[:,0])
        noise=odds(p[:,2])-odds(p[:,0])
        direction=np.where(anchor,-1.,1.)
        real_gain=change*direction>abs(odds(native))
        fake_gain=noise*direction>abs(odds(native))
        proposals={'baseline':paid&baseline,'profiles':paid&true,'shuffled':paid&sham,
                   'profiles_only':paid&true&~baseline&~sham,'new_direction':paid&true&~baseline,
                   'contrast_only':paid&real_gain&~fake_gain,'confirmed_contrast':paid&real_gain&~fake_gain&true}
        proposals.update({k+'_solo':a&solo for k,a in list(proposals.items()) if k in ['profiles_only','confirmed_contrast']})
        table={}
        for name,a in proposals.items():
            d=a[:n]*np.where(wrong,1,-1)
            table[name]=dict(scopes={k:count(a[:n],m,wrong) for k,m in scopes.items()},test=int(a[n:].sum()),
                common_test=int((a[n:]&agree[n:]&solo[n:]).sum()),fold_net=[int(d[known&(f==k)].sum()) for k in range(8)])
        report['protocols'][scheme]={'standalone':{},'rules':table}
        for name,m in [('all_paid',paid[:n]),('paid_solo',paid[:n]&solo[:n]),('solo_common',paid[:n]&scopes['common_solo'])]:
            acc=((p[:n][m]>=.5)==y[m,None]).mean(0)
            aux=nll[:n][m].mean(0)
            report['protocols'][scheme]['standalone'][name]=dict(rows=int(m.sum()),baseline_true_shuffled_accuracy=acc.tolist(),true_shuffled_leave_one_service_nll=aux.tolist())
        saved[scheme]=p
        print(report['protocols'][scheme]['standalone'],flush=True)
    np.testing.assert_array_equal(saved['random'][n:],saved['contiguous'][n:])
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,fold=fold,ordered=ordered,parent=anchor,y=y,**saved)
    prod=pd.DataFrame({'PassengerId':ids[n:],'paid':paid[n:],'solo':solo[n:]})
    parent_path=ART/'proposal_n27_directional_foundation/test_probabilities.csv'
    parent=pd.read_csv(parent_path)
    np.testing.assert_array_equal(parent.PassengerId,ids[n:])
    np.testing.assert_array_equal(parent.probability>=.5,anchor[n:])
    for k,v in zip(['baseline','profiles','shuffled'],saved['random'][n:].T):prod[k]=np.where(paid[n:],v,parent.probability)
    prod.to_csv(OUT/'test_probabilities.csv',index=False)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report['protocols'].items():print(k,{w:r['scopes']['common_solo'] for w,r in v['rules'].items()})

if __name__=='__main__':
    with threadpool_limits(1):run()
