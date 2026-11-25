"""Learn class-conditional passenger cabin choice probabilities normalized over feasible tracks."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import compact_features
from family_identity_transfer_n26 import fit, panels_and_reference
from full_roster_booking_validation_n23 import data,ROOT
from global_cabin_assignment_n26 import OUT as GEOMETRY
from outstanding_error_families_n26 import observed

OUT=ART/'nov26_constrained_track_odds'


def inputs(roster):
    records=json.loads((GEOMETRY/'missing.json').read_text())+json.loads((GEOMETRY/'masked_known.json').read_text())
    records.sort(key=lambda v:v['row'])
    np.testing.assert_array_equal([r['row'] for r in records],np.arange(len(roster)))
    frame=compact_features(roster)
    track=frame.Deck+frame.Side
    known=roster.Cabin.notna().to_numpy()
    labels=pd.Index(np.sort(track[known].unique()))
    obs=labels.get_indexer(track)
    allowed=np.zeros((len(frame),len(labels)),bool)
    for i,r in enumerate(records):
        allowed[i,labels.get_indexer(list(r['global_options']))]=True
    assert allowed[np.flatnonzero(known),obs[known]].all()
    plain=frame.drop(columns=['CabinNumber','Deck','Side']).copy()
    groups=roster.PassengerId.str[:4]
    plain['party']=np.log(groups.map(groups.value_counts()))
    plain['sequence']=groups.astype(int)/9280
    for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
        plain[c+'_positive']=(plain[c]>0).astype(float)
        plain[c]=np.log1p(plain[c])/6
    plain['Age']=plain.Age/35
    x=pd.get_dummies(plain,dtype=float).fillna(-1).to_numpy()
    x=np.column_stack([np.ones(len(x)),x])
    # The transport prior can see the menu itself, but no recorded cabin attribute.
    base=np.column_stack([x,allowed])
    blocks=(frame.HomePlanet+'|'+frame.CryoSleep).to_numpy()
    return x,base,allowed,obs,blocks


def score_menu(x,menu,observed,y,t,q,blocks,placebo,seed):
    # Singleton menus have zero likelihood gradient for every coefficient.
    selected=t[(observed[t]>=0)&(menu[t].sum(1)>1)]
    a=x[selected]
    choice=observed[selected]
    support=menu[selected]
    cond=y[selected].astype(float)-.5
    if placebo:
        rng=np.random.default_rng(seed)
        for key in np.unique(blocks[selected]):
            take=np.flatnonzero(blocks[selected]==key)
            cond[take]=rng.permutation(cond[take])
    d,k=x.shape[1],menu.shape[1]
    def objective(theta):
        main,interaction=theta.reshape(2,d,k)
        logits=a@main+cond[:,None]*(a@interaction)
        logits[~support]=-np.inf
        norm=logsumexp(logits,axis=1)
        loss=(norm-logits[np.arange(len(a)),choice]).sum()
        diff=np.exp(logits-norm[:,None])
        diff[np.arange(len(a)),choice]-=1.
        grad=np.stack([a.T@diff+4*main,a.T@(diff*cond[:,None])+12*interaction])
        return loss+2*np.square(main).sum()+6*np.square(interaction).sum(),grad.ravel()
    opt=minimize(objective,np.zeros(2*d*k),jac=True,method='L-BFGS-B',options={'maxiter':280,'ftol':1e-8})
    assert opt.success,opt.message
    main,interaction=opt.x.reshape(2,d,k)
    main=x[q]@main
    half=.5*(x[q]@interaction)
    possible=menu[q]
    def choice_log(prob):
        prob[~possible]=-np.inf
        norm=logsumexp(prob,axis=1)
        val=np.zeros(len(q))
        actual=observed[q]>=0
        val[actual]=prob[np.flatnonzero(actual),observed[q[actual]]]-norm[actual]
        return val
    llr=choice_log(main+half)-choice_log(main-half)
    np.testing.assert_allclose(llr[possible.sum(1)==1],0,atol=1e-10)
    assert np.isfinite(llr).all()
    return llr


def run():
    raw,y,g,f=data()
    roster=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())],ignore_index=True)
    x,masked,menu,code,blocks=inputs(roster)
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    OUT.mkdir(exist_ok=True)
    result={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,current)]:
        flags={}
        clean=~np.isin(g[ids],g[rows])
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k)
            q=ids[loc]
            others=ids[fold!=k]
            t=others[~np.isin(g[others],g[q])]
            assert not np.isin(g[q],g[t]).any()
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[others])
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                model=CatBoostClassifier(iterations=440,depth=6,learning_rate=.055,random_seed=6477,verbose=False,thread_count=2,allow_writing_files=False)
                model.fit(masked[t],y[t])
                prior=logit(model.predict_proba(masked[q])[:,1].clip(.002,.998))
                ratios=[score_menu(x,menu,code,y,t,q,blocks,shuffled,2664+int(k)) for shuffled in [False,True]]
                with path.with_suffix('.tmp').open('wb') as file:
                    np.savez_compressed(file,t=t,q=q,train_y=y[t],prior=prior,real=ratios[0],sham=ratios[1])
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[key],val)
                prior,real,sham=z['prior'],z['real'],z['sham']
            offset=logit(nat[loc].clip(.002,.998))
            direct={'masked':prior,'restored':prior+real,'shuffled':prior+sham}
            for name,score in direct.items():
                for weight in [1.,.25]:
                    pred=(1-weight)*nat[loc]+weight*expit(score)>=.5
                    flags.setdefault(name+'_'+str(weight),np.zeros(len(ids),bool))[loc]=pred!=(offset>=0)
            core=np.column_stack([np.ones(len(q)),offset,prior-offset])
            designs={'masked':core,'restored':np.column_stack([core,real]),'shuffled':np.column_stack([core,sham])}
            for mode in ['all','new']:
                if mode=='new' and tag=='historical':
                    continue
                sets=([(np.flatnonzero(~clean[loc]),np.flatnonzero(clean[loc]))] if mode=='new' else
                      GroupKFold(4,shuffle=True,random_state=9102).split(q,groups=g[q]))
                for tr,held in sets:
                    assert not np.isin(g[q[tr]],g[q[held]]).any()
                    for name,features in designs.items():
                        beta=fit(features[tr],offset[tr],y[q[tr]])
                        pred=offset[held]+features[held]@beta>=0
                        flags.setdefault(mode+'_'+name,np.zeros(len(ids),bool))[loc[held]]=pred!=(offset[held]>=0)
            print(tag,k,flush=True)
        anchor=view['anchor']
        wrong=anchor!=y[ids]
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        report={}
        for name,flag in flags.items():
            cohort={}
            for group,mask in [('all',view['known']),('new',view['known']&clean),('families',common),('new_families',clean&common)]:
                if name.startswith('new_') and not group.startswith('new'):
                    continue
                edit=flag&(anchor==(nat>=.5))&mask
                cohort[group]=dict(rows=int(mask.sum()),fixed=int((edit&wrong).sum()),harmed=int((edit&~wrong).sum()))
            report[name]=cohort
        result[tag]=report
        print(tag,[(s,r.get('all',r['new'])) for s,r in report.items()],flush=True)
    DEST.with_name('constrained_track_odds.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
