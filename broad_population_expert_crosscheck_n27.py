"""Require concordant probability changes from additive and tree experts trained on the same passenger population."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from matched_state_context_n26 import inputs, ROOT, FEES
from exaone_validation_data import compact_features, CATEGORICAL
from repeated_archive_misses_n26 import ART, cohort, count

OUT=ART/'nov27_broad_population_expert_crosscheck'
REPORT=ROOT/'reports/nov23_matched_learning_curve/broad_population_expert_crosscheck.json'
CHAMP='proposal_n27_directional_foundation'


def design(frame,g):
    c=compact_features(frame)
    size=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    income=frame[FEES]
    amount=income.fillna(0).to_numpy().sum(1)
    paid=amount>0
    age=frame.Age.to_numpy()
    design=pd.get_dummies(c,columns=CATEGORICAL,dtype=float).to_numpy()
    tree=np.column_stack([design,size,np.log1p(amount)])
    cat=pd.get_dummies(c[CATEGORICAL],dtype=float)
    num=[]
    for feat,scale,cuts in [(age,60,[6,12,18,30,50]),(c.CabinNumber.to_numpy(),1600,[200,600,1000,1400]),
                            *[(np.log1p(income[col].to_numpy()),8,[2,4,6.5,7.5]) for col in FEES]]:
        value=np.nan_to_num(feat)
        num.extend([np.isnan(feat).astype(float),value/scale,(value>0).astype(float)])
        num.extend(np.maximum(value-t,0)/scale for t in cuts)
    numeric=np.column_stack(num)
    sleep=frame.CryoSleep.eq(True).to_numpy()
    # Additive spline model uses smooth service and cabin terms and origin/sleep interactions, no tree partitions.
    linear=np.column_stack([cat.to_numpy(),numeric,np.log1p(size),np.log1p(amount)/8,
                            numeric*sleep[:,None],numeric*frame.HomePlanet.eq('Earth').to_numpy()[:,None],
                            numeric*frame.HomePlanet.eq('Europa').to_numpy()[:,None]])
    return tree,linear,dict(paid=paid.astype(int),party=(size>1).astype(int),
                           age=np.where(~np.isfinite(age),-1,(age<18).astype(int)))


def train(tree,linear,y,t,q):
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=22,l2_regularization=5,learning_rate=.075,
                                        early_stopping=False,random_state=743)
    m=LogisticRegression(C=1.4,max_iter=450,solver='lbfgs',tol=2e-4)
    return np.column_stack([model.fit(tree[t],y[t]).predict_proba(tree[q])[:,1],
                            m.fit(linear[t],y[t]).predict_proba(linear[q])[:,1]])


def run():
    frame,y,g,f,states=inputs();ids=frame.PassengerId.to_numpy(str);n=len(y)
    tree,linear,axes=design(frame,g)
    _,blind,known,fresh=cohort(frame.iloc[:n],y,f)
    OUT.mkdir(exist_ok=True)
    ordered=np.zeros(n,int)
    for k,orders in enumerate(np.array_split(np.unique(g[:n]),8)):
        ordered[np.isin(g[:n],orders)]=k
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        anchor=z['parent']
        outsiders=[np.median(z[k],axis=1) if z[k].ndim==2 else z[k] for k in ('dpt','tabnet')]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        panel=z['scores']
        native=np.median(panel[:,-3:],axis=1)
        votes=np.column_stack([np.median(panel[:,:18],axis=1),panel[:,-3:],native,*outsiders])
    seven=((votes[:n]>=.5)==anchor[:n,None]).all(1)&known
    with np.load(ART/'nov26_full_state_context/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        tab=z['p'] if 'p' in z else z['full']
    sign=np.where(anchor,-1.,1.); odds=lambda x:logit(np.clip(x,.004,.996))
    tg=(odds(tab)-odds(panel[:,-2]))*sign
    ta=np.isfinite(tab)&((tab>=.5)!=anchor)&(tg>0)
    full={};summary={};cache={}
    for mode,fold in [('random',f),('contiguous',ordered)]:
        bank={key:np.empty((len(ids),2)) for key in ['global',*axes]}
        for k in range(9):
            t=np.flatnonzero(fold!=k)
            q=np.flatnonzero(fold==k) if k<8 else np.arange(n,len(ids))
            assert not np.isin(g[t],g[q]).any()
            path=OUT/(f'{mode}_{k}.npz' if k<8 else 'test.npz')
            if not path.exists():
                per={'global':train(tree,linear,y,t,q)}
                for name,a in axes.items():
                    p=per['global'].copy()
                    for val in [0,1]:
                        tr=t[a[t]==val];qr=q[a[q]==val]
                        if len(qr):p[a[q]==val]=train(tree,linear,y,tr,qr)
                    per[name]=p
                np.savez_compressed(path,t=t,q=q,labels=y[t],ids=ids[q],**per)
            with np.load(path) as z:
                for field,exp in [('t',t),('q',q),('labels',y[t]),('ids',ids[q])]:
                    np.testing.assert_array_equal(z[field],exp)
                for name in bank:bank[name][q]=z[name]
            print(mode,k,flush=True)
        rules={}; activity={}; signed={}
        for name,axis in axes.items():
            spec=bank[name];base=bank['global']
            gain=(odds(spec)-odds(base))*sign[:,None]
            opposed=(spec>=.5)!=anchor[:,None]
            both=(gain>0).all(1)&opposed.all(1)&(axis>=0)
            crosses=(gain>abs(odds(native))[:,None]).all(1)&opposed.all(1)&(axis>=0)
            new=both&((base>=.5)==anchor[:,None]).any(1)
            activity[name]=both; signed[name]=gain
            rules[name+'_both']=both; rules[name+'_new']=new; rules[name+'_increment']=crosses
            labels={'paid':('unpaid','paying'),'party':('solo','accompanied'),'age':('adult','child')}[name]
            for value,label in enumerate(labels):
                rules[label+'_new_votes']=new&(axis==value)
                rules[label+'_increment_votes']=crosses&(axis==value)

            if name=='paid':
                # The TabICL expert was fitted on the random eight folds only; its label is auxiliary in the second protocol.
                rules[name+'_tab_confirmed']=both&ta
                rules[name+'_tab_new']=new&ta
        votes_axis=np.column_stack(list(activity.values())).sum(1)
        rules['two_populations']=votes_axis>=2
        rules['three_populations']=votes_axis==3
        unique=np.stack([((odds(bank[name])-odds(bank['global']))*sign[:,None]) for name in axes],axis=1)
        median=(np.median(unique,axis=1)>abs(odds(native))[:,None]).all(1)&(votes_axis>=2)
        rules['two_populations_increment']=median
        # Learn which broad expert populations earn support, within each outer fold on separate bookings.
        import hashlib
        part=np.array([int.from_bytes(hashlib.sha256(('expert-booking-'+v).encode()).digest()[:4],'big')%3 for v in g[:n]])
        for policy in ('both','new'):
            calls={name:rules[name+'_'+policy] for name in axes}
            votes=np.zeros((len(ids),3),int)
            def select(t,q):
                prediction=np.zeros((len(q),3),int)
                for j,(name,axis) in enumerate(axes.items()):
                    for direction in (False,True):
                        for field in (0,1):
                            eligible=(axis==field)&(anchor==direction)&calls[name]
                            examples=t[eligible[t]]
                            successes=np.sum(anchor[examples]!=y[examples])
                            gain=2*successes-len(examples)
                            if gain>=max(2,np.sqrt(len(examples))):
                                prediction[:,j]|=eligible[q]
                return prediction
            for k in range(8):
                rows=np.flatnonzero(fold==k)
                for j in range(3):
                    t=rows[(part[rows]!=j)&known[rows]]
                    q=rows[part[rows]==j]
                    assert not np.isin(g[t],g[q]).any()
                    votes[q]=select(t,q)
            votes[n:]=select(np.flatnonzero(known),np.arange(n,len(ids)))
            rules['trained_'+policy]=(votes>0).any(1)
            rules['trained_'+policy+'_quorum']=(votes>0).sum(1)>=2
        masknames={'all':known,'fresh':known&fresh,'previously_screened':known&~fresh,'common':seven,'old_common':blind}
        wrong=anchor[:n]!=y
        summary[mode]={}
        for name,act in rules.items():
            d=act[:n]*np.where(wrong,1,-1)
            summary[mode][name]=dict(scopes={label:count(act[:n],mask,wrong) for label,mask in masknames.items()},
                fold_net=[int(d[(fold==k)&known].sum()) for k in range(8)],test=int(act[n:].sum()),orders=int(len(np.unique(g[n:][act[n:]]))))
        full[mode]=rules
        for key,val in bank.items():cache[mode+'_'+key]=val
    for key in [k for k in full['random'] if not k.startswith('trained_')]:
        np.testing.assert_array_equal(full['random'][key][n:],full['contiguous'][key][n:])
    for key in [k for k in full['random'] if k.startswith('trained_')]:
        common_test=full['random'][key][n:]&full['contiguous'][key][n:]
        for mode in full:summary[mode][key]['confirmed_test']=int(common_test.sum())
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,fold=f,y=y,parent=anchor,ordered=ordered,**cache)
    test=pd.DataFrame({'PassengerId':ids[n:]})
    for key in ['global',*axes]:
        for index,arch in enumerate(['tree','spline']):
            test[arch+'_'+key]=cache['random_'+key][n:,index]
    test['tabicl_state']=tab[n:]
    test.to_csv(OUT/'test_probabilities.csv',index=False)
    REPORT.write_text(json.dumps(dict(known=int(known.sum()),fresh=int((known&fresh).sum()),common=int(seven.sum()),
        current=CHAMP,external_specialist='TabICL full passenger state against global fare foundation; random outer folds only',
        primary='two separately refit architectures and the same group definition; no target-based subgroup choice or alpha',
        results=summary),indent=2)+'\n')
    for key in full['random']:
        print(key,[(mode,summary[mode][key]['scopes']) for mode in ['random','contiguous']],summary['random'][key]['test'])


if __name__=='__main__':
    with threadpool_limits(1):run()
