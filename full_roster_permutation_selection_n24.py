"""Nested permutation selection of engineered fields, with a matched random-field control."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import CATEGORICAL,compact_features
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS,ROOT,banks,data,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities

MODES=('all','selected','random','protected_selected','protected_random')
OUT=ROOT/'artifacts/nov24_permutation_selection'


def feature_bank(raw):
    frame=compact_features(raw)
    core=pd.get_dummies(frame,columns=CATEGORICAL,dtype=float)
    core_names=list(core)
    bill=frame[BILLS]
    total=bill.sum(axis=1,min_count=1)
    group=raw.PassengerId.str.split('_').str[0]
    size=group.map(group.value_counts())
    fields=dict(group_size=size,group_position=pd.to_numeric(raw.PassengerId.str.split('_').str[1]),
                cabin_occupancy=raw.Cabin.map(raw.Cabin.value_counts()),age_sq=frame.Age**2,
                age_cube=frame.Age**3,child=frame.Age.lt(13),total=total,services=bill.gt(0).sum(axis=1),
                missing_bills=bill.isna().sum(axis=1),max_charge=bill.max(axis=1),median_charge=bill.median(axis=1),
                per_year=total/(1+frame.Age),per_person=total/size)
    for col in [*BILLS,'Age','CabinNumber']:
        value=frame[col]
        peer=value.groupby(group)
        peer_n=peer.transform('count')-value.notna().astype(int)
        mean=(peer.transform('sum')-value.fillna(0))/peer_n.replace(0,np.nan)
        fields.update({col+'_missing':value.isna(),col+'_peer_mean':mean,
                       col+'_peer_contrast':value-mean,col+'_group_max':peer.transform('max')})
    for col in BILLS:
        value=bill[col]
        fields.update({col+'_log':np.log1p(value),col+'_positive':value>0,col+'_share':value/(1+total),
                       col+'_age_ratio':value/(1+frame.Age),col+'_size_ratio':value/size,
                       col+'_sleep':value*raw.CryoSleep.eq(True),col+'_vip':value*raw.VIP.eq(True)})
    for left,right in combinations(BILLS,2):
        a,b=bill[left],bill[right]
        k=left+'_'+right
        fields.update({k+'_sum':a+b,k+'_ratio':(a+1)/(b+1),k+'_contrast':np.log1p(a)-np.log1p(b),
                       k+'_both':(a>0)&(b>0),k+'_over_total':(a+b)/(1+total)})
    for name in ['Deck','Side','HomePlanet']:
        for val in sorted(frame[name].unique()):
            mask=frame[name].eq(val)
            fields[name+'_'+val+'_age']=mask*frame.Age
            fields[name+'_'+val+'_charge']=mask*np.log1p(total)
    wide=pd.concat([core,pd.DataFrame(fields,index=frame.index)],axis=1)
    assert wide.columns.is_unique and len(core_names)<52<wide.shape[1]
    return wide.to_numpy(dtype='float32'),list(wide),len(core_names)


def cat(seed):
    return CatBoostClassifier(iterations=350,depth=6,learning_rate=.055,random_seed=seed,
                              verbose=False,thread_count=3,allow_writing_files=False)


def select(x,y,t,g,core,seed):
    gains=np.zeros(x.shape[1])
    for h,(a,b) in enumerate(GroupKFold(2,shuffle=True,random_state=49).split(t,groups=g[t])):
        train,val=t[a],t[b]
        assert not np.isin(g[train],g[val]).any()
        model=cat(seed).fit(x[train],y[train])
        original=x[val]
        target=y[val]
        def loss(matrix):
            p=model.predict_proba(matrix,thread_count=3)[:,1].clip(1e-7,1-1e-7)
            return -np.log(np.where(target,p,1-p))
        base_loss=loss(original).sum()
        indices=np.random.default_rng(seed+h).permutation(len(val))
        # Batching makes repeated scoring faster without changing the permutations.
        for start in range(0,x.shape[1],8):
            cols=np.arange(start,min(start+8,x.shape[1]))
            block=np.tile(original,(len(cols),1))
            for j,col in enumerate(cols):
                block[j*len(val):(j+1)*len(val),col]=original[indices,col]
            p=model.predict_proba(block,thread_count=3)[:,1].reshape(len(cols),-1).clip(1e-7,1-1e-7)
            gains[cols]+=-np.log(np.where(target,p,1-p)).sum(1)-base_loss
    ranks=np.argsort(-gains,kind='stable')
    random=np.random.default_rng(seed+719).permutation(x.shape[1])
    def protect(order):
        return np.r_[np.arange(core),order[order>=core][:52-core]]
    return [np.arange(x.shape[1]),ranks[:52],random[:52],protect(ranks),protect(random)]


def fit_ensemble(x,y,seed):
    models=[cat(seed),
            LGBMClassifier(n_estimators=320,num_leaves=17,min_child_samples=30,learning_rate=.045,
                           verbosity=-1,n_jobs=3,random_state=seed),
            XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.055,min_child_weight=5,
                          n_jobs=3,tree_method='hist',random_state=seed)]
    return [model.fit(x,y) for model in models]


def ensemble_probability(models,x):
    probabilities=[]
    for model in models:
        if isinstance(model,LGBMClassifier):
            probabilities.append(model.booster_.predict(x,num_threads=3))
        else:
            probabilities.append(model.predict_proba(x)[:,1])
    return np.array([.4766,.2901,.2333])@np.stack(probabilities)


def ensemble(x,y,t,q,seed):
    return ensemble_probability(fit_ensemble(x[t],y[t],seed),x[q])


def protocol(raw,labels,groups,ix,splits,tag):
    x,names,core=feature_bank(raw)
    out=np.zeros((len(ix),len(MODES)))
    selections=[]
    OUT.mkdir(exist_ok=True)
    for f in np.unique(splits):
        q=ix[splits==f]
        t=ix[(splits!=f)&~np.isin(groups[ix],groups[q])]
        assert not np.isin(groups[t],groups[q]).any()
        path=OUT/f'{tag}_{f}.npz'
        if not path.exists():
            picks=select(x,labels,t,groups,core,497)
            prob=np.column_stack([ensemble(x[:,k],labels,t,q,497) for k in picks])
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,train=t,query=q,labels=labels[t],features=np.array(names),
                                    cols=np.stack(picks[1:]),prob=prob)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,val in [('train',t),('query',q),('labels',labels[t]),('features',names)]:
                np.testing.assert_array_equal(z[k],val)
            out[splits==f]=z['prob']
            selections.append(z['cols'][0].tolist())
        print(tag,int(f),len(t),len(q),flush=True)
    assert np.isfinite(out).all()
    return out,selections,names,core


def run():
    _,y,g,rows,oldfold,_,champ,_,_,_,_,old,_=aligned()
    raw,labels,groups,fold=data()
    names,half,full=banks(raw,labels,fold)
    ref=latest_reference(labels,groups,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,labels,groups,fold)
    a,os,fields,core=protocol(raw,labels,groups,rows,oldfold,'historical')
    b,ns,_,_=protocol(raw,labels,groups,np.arange(len(raw)),fold,'full')
    shared=((full>=.5)!=labels[:,None]).all(1)
    hard=shared[rows]&(champ!=y)
    unseen=~np.isin(groups,g)
    def stats(z,baseline,target,grp,split):
        d=(z==target).astype(int)-(baseline==target)
        return dict(accuracy=float((z==target).mean()),fixed=int((d==1).sum()),harmed=int((d==-1).sum()),
                    fold_net=[int(d[split==k].sum()) for k in np.unique(split)],ci_pp=interval(d,grp))
    report=dict(field_count=len(fields),core_count=core,field_names=fields,
                selected_columns_by_fold=dict(historical=os,full=ns),results={})
    for j,mode in enumerate(MODES):
        outcomes={}
        for mix,w in [('direct',1.),('quarter',.25)]:
            pa,pb=a[:,j]>=.5,b[:,j]>=.5
            if w<1:
                pa=np.where(champ==(old[:,0]>=.5),((1-w)*old[:,0]+w*a[:,j])>=.5,champ)
                pb=((1-w)*gate+w*b[:,j])>=.5
            outcomes[mix]=dict(historical=stats(pa,champ,y,g,oldfold),full=stats(pb,ref,labels,groups,fold),
                               unseen=stats(pb[unseen],ref[unseen],labels[unseen],groups[unseen],fold[unseen]),
                               common_fixed=int((shared&(pb==labels)).sum()),stable_fixed=int((hard&(pa==y)&(pb[rows]==y)).sum()))
        report['results'][mode]=outcomes
        print(mode,[(k,v['historical']['fixed']-v['historical']['harmed'],v['full']['fixed']-v['full']['harmed'],
                     v['unseen']['fixed']-v['unseen']['harmed']) for k,v in outcomes.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/permutation_selection.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(3):
        run()
