"""Estimate all four outcomes for pairs from separate bookings with similar registration fields."""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from exaone_validation_data import CATEGORICAL, compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

BILLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def vectors(raw):
    x = compact_features(raw)
    category = pd.get_dummies(x[CATEGORICAL], dtype=float).to_numpy()
    reg = np.column_stack([category,x.Age.fillna(30)/30,x.Age.isna(),
                           x.CabinNumber.fillna(800)/800,x.CabinNumber.isna()])
    bills = x[BILLS].to_numpy(float)
    basket = np.column_stack([np.nan_to_num(np.log1p(bills),nan=-1),np.isnan(bills)])
    return reg,basket


def neighbors(reg,g,bank,queries,width):
    tree=cKDTree(reg[bank])
    dist,positions=tree.query(reg[queries],k=min(len(bank),width+12))
    result=np.empty((len(queries),width),int)
    distances=np.empty((len(queries),width))
    for i,(row,positions_i) in enumerate(zip(queries,positions)):
        ok=g[bank[positions_i]]!=g[row]
        result[i]=bank[positions_i[ok][:width]]
        distances[i]=dist[i,ok][:width]
    return result,distances


def independent_pairs(reg,g,q):
    near,dist=neighbors(reg,g,q,q,12)
    owners=np.broadcast_to(q[:,None],near.shape).ravel()
    donors=near.ravel()
    used=set()
    left,right=[],[]
    for pos in np.argsort(dist.ravel(),kind='stable'):
        a,b=owners[pos],donors[pos]
        if g[a] not in used and g[b] not in used:
            left.append(a)
            right.append(b)
            used.update([g[a],g[b]])
    return np.asarray(left),np.asarray(right)


def features(reg,bill,left,right,mode):
    a,b=reg[left],reg[right]
    parts=[a,b,a-b]
    if mode!='masked':
        c,d=bill[left],bill[right]
        if mode=='swapped':
            c,d=d,c
        parts.extend([c,d,c-d,c*d])
    return np.column_stack(parts)


def symmetric(model,reg,bill,i,j,mode):
    p=model.predict_proba(features(reg,bill,i,j,mode))
    r=model.predict_proba(features(reg,bill,j,i,mode))
    return (p+r[:,[0,2,1,3]])/2


def evaluate(raw,y,g,ids,fold,anchor,known,families,fresh):
    reg,bill=vectors(raw)
    rowscores={m:np.zeros(len(ids)) for m in ('masked','observed','swapped')}
    accuracy={m:[] for m in rowscores}
    pair_actions={}
    pair_diagnostics={}
    for k in np.unique(fold):
        loc=np.flatnonzero(fold==k)
        q=ids[loc]
        t=ids[(fold!=k)&~np.isin(g[ids],g[q])]
        assert not np.isin(g[q],g[t]).any()
        candidates,_=neighbors(reg,g,t,t,16)
        gen=np.random.default_rng(319+k)
        # Independently choose four of each passenger's registration neighbours.
        choice=np.argsort(gen.uniform(size=candidates.shape),axis=1)[:,:4]
        a=np.repeat(t,4)
        b=np.take_along_axis(candidates,choice,axis=1).ravel()
        i,j=np.r_[a,b],np.r_[b,a]
        label=2*y[i].astype(int)+y[j].astype(int)
        left,right=independent_pairs(reg,g,q)
        bank,_=neighbors(reg,g,t,q,8)
        query,donor=np.repeat(q,8),bank.ravel()
        for train_mode in ('masked','observed'):
            model=HistGradientBoostingClassifier(max_iter=85,learning_rate=.09,max_leaf_nodes=15,
                min_samples_leaf=32,l2_regularization=8,early_stopping=False,random_state=991)
            model.fit(features(reg,bill,i,j,train_mode),label)
            np.testing.assert_array_equal(model.classes_,np.arange(4))
            for mode in ([train_mode] if train_mode=='masked' else ['observed','swapped']):
                joint=symmetric(model,reg,bill,left,right,mode)
                # Historical rows are not necessarily sorted.
                lookup={int(v):j for j,v in enumerate(ids)}
                li=np.array([lookup[v] for v in left])
                ri=np.array([lookup[v] for v in right])
                valid=known[li]&known[ri]
                give_left=joint[:,2]>joint[:,1]
                mixed=joint[:,1]+joint[:,2]>.5
                clear=abs(joint[:,2]-joint[:,1])>1e-12
                reverse=(anchor[li]!=anchor[ri])&(anchor[li]!=give_left)&clear
                for name,condition in [('reverse',reverse),('mixed_reverse',mixed&reverse),
                                       ('mixed_orient',mixed&clear)]:
                    active=valid&condition
                    edits=pair_actions.setdefault(mode+'_'+name,np.zeros(len(ids),bool))
                    edits[li[active]]=anchor[li[active]]!=give_left[active]
                    edits[ri[active]]=anchor[ri[active]]==give_left[active]
                unequal=y[left]!=y[right]
                signed=(joint[:,2]-joint[:,1])*np.where(y[left],1,-1)
                for cohort,selected in [('same_anchor',valid&(anchor[li]==anchor[ri])),
                                       ('both_families',families[li]&families[ri])]:
                    actual=selected&unequal
                    entry=pair_diagnostics.setdefault(mode+'_'+cohort,np.zeros(3,int))
                    entry+=[actual.sum(),(actual&(signed>1e-12)).sum(),(actual&(abs(signed)<=1e-12)).sum()]
                accuracy[mode].append([int(unequal.sum()),int((unequal&(signed>1e-12)).sum()),
                                       int((unequal&(abs(signed)<=1e-12)).sum()),len(left)])
                # In deployment no knowledge of whether labels differ is available.
                # Integrate the query probability conditional on each known training donor label.
                prob=symmetric(model,reg,bill,query,donor,mode)
                d=y[donor].astype(int)
                m=np.arange(len(d))
                pos=prob[m,2+d]
                rowscores[mode][loc]=(pos/(pos+prob[m,d])).reshape(-1,8).mean(1)
        print('fold',k,flush=True)
    result={}
    for mode,score in rowscores.items():
        n,w,tie,total=np.sum(accuracy[mode],axis=0).tolist()
        side=score>=.5
        result[mode]=dict(all_pairs=total,discordant=n,correct=w,ties=tie,
                          pair_accuracy=(w+.5*tie)/n,deployment={})
        for name,mask in [('all',known),('new_orders',known&fresh),('families',families)]:
            wrong=anchor!=y[ids]
            edits=side!=anchor
            def count(change):
                return dict(fixed=int((mask&wrong&change).sum()),harmed=int((mask&~wrong&change).sum()))
            result[mode]['deployment'][name]=dict(rows=int(mask.sum()),direct=count(edits),
                induced=count(edits&(side!=(rowscores['masked']>=.5))))
    wrong=anchor!=y[ids]
    result['conditional_pair_diagnostics']={k:dict(discordant=int(n),correct=int(w),ties=int(t))
                  for k,(n,w,t) in pair_diagnostics.items()}
    result['paired_deployment']={name:{scope:dict(fixed=int((edits&wrong&msk).sum()),
                harmed=int((edits&~wrong&msk).sum())) for scope,msk in
                [('all',known),('new',known&fresh),('families',families)]} for name,edits in pair_actions.items()}
    return result


def reference():
    """Score exactly the same held pairs with existing per-passenger Mitra probabilities."""
    raw,y,g,f=data()
    _,_,_,rows,_,_,_,native,old_native,_=panels_and_reference()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf=z['fold']
    reg,_=vectors(raw)
    result={}
    for tag,ids,fold,scores in [('full',np.arange(len(y)),f,native),('historical',rows,hf,old_native)]:
        score=np.full(len(y),np.nan)
        score[ids]=scores
        counts=np.zeros(3,int)
        for k in np.unique(fold):
            i,j=independent_pairs(reg,g,ids[fold==k])
            mask=y[i]!=y[j]
            margin=(score[i]-score[j])*np.where(y[i],1,-1)
            counts+=[mask.sum(),(mask&(margin>1e-12)).sum(),(mask&(abs(margin)<=1e-12)).sum()]
        n,w,tied=map(int,counts)
        result[tag]=dict(discordant=n,correct=w,ties=tied,pair_accuracy=(w+.5*tied)/n)
    return result


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,on,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,nine,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    history=observed(rows,hf,h,names,on,par,nine,nine,awake,'historical')
    full=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    output={}
    for tag,ids,fold,view in [('historical',rows,hf,history),('full',np.arange(len(y)),f,full)]:
        fresh=~np.isin(g[ids],g[rows])
        if tag=='historical':
            for k in np.unique(fold):
                fresh[fold==k]=~np.isin(g[ids[fold==k]],g[ids[fold!=k]])
        common=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        output[tag]=evaluate(raw,y,g,ids,fold,view['anchor'],view['known'],common,fresh)
        print(tag,output[tag],flush=True)
    output['native_reference']=reference()
    DEST.with_name('matched_bill_pair_states.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
