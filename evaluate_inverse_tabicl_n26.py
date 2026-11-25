"""Compare inverse attribute evidence with masked priors, label shuffles and ordinary forward TabICL."""
import argparse
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import panels_and_reference,fit
from full_roster_booking_validation_n23 import data
from grouped_peer_release_replay_n25 import edit_effect_bounds

OUT=ART/'nov26_inverse_tabicl_manifest'
VIEWS=('dominant_service','cabin_side','home_planet')


def odds(p):
    return logit(np.clip(p,.001,.999))


def ratio(values):
    return np.log(values[:,1].clip(.001,1)/values[:,0].clip(.001,1))


def run(folds,paid):
    raw,y,g,f=data()
    _,_,_,rows,_,archive,hist,native,_,ref=panels_and_reference()
    votes,old=[(v>=.5).sum(1) for v in (archive,hist)]
    common=(votes==0)|(votes==68)
    hard=(votes==0)&raw.CryoSleep.eq(False).to_numpy()&~raw.PassengerId.str[:4].duplicated(keep=False).to_numpy()
    assert (hard&y).sum()==143
    expenses=raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].to_numpy(float)
    ranked=np.sort(np.nan_to_num(expenses,nan=-1),axis=1)
    unique_basket=(ranked[:,-1]>0)&(ranked[:,-1]>ranked[:,-2])
    repeat=np.zeros(len(y),bool)
    repeat[rows]=common[rows]&((old==0)|(old==68))&((old>34)==(votes[rows]>34))
    fresh=~np.isin(g,g[rows])
    records={}
    calibration={}
    opportunity={}
    for k in folds:
        q=np.flatnonzero(f==k)
        with np.load(ART/f'nov25_passenger_state_contexts/{k}_global.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            forward=z['p']
        native_sign=native[q]>=.5
        routes={'forward_direct':forward>=.5,'forward_quarter':.75*native[q]+.25*forward>=.5}
        priors,real,sham=[],[],[]
        for name in VIEWS:
            with np.load(OUT/f'{k}_{name}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
                p=z['prior']
                r,s=ratio(z['observed']),ratio(z['shuffled'])
                np.testing.assert_allclose(expit(odds(p)+r),z['restored'],atol=1e-6)
            if name=='dominant_service':
                # When there is no unique positive maximum the identity code is
                # determined by the sorted basket, for either hypothesized outcome.
                r=np.where(unique_basket[q],r,0.)
                s=np.where(unique_basket[q],s,0.)
                if paid:
                    with np.load(OUT/f'{k}_paid_identity.npz') as z:
                        ids=q[unique_basket[q]]
                        np.testing.assert_array_equal(z['q'],ids)
                        for key,score in [('observed',r),('shuffled',s)]:
                            test=native[q].copy()
                            test[unique_basket[q]]=expit(odds(z['prior'])+ratio(z[key]))
                            routes['paid_only_'+key]=test>=.5
                            routes['paid_only_quarter_'+key]=.75*native[q]+.25*test>=.5
            masked=odds(p)
            priors.append(masked)
            real.append(r)
            sham.append(s)
            for key,estimate in [('masked',p),('restored',expit(masked+r)),('shuffled',expit(masked+s))]:
                routes[name+'_'+key]=estimate>=.5
                routes[name+'_quarter_'+key]=.75*native[q]+.25*estimate>=.5
        priors,real,sham=(np.column_stack(v) for v in (priors,real,sham))
        routes['majority_restored']=((priors+real>=0).sum(1)>=2)
        routes['majority_shuffled']=((priors+sham>=0).sum(1)>=2)
        far=common[q]
        against_68=(priors+real>=0)!=(votes[q,None]>34)
        shuffled_opposes=(priors+sham>=0)!=(votes[q,None]>34)
        for key,flag in [('any_original',(priors>=0)!=(votes[q,None]>34)),('any_restored',against_68),('any_shuffled',shuffled_opposes)]:
            flips=flag.any(1)&far
            wrong=(votes[q]>34)!=y[q]
            opportunity.setdefault(key,[]).append((int((flips&wrong).sum()),int((flips&~wrong).sum())))
        # Every calibration observation comes from the same held fold as the query.
        # The base and all hypothesis models therefore excluded every booking used here.
        offset=odds(native[q])
        main=np.column_stack([np.ones(len(q)),offset,odds(forward)-offset])
        masked=priors-offset[:,None]
        designs={'forward':main,'masked':np.column_stack([main,masked]),
                 'real':np.column_stack([main,masked,real]),'shuffled':np.column_stack([main,masked,sham])}
        schemes={'all':list(GroupKFold(4,shuffle=True,random_state=2649).split(q,groups=g[q])),
                 'new':[(np.flatnonzero(~fresh[q]),np.flatnonzero(fresh[q]))]}
        for scheme,parts in schemes.items():
            for train,test in parts:
                assert not np.isin(g[q[train]],g[q[test]]).any()
                for name,x in designs.items():
                    beta=fit(x[train],offset[train],y[q[train]])
                    decision=offset[test]+x[test]@beta>=0
                    a,b=edit_effect_bounds(decision,native_sign[test],*ref[:,q[test]],y[q[test]])
                    calibration.setdefault(scheme+'/'+name,[]).append((q[test],a,b))
        for name,decision in routes.items():
            a,b=edit_effect_bounds(decision,native_sign,*ref[:,q],y[q])
            records.setdefault(name,[]).append((q,a,b))
    def report(items):
        output={}
        for name,parts in items.items():
            q,a,b=(np.concatenate([block[j] for block in parts]) for j in range(3))
            assert len(np.unique(q))==len(q)
            known=ref[0,q]>=0
            np.testing.assert_array_equal(a[known],b[known])
            r=dict(bounds=[int(a.sum()),int(b.sum())])
            for title,mask in [('known',known),('new_orders',known&fresh[q]),('common',known&common[q]),
                ('repeat',known&repeat[q]),('awake_solo_negative',known&hard[q])]:
                if mask.any():
                    r[title]=dict(rows=int(mask.sum()),fixed=int((a[mask]>0).sum()),harmed=int((a[mask]<0).sum()),
                        by_fold=[int(a[mask&(f[q]==k)].sum()) for k in folds])
            output[name]=r
        return output
    final=dict(folds=folds,unique_paid_model=paid,direct=report(records),calibrated=report(calibration),
               common_opportunity={name:np.sum(values,axis=0).tolist() for name,values in opportunity.items()})
    for table,names in [('direct',['forward_direct','dominant_service_restored','cabin_side_restored','home_planet_restored','majority_restored']),
                        ('calibrated',list(final['calibrated']))]:
        for name in names:
            r=final[table][name]
            print(table,name,[(t,r[t]['fixed'],r[t]['harmed']) for t in ('known','new_orders','common') if t in r])
    print('union',final['common_opportunity'])
    if folds==list(range(8)) and paid:
        DEST.with_name('inverse_tabicl_manifest.json').write_text(json.dumps(final,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folds',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--without-paid',action='store_true')
    args=p.parse_args()
    with threadpool_limits(2):
        run(args.folds,not args.without_paid)
