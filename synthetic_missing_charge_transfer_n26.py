"""Use masked complete passengers to test correction rules on other bookings with genuinely absent bills."""
import json

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from threadpoolctl import threadpool_limits

from classifying_reconstructed_bills_n26 import load_views
from disjoint_tree_fields_n25 import ART,DEST,stats
from exaone_validation_data import CATEGORICAL
from family_identity_transfer_n26 import fit,panels_and_reference
from full_roster_booking_validation_n23 import data
from latent_bill_rounding_n26 import BILLS,OUT
from outstanding_error_families_n26 import observed


def tree():
    return CatBoostClassifier(iterations=430,depth=6,learning_rate=.06,random_seed=9841,thread_count=2,
                              verbose=False,allow_writing_files=False)


def estimates(base,values,miss,y,t,q):
    model=tree().fit(base.iloc[t],y[t],cat_features=CATEGORICAL)
    matrix=np.full((len(q),len(BILLS),3),np.nan)
    for j,bill in enumerate(BILLS):
        valid=(miss[q].sum(1)-miss[q,j])==0
        query=q[valid]
        masked=base.iloc[query].copy()
        masked[bill]=np.nan
        standard=model.predict_proba(masked)[:,1]
        integrated=np.zeros(len(query))
        for k in range(values.shape[-1]):
            hypothetical=masked.copy()
            hypothetical[bill]=values[query,j,k]
            integrated+=model.predict_proba(hypothetical)[:,1]/values.shape[-1]
        cols=base.columns.drop(bill)
        specialist=tree().fit(base.iloc[t][cols],y[t],cat_features=CATEGORICAL)
        blinded=specialist.predict_proba(masked[cols])[:,1]
        matrix[valid,j,:]=np.column_stack([standard,integrated,blinded])
    return matrix


def rule_design(probs,services,values,idx):
    sign=np.where(probs[:,1]>=.5,1,-1)
    logits=sign[:,None]*logit(probs.clip(.002,.998))
    q=values[idx,services]
    return logits[:,1],np.column_stack([np.ones(len(idx)),sign,logits, np.eye(5)[services],
                            (q[:,2]==0),np.log1p(q[:,4])/6,np.log1p(q[:,2])/6])


def run():
    raw,y,g,f=data()
    base,_,quantiles,missing=load_views(raw)
    values=quantiles['spectrum']
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    result={}
    for tag,ids,fold,view in [('historical',rows,hf,hist),('full',np.arange(len(raw)),f,current)]:
        flags={}
        artificial=np.zeros((5,3),int)
        clean=~np.isin(g[ids],g[rows])
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k); q=ids[loc]
            rest=ids[fold!=k]; t=rest[~np.isin(g[rest],g[q])]
            assert not np.isin(g[t],g[q]).any()
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[rest])
            path=OUT/f'synthetic_{tag}_{k}.npz'
            if not path.exists():
                score=estimates(base,values,missing,y,t,q)
                with path.with_suffix('.tmp').open('wb') as out:
                    np.savez_compressed(out,t=t,q=q,train_y=y[t],score=score)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,val in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[name],val)
                score=z['score']
            real=np.flatnonzero(missing[q].sum(1)==1)
            complete=np.flatnonzero(missing[q].sum(1)==0)
            with np.load(OUT/f'charge_classifier_{tag}_{k}.npz') as z:
                s=missing[q[real]].argmax(1)
                np.testing.assert_allclose(score[real,s,0],z['raw'][real],atol=1e-10,rtol=0)
                np.testing.assert_allclose(score[real,s,1],z['spectrum_imputed'][real],atol=1e-10,rtol=0)
            ordinary=score[complete,:,0]>=.5
            imputed=score[complete,:,1]>=.5
            for j in range(5):
                change=ordinary[:,j]!=imputed[:,j]
                right=imputed[:,j]==y[q[complete]]
                artificial[j]+=[len(complete),(change&right).sum(),(change&~right).sum()]
            # Calibrate only on synthetic masks from unrelated held-out bookings.
            pool=complete[~np.isin(g[q[complete]],g[q[real]])]
            assert not np.isin(g[q[pool]],g[q[real]]).any()
            x,sj=np.where((score[pool,:,0]>=.5)!=(score[pool,:,1]>=.5))
            tr=pool[x]
            off,features=rule_design(score[tr,sj],sj,values,q[tr])
            labels=(score[tr,sj,1]>=.5)==y[q[tr]]
            beta=fit(features,off,labels)
            before,after,expert=(score[real,s].T>=.5)
            new_off,new_x=rule_design(score[real,s],s,values,q[real])
            anchor=view['anchor'][loc[real]]
            adopt=(after!=before)&(after!=anchor)
            acts={'integrated':adopt, 'specialist':(expert!=before)&(expert!=anchor),
                  'confirmed':adopt&(expert==after), 'learned':adopt&(new_off+new_x@beta>0)}
            for name,dec in acts.items():
                flags.setdefault(name,np.zeros(len(ids),bool))[loc[real]]=dec
            print(tag,k,len(tr),flush=True)
        wrong=view['anchor']!=y[ids]
        fam=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        res={}
        for name,flag in flags.items():
            delta=flag*np.where(wrong,1,-1)
            res[name]={scope:stats(delta,mask,fold,g[ids]) for scope,mask in [('all',view['known']),('new',view['known']&clean),
                        ('true_missing_one',view['known']&(missing[ids].sum(1)==1)),('families',fam)]}
        result[tag]=dict(real_missing=res,synthetic={bill:dict(zip(('cases','fixed','harmed'),map(int,r))) for bill,r in zip(BILLS,artificial)})
        print(tag,[(n,s['all']['fixed'],s['all']['harmed']) for n,s in res.items()],artificial.sum(0),flush=True)
    DEST.with_name('synthetic_missing_charge_transfer.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
