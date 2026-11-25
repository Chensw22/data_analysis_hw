"""Predict transport using estimated charges where absent and observable receipt surprise elsewhere."""
import json

import numpy as np
from catboost import CatBoostClassifier

from disjoint_tree_fields_n25 import ART,DEST
from exaone_validation_data import CATEGORICAL,compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from latent_bill_rounding_n26 import BILLS,OUT
from outstanding_error_families_n26 import observed

MODES=('donor','spectrum','shifted')


def load_views(raw):
    base=compact_features(raw)
    views={}
    quantiles={}
    with np.load(OUT/'missing_charge_estimates.npz') as file:
        np.testing.assert_array_equal(file['passenger'][:len(raw)],raw.PassengerId)
        missing=file['missing'][:len(raw)]
        np.testing.assert_array_equal(missing,base[BILLS].isna().to_numpy())
        for mode in MODES:
            x=base.copy()
            probs=file[mode+'_zero'][:len(raw)]
            log_mass=file[mode+'_logprob'][:len(raw)]
            quantiles[mode]=file[mode][:len(raw)]
            for j,name in enumerate(BILLS):
                can_estimate=(missing.sum(1)-missing[:,j])==0
                observed=can_estimate&~missing[:,j]
                x[name+'_expected_zero']=np.where(can_estimate,probs[:,j],np.nan)
                x[name+'_log_probability']=np.where(observed,log_mass[:,j],np.nan)
                x[name+'_expected_median']=np.where(can_estimate,quantiles[mode][:,j,2],np.nan)
            views[mode]=x
    return base,views,quantiles,missing


def marginal(model,base,q,values,missing,standard):
    loc=np.flatnonzero(missing[q].sum(1)==1)
    ids=q[loc]
    expense=np.argmax(missing[ids],axis=1)
    pred=np.zeros(len(ids))
    # Each row has exactly one unobserved service; the five quantiles integrate it out.
    for node in range(values.shape[2]):
        x=base.iloc[ids].copy()
        for j,bill in enumerate(BILLS):
            use=expense==j
            x.loc[x.index[use],bill]=values[ids[use],j,node]
        pred+=model.predict_proba(x)[:,1]/values.shape[2]
    full=standard.copy()
    full[loc]=pred
    return full


def run():
    raw,y,g,f=data()
    base,views,quantiles,missing=load_views(raw)
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    result={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,current)]:
        clean=~np.isin(g[ids],g[rows])
        all_probs={}
        for k in np.unique(fold):
            loc=fold==k
            q=ids[loc]; rest=ids[~loc]; t=rest[~np.isin(g[rest],g[q])]
            assert not np.isin(g[t],g[q]).any()
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[rest])
            path=OUT/f'charge_classifier_{tag}_{k}.npz'
            if not path.exists():
                scores={}
                for name,x in [('raw',base),*views.items()]:
                    model=CatBoostClassifier(iterations=430,depth=6,learning_rate=.06,random_seed=9841,thread_count=2,
                                             verbose=False,allow_writing_files=False)
                    model.fit(x.iloc[t],y[t],cat_features=CATEGORICAL)
                    scores[name]=model.predict_proba(x.iloc[q])[:,1]
                    if name=='raw':
                        for mode in MODES:
                            scores[mode+'_imputed']=marginal(model,base,q,quantiles[mode],missing,scores['raw'])
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle,t=t,q=q,train_y=y[t],**scores)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,value in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[name],value)
                for name in ['raw',*MODES,*[m+'_imputed' for m in MODES]]:
                    all_probs.setdefault(name,np.empty(len(ids)))[loc]=z[name]
            print(tag,k,flush=True)
        anchor=view['anchor']
        wrong=anchor!=y[ids]
        family=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        scoped=dict(all=view['known'],new=view['known']&clean,families=family,new_families=family&clean,
                    missing_one=view['known']&(missing[ids].sum(1)==1))
        table={}
        for name,probs in all_probs.items():
            before=all_probs['raw']>=.5
            after=probs>=.5
            rules={'direct':after!=anchor,'induced':(before!=after)&(after!=anchor),
                   'quarter':(anchor==(nat>=.5))&((.75*nat+.25*probs>=.5)!=anchor)}
            if name.startswith('spectrum'):
                peer=all_probs[name.replace('spectrum','donor')]>=.5
                rules['incremental']=(peer!=after)&(after!=anchor)
            for rule,flag in rules.items():
                counts={}
                for cohort,mask in scoped.items():
                    edit=flag&mask
                    counts[cohort]=dict(rows=int(mask.sum()),fixed=int((edit&wrong).sum()),harmed=int((edit&~wrong).sum()),
                        by_fold=[int((edit&wrong&(fold==k)).sum())-int((edit&~wrong&(fold==k)).sum()) for k in np.unique(fold)])
                table[name+'_'+rule]=counts
        result[tag]=table
        print(tag,[(k,v['all']['fixed'],v['all']['harmed']) for k,v in table.items() if k.endswith('induced')],flush=True)
    DEST.with_name('classifying_reconstructed_bills.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    run()
