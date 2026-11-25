"""Estimate and marginalize joint error patterns of the frozen ensemble across unseen bookings."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import fit,panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed


def odds(p):
    return logit(np.clip(p,.005,.995))


def group_parts(raw,ids,fold,prob,native,view,y,g):
    known=view['known']
    groups=[]
    blocks=[]
    prior=np.where(view['anchor'],-1.,1.)*odds(native)
    signed=np.where(view['anchor'],-1.,1.)[:,None]*odds(prob)
    cov=np.column_stack([np.ones(len(ids)),view['anchor'],prior,signed.mean(1)-prior,
                         signed.std(1),(signed>0).mean(1),abs(prior)])
    extended=np.zeros((len(ids),5))
    err=view['anchor']!=y[ids]
    cabin=raw.Cabin.fillna('?').to_numpy()[ids]
    for k in np.unique(fold):
        eligible=np.flatnonzero(known&(fold==k))
        for sub in pd.Series(g[ids[eligible]]).groupby(g[ids[eligible]],sort=False).indices.values():
            q=eligible[sub]
            n=len(q)
            index=np.arange(1<<n)
            states=((index[:,None]>>np.arange(n))&1).astype(float)
            if n>1:
                risk=expit(prior[q])
                others=(risk.sum()-risk)/(n-1)
                extended[q]=np.column_stack([np.full(n,np.log(n)),others,others*risk,np.full(n,np.std(risk)),
                                     np.full(n,np.std(view['anchor'][q]))])
            pair=np.zeros((len(index),5))
            i,j=np.triu_indices(n,k=1)
            if len(i):
                # Two errors can point toward the same or opposite transport label.
                same=view['anchor'][q[i]]==view['anchor'][q[j]]
                shared=(cabin[q[i]]!='?')&(cabin[q[i]]==cabin[q[j]])
                residual=states-expit(prior[q])
                corr=residual[:,i]*residual[:,j]
                norm=np.sqrt(len(i))
                pair=np.column_stack([corr.sum(1),corr@same,corr@(~same),corr@shared,
                                     (states.sum(1)>0).astype(float)])/norm
            truth=int(err[q]@(1<<np.arange(n)))
            groups.append(q)
            blocks.append((states,pair,truth))
    return dict(cov=cov,extended=np.column_stack([cov,extended]),prior=prior,err=err,
                groups=groups,blocks=blocks,fold=fold,anchor=view['anchor'],ids=ids,raw=raw)


def permute_errors(bundle,generator,selected=None):
    parts=bundle['groups'] if selected is None else [bundle['groups'][k] for k in selected]
    rows=np.concatenate(parts)
    original=bundle['err']
    strata=np.column_stack([bundle['fold'][rows],bundle['anchor'][rows],
            np.digitize(bundle['prior'][rows],[-3.,-1.6,-.8,0.]),
            np.digitize(bundle['cov'][rows,5],[.001,.1,.3]),bundle['extended'][rows,-5]>0])
    df=pd.DataFrame(strata)
    sets=[rows[idx] for idx in df.groupby(list(df),sort=False).indices.values()]
    result=original.copy()
    for ix in sets:
        result[ix]=generator.permutation(original[ix])
    return result


def pair_audit(bundle):
    groups=bundle['groups']
    left=[];right=[]
    for grp in groups:
        i,j=np.triu_indices(len(grp),1)
        left.extend(grp[i]);right.extend(grp[j])
    left,right=np.array(left),np.array(right)
    same=bundle['anchor'][left]==bundle['anchor'][right]
    cabins=bundle['raw'].Cabin.fillna('?').to_numpy()[bundle['ids']]
    sleep=bundle['raw'].CryoSleep.astype('string').fillna('?').to_numpy()[bundle['ids']]
    cabin_known=(cabins[left]!='?')&(cabins[right]!='?')
    cabin_shared=cabin_known&(cabins[left]==cabins[right])
    sleep_shared=(sleep[left]!='?')&(sleep[left]==sleep[right])
    masks={'same_prediction':same,'opposite_prediction':~same,'same_cabin':cabin_shared,
           'different_known_cabins':cabin_known&~cabin_shared,'same_cabin_sleep_and_prediction':cabin_shared&sleep_shared&same}
    def count(labels):
        both=labels[left]&labels[right]
        return np.array([both[mask].sum() for mask in masks.values()])
    rng=np.random.default_rng(7702)
    observed=count(bundle['err'])
    placebo=np.stack([count(permute_errors(bundle,rng)) for _ in range(75)])
    return {name:dict(pairs=int(mask.sum()),both_wrong=int(observed[i]),shuffle_median=float(np.median(placebo[:,i])),
                      shuffle_range=np.quantile(placebo[:,i],[.025,.975]).tolist()) for i,(name,mask) in enumerate(masks.items())}


def packed(bundle,selected,unary,target=None):
    sizes=[];bias=[];features=[];truth=[]
    for k in selected:
        q=bundle['groups'][k]
        states,pair,observed=bundle['blocks'][k]
        if target is not None:
            observed=int(target[q]@(1<<np.arange(len(q))))
        sizes.append(len(states));bias.append(states@unary[q]);features.append(pair);truth.append(observed)
    starts=np.cumsum(np.r_[0,sizes[:-1]])
    return np.concatenate(bias),np.vstack(features),np.asarray(sizes),starts,starts+truth


def normalized(logs,sizes,starts):
    top=np.maximum.reduceat(logs,starts)
    unscaled=np.exp(logs-np.repeat(top,sizes))
    norm=np.add.reduceat(unscaled,starts)
    return unscaled/np.repeat(norm,sizes),top+np.log(norm)


def coupling(bundle,selected,unary,target=None):
    b,x,n,i,truth=packed(bundle,selected,unary,target)
    ref=x[truth].sum(0)
    def objective(beta):
        s=b+x@beta
        p,norm=normalized(s,n,i)
        return norm.sum()-s[truth].sum()+8*(beta@beta), x.T@p-ref+16*beta
    opt=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True)
    assert opt.success,opt.message
    return opt.x


def train_model(b,groups):
    shuffled=permute_errors(b,np.random.default_rng(5592),groups)
    rows=np.concatenate([b['groups'][k] for k in groups])
    coefs=[fit(b[name][rows],b['prior'][rows],b['err'][rows]) for name in ['cov','extended']]
    unary=b['prior']+b['extended']@coefs[1]
    return coefs,coupling(b,groups,unary),coupling(b,groups,unary,shuffled)


def apply_model(b,groups,model,dst,nll):
    coefs,beta,null=model
    for name,feat,weights,pair in [('own','cov',coefs[0],None),('context','extended',coefs[1],None),
                                   ('joint','extended',coefs[1],beta),('shuffled','extended',coefs[1],null)]:
        unary=b['prior']+b[feat]@weights
        bias,x,n,i,truth=packed(b,groups,unary)
        score=bias if pair is None else bias+x@pair
        prob,norm=normalized(score,n,i)
        for j,ind in enumerate(groups):
            q=b['groups'][ind]
            dst[name][q]=prob[i[j]:i[j]+n[j]]@b['blocks'][ind][0]
            if len(q)>1:
                nll[name].append(float(norm[j]-score[truth[j]]))


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0]);ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    cur=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    reports={}
    for tag,ids,fold,panel,base,view in [('historical',rows,hf,h,old_native,hist),('full',np.arange(len(y)),f,p,native,cur)]:
        b=group_parts(raw,ids,fold,panel,base,view,y,g)
        groups=b['groups']
        first=np.array([q[0] for q in groups])
        fresh=~np.isin(g[ids],g[rows])
        preds={m:np.full(len(ids),np.nan) for m in ['own','context','joint','shuffled']}
        nll={m:[] for m in preds}
        for k in np.unique(fold):
            sel=np.flatnonzero(fold[first]==k)
            if tag=='historical':
                a=fold==k
                fresh[a]=~np.isin(g[ids[a]],g[ids[~a]])
            for tr,te in GroupKFold(4,shuffle=True,random_state=4654).split(sel,groups=g[ids[first[sel]]]):
                model=train_model(b,sel[tr])
                assert not np.isin(g[ids[first[sel[tr]]]],g[ids[first[sel[te]]]]).any()
                apply_model(b,sel[te],model,preds,nll)
        assert all(np.isfinite(v[view['known']]).all() for v in preds.values())
        joint_nll={m:float(np.mean(v)) for m,v in nll.items()}
        if tag=='historical':
            history_model=train_model(b,np.arange(len(groups)))
        else:
            sel=np.flatnonzero(fresh[first])
            transferred={m:np.full(len(ids),np.nan) for m in preds}
            apply_model(b,sel,history_model,transferred,{m:[] for m in transferred})
            preds.update({'history_'+m:v for m,v in transferred.items()})
        family=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        table={}
        for m,v in preds.items():
            valid=np.isfinite(v)
            edit=v>.5
            delta=edit*np.where(b['err'],1,-1)
            table[m]={name:stats(delta,mask&valid,fold,g[ids]) for name,mask in
                      [('all',view['known']),('parties',view['known']&(b['extended'][:,-5]>0)),('new',view['known']&fresh),('families',family)]}
        audit=pair_audit(b)
        reports[tag]=dict(pair_counts=audit,group_negative_loglikelihood=joint_nll,metrics=table)
        print(tag,audit,joint_nll,[(m,v['all']['fixed'],v['all']['harmed']) for m,v in table.items()],flush=True)
    DEST.with_name('clustered_booking_errors.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
