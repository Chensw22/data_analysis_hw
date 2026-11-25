"""Compare predictive marginalization of missing bills and ticket evidence in frozen TabICL."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data
from latent_bill_rounding_n26 import OUT,BILLS


def build(raw):
    frame=compact_features(raw)
    extra=frame.copy()
    with np.load(OUT/'missing_charge_estimates.npz') as z:
        n=len(raw)
        np.testing.assert_array_equal(z['passenger'][:n],raw.PassengerId)
        missing=z['missing'][:n]
        qvalues=z['spectrum'][:n]
        for j,bill in enumerate(BILLS):
            recorded=(missing.sum(1)==0)
            delta=z['spectrum_logprob'][:n,j]-z['donor_logprob'][:n,j]
            extra[bill+'_spectrum_evidence']=np.where(recorded,delta,np.nan)
        zero=z['spectrum_zero'][:n]
        extra['missing_zero_probability']=np.where(missing.sum(1)==1,(missing*zero).sum(1),np.nan)
    return frame,extra,missing,qvalues


def predict_augmented_queries(frame,y,t,q,missing,values,seed):
    from tabicl_routed_context_n23 import infer
    one=q[missing[q].sum(1)==1]
    service=missing[one].argmax(1)
    query=[frame.iloc[q]]
    for k in range(values.shape[2]):
        node=frame.iloc[one].copy()
        for j,bill in enumerate(BILLS):
            take=service==j
            node.loc[one[take],bill]=values[one[take],j,k]
        query.append(node)
    design=pd.concat([frame.iloc[t],*query],ignore_index=True)
    target=np.r_[y[t],np.zeros(len(design)-len(t),bool)]
    sample=np.arange(len(t),len(design))
    prob=infer(design,target,np.arange(len(t)),sample,seed)
    raw=prob[:len(q)]
    integrated=raw.copy()
    integrated[missing[q].sum(1)==1]=prob[len(q):].reshape(values.shape[2],len(one)).mean(0)
    return raw,integrated


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,f=data()
    frame,extra,missing,values=build(raw)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            loc=fold==k
            q=ids[loc]; other=ids[~loc]; t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[t],g[q]).any()
            path=OUT/f'charge_tabicl_{tag}_{k}.npz'
            if not path.exists():
                base,integrated=predict_augmented_queries(frame,y,t,q,missing,values,891+int(k))
                source=(ROOT/f'artifacts/nov25_passenger_state_contexts/{k}_global.npz' if tag=='full' else
                        ROOT/f'artifacts/nov26_global_cabin_assignment/tabicl_historical_{k}.npz')
                with np.load(source) as prior:
                    for name,val in [('t',t),('q',q),('train_y',y[t]),('checkpoint',DIGEST)]:
                        np.testing.assert_array_equal(prior[name],val)
                    ordinary=prior['p' if tag=='full' else 'base']
                    max_diff=float(np.max(np.abs(base-ordinary)))
                    print('reordered query maximum difference',tag,k,max_diff,flush=True)
                new=infer(extra,y,t,q,891+int(k))
                with path.with_suffix('.tmp').open('wb') as buf:
                    np.savez_compressed(buf,t=t,q=q,train_y=y[t],ordinary=ordinary,base=base,integrated=integrated,
                                        evidence=new,digest=DIGEST,max_diff=max_diff)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,value in [('t',t),('q',q),('train_y',y[t]),('digest',DIGEST)]:
                    np.testing.assert_array_equal(z[name],value)
                kept=missing[q].sum(1)!=1
                np.testing.assert_array_equal(z['integrated'][kept],z['base'][kept])
            print(tag,k,flush=True)


def score():
    from family_identity_transfer_n26 import panels_and_reference
    from outstanding_error_families_n26 import observed
    from disjoint_tree_fields_n25 import ART,DEST
    raw,y,g,f=data()
    missing=raw[BILLS].isna().sum(axis=1).to_numpy()==1
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    current=observed(np.arange(len(raw)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    output={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(raw)),f,native,current)]:
        pred={n:np.empty(len(ids)) for n in ['ordinary','base','integrated','evidence','cat_base','cat_spectrum','cat_imputed']}
        clean=~np.isin(g[ids],g[rows])
        for k in np.unique(fold):
            mask=fold==k; q=ids[mask]
            if tag=='historical':
                clean[mask]=~np.isin(g[q],g[ids[~mask]])
            with np.load(OUT/f'charge_tabicl_{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                for n in ['ordinary','base','integrated','evidence']:
                    pred[n][mask]=z[n]
            with np.load(OUT/f'charge_classifier_{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                for key,n in [('raw','cat_base'),('spectrum','cat_spectrum'),('spectrum_imputed','cat_imputed')]:
                    pred[n][mask]=z[key]
        anchor=view['anchor']
        wrong=anchor!=y[ids]
        inputs={'integrated':('integrated','base'),'evidence':('evidence','ordinary'),
                'cat_spectrum':('cat_spectrum','cat_base'),'cat_imputed':('cat_imputed','cat_base')}
        edits={}
        for n,(new,previous) in inputs.items():
            alt=pred[new]>=.5
            edits[n+'_induced']=(alt!=anchor)&(alt!=(pred[previous]>=.5))
            edits[n+'_quarter']=(anchor==(nat>=.5))&((.75*nat+.25*pred[new]>=.5)!=anchor)
        edits['two_evidence']=edits['evidence_induced']&edits['cat_spectrum_induced']
        edits['two_imputations']=edits['integrated_induced']&edits['cat_imputed_induced']
        family=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        counts={}
        for n,edit in edits.items():
            per={}
            for scope,mask in [('all',view['known']),('new',view['known']&clean),('families',family),
                               ('missing_one',view['known']&missing[ids])]:
                keep=edit&mask
                per[scope]=dict(rows=int(mask.sum()),fixed=int((keep&wrong).sum()),harmed=int((keep&~wrong).sum()),
                                by_fold=[int((keep&wrong&(fold==k)).sum())-int((keep&~wrong&(fold==k)).sum()) for k in np.unique(fold)])
            counts[n]=per
        output[tag]=counts
        print(tag,[(n,s['all']['fixed'],s['all']['harmed']) for n,s in counts.items()],flush=True)
    DEST.with_name('tabicl_reconstructed_charges.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','score'])
    {'train':train,'score':score}[p.parse_args().action]()
