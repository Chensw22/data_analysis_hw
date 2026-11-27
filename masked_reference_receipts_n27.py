"""Match reference bills to naturally hidden services while keeping every query field unchanged."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

from matched_state_context_n26 import inputs,one_per_order,ROOT
from exaone_validation_data import compact_features,CATEGORICAL
from price_implied_deck_n26 import FEES

ART=ROOT/'artifacts'
OUT=ART/'nov27_masked_reference_receipts'
REPORT=ROOT/'reports/nov23_matched_learning_curve/masked_reference_receipts.json'
SIZE=1536


def bank_for(t,groups,fold,replica):
    ordering=one_per_order(t,groups,np.random.default_rng(7233+fold))
    bank=np.sort(ordering[SIZE*replica:SIZE*(replica+1)])
    assert len(bank)==SIZE
    return bank


def train(replica=0):
    import torch
    from catboost import CatBoostClassifier
    from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    frame,y,g,f,_=inputs()
    x=compact_features(frame)
    assert 'Name' not in x and 'PassengerId' not in x
    mask=x[FEES].isna().to_numpy()
    n=len(y)
    out=OUT if replica==0 else OUT/'replica'
    out.mkdir(parents=True,exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(frame))
        q=q[mask[q].any(1)]
        seed=7233+k
        bank=bank_for(t,g,k,replica)
        assert len(bank)==len(np.unique(g[bank]))==SIZE and not np.isin(g[t],g[q]).any()
        for j,name in enumerate(['original',*FEES]):
            path=out/f'fold{k}_{name}.npz'
            if path.exists():
                with np.load(path) as z:
                    for field,val in [('bank',bank),('labels',y[bank]),('q',q),('ids',frame.PassengerId.iloc[q]),('sha',DIGEST)]:
                        np.testing.assert_array_equal(z[field],val)
                continue
            view=x.copy()
            if j:
                view.loc[bank,name]=np.nan
            pd.testing.assert_frame_equal(view.iloc[q],x.iloc[q])
            p=infer(view,y,bank,q,seed)
            model=CatBoostClassifier(iterations=380,depth=5,learning_rate=.055,l2_leaf_reg=6,thread_count=2,
                                     random_seed=3027,cat_features=CATEGORICAL,verbose=False,allow_writing_files=False)
            model.fit(view.iloc[bank],y[bank])
            c=model.predict_proba(view.iloc[q])[:,1]
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,bank=bank,labels=y[bank],q=q,ids=frame.PassengerId.iloc[q].to_numpy(str),
                                sha=DIGEST,tab=p,cat=c)
            tmp.replace(path)
        print(k,flush=True)


def report(replica=0):
    out=OUT if replica==0 else OUT/'replica'
    from scipy.special import expit,logit
    from booking_mix_confidence_transfer_n26 import partition
    from directional_foundation_transfer_n27 import calibrate
    from repeated_archive_misses_n26 import count,cohort
    from full_roster_booking_validation_n23 import data
    raw,y,g,f=data()
    base161,blind,known,fresh=cohort(raw,y,f)
    frame,yy,groups,ff,_=inputs()
    np.testing.assert_array_equal(y,yy)
    np.testing.assert_array_equal(f,ff)
    ids=frame.PassengerId.to_numpy(str)
    mask=frame[FEES].isna().to_numpy()
    incomplete=mask.any(1)
    n=len(y)
    assert (incomplete[:n].sum(),incomplete[n:].sum())==(908,455)
    scores={}
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(ids))
        q=q[incomplete[q]]
        bank=bank_for(t,groups,k,replica)
        for arch in ('tab','cat'):
            pp=[]
            for name in ['original',*FEES]:
                with np.load(out/f'fold{k}_{name}.npz') as z:
                    for key,val in [('q',q),('ids',ids[q]),('labels',y[bank]),('bank',bank)]:
                        np.testing.assert_array_equal(z[key],val)
                    pp.append(z[arch])
            observed=np.stack(pp[1:],axis=1)
            right=np.empty(len(q));wrong=np.empty(len(q));wrong_all=np.empty(len(q))
            # For the 47 double-missing bills, average the two single-omission experts.
            for pos,m in enumerate(mask[q]):
                hidden=np.flatnonzero(m)
                donors=[]
                for j in hidden:
                    alt=next((j+d)%len(FEES) for d in range(1,len(FEES)) if not m[(j+d)%len(FEES)])
                    donors.append(alt)
                right[pos]=observed[pos,hidden].mean()
                wrong[pos]=observed[pos,donors].mean()
                wrong_all[pos]=observed[pos,~m].mean()
            for name,vals in [('original',pp[0]),('matched',right),('wrong',wrong),('wrong_mean',wrong_all)]:
                scores.setdefault(arch+'_'+name,np.full(len(ids),np.nan))[q]=vals
    with np.load(ART/'nov27_partial_receipt_deck/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        aug=z['augmented'];point=z['base']
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        first=z['champion111'].astype(bool)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['names'][-3:],['Mitra','TabICL fare','Nori fare'])
        median=np.median(z['scores'][:,-3:],axis=1)
    side=partition(g,'directional_foundation',4)
    risk=np.where(first,1-median[:n],median[:n])
    delta=np.zeros(n,bool)
    for k in range(8):
        for j in range(4):
            t=np.flatnonzero(known&(f==k)&(side!=j));q=np.flatnonzero((f==k)&(side==j))
            cut=calibrate(risk,first,y,t,1.)
            delta[q]=risk[q]>=cut[first[q].astype(int)]
    new=pd.read_csv(ART/'proposal_n27_directional_foundation/submission.csv')
    np.testing.assert_array_equal(new.PassengerId,ids[n:])
    anchor=np.r_[first^delta,new.Transported.to_numpy()]
    odds=lambda p:logit(p.clip(.005,.995))
    native=np.where(anchor,1.,-1.)*abs(odds(median))
    flags={key:incomplete&((p>=.5)!=anchor) for key,p in scores.items()}
    for ctrl in ('original','wrong','wrong_mean'):
        flags['tab_relative_'+ctrl]=incomplete&((native+odds(scores['tab_matched'])-odds(scores['tab_'+ctrl])>=0)!=anchor)
    flags['tab_exact_only']=flags['tab_matched']&~flags['tab_original']&~flags['tab_wrong_mean']
    flags['raw_pair']=flags['tab_original']&flags['cat_original']
    flags['masked_pair']=flags['tab_matched']&flags['cat_matched']
    flags['wrong_pair']=flags['tab_wrong']&flags['cat_wrong']
    flags['unique_pair']=flags['masked_pair']&~flags['raw_pair']&~flags['wrong_pair']
    correction=((aug>=.5)!=anchor)&((point>=.5)==anchor)
    flags['tree_augmented']=incomplete&correction
    flags['augmentation_confirmed']=flags['tab_matched']&correction
    flags['independent_shift']=flags['tab_relative_original']&flags['tab_relative_wrong']&correction
    if replica==1:
        with np.load(OUT/'aligned.npz') as other:
            for field,target in [('passenger',ids),('fold',f),('y',y),('anchor',anchor)]:
                np.testing.assert_array_equal(other[field],target)
            names=('tab_matched','tab_original','tab_wrong_mean','masked_pair','unique_pair','independent_shift',
                   'augmentation_confirmed','tab_relative_original','tab_relative_wrong_mean')
            for key in names:
                flags['replicate_'+key]=flags[key]&other['action_'+key]
            for ctrl in ('original','wrong_mean'):
                gain0=odds(other['tab_matched'])-odds(other['tab_'+ctrl])
                gain1=odds(scores['tab_matched'])-odds(scores['tab_'+ctrl])
                flags['mean_gain_'+ctrl]=incomplete&((native+(gain0+gain1)/2>=0)!=anchor)
            flags['replicated_extra']=flags['replicate_tab_matched'] & ~flags['replicate_tab_original'] & ~flags['replicate_tab_wrong_mean']
            flags['all_masked_with_tree']=flags['replicate_tab_matched'] & ((aug>=.5)!=anchor)
            flags['replicated_with_tree']=flags['replicated_extra'] & ((aug>=.5)!=anchor)
            flags['agreed_gain_with_tree']=flags['mean_gain_original'] & flags['mean_gain_wrong_mean'] & ((aug>=.5)!=anchor)
    scope=dict(all=known&incomplete[:n],fresh=known&fresh&incomplete[:n],common=blind&incomplete[:n]&(anchor[:n]==base161),
               one_missing=known&(mask[:n].sum(1)==1),two_missing=known&(mask[:n].sum(1)==2))
    scope.update({name:known&mask[:n,i]&(mask[:n].sum(1)==1) for i,name in enumerate(FEES)})
    report=dict(inputs=FEES,reference_orders=SIZE,multi_policy='mean over each missing service; cyclic next observed service for control',
                accuracy={k:float(((p[:n][incomplete[:n]]>=.5)==y[incomplete[:n]]).mean()) for k,p in scores.items()},rules={})
    for k,v in flags.items():
        d=v[:n]*np.where(anchor[:n]!=y,1,-1)
        report['rules'][k]=dict(test_changes=int(v[n:].sum()),folds=[int(d[known&(f==j)].sum()) for j in range(8)],
            scopes={s:count(v[:n],m,anchor[:n]!=y) for s,m in scope.items()})
    np.savez_compressed(out/'aligned.npz',passenger=ids,y=y,fold=f,anchor=anchor,**scores,**{'action_'+k:v for k,v in flags.items()})
    path=REPORT if replica==0 else REPORT.with_name('masked_reference_receipts_replica.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'],[(k,v['scopes']['all']) for k,v in report['rules'].items()])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','report'])
    p.add_argument('--replica',type=int,choices=[0,1],default=0)
    args=p.parse_args()
    if args.mode=='train':train(args.replica)
    else:report(args.replica)
