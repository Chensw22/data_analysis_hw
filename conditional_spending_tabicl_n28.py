"""Frozen TabICLv2 on registered fields versus five context-relative receipt percentiles."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from full_roster_booking_validation_n23 import ROOT,OUT as BASE_CACHE,SEED,data

CACHE=ROOT/'artifacts/nov28_conditional_spending_tabicl'
FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
REPORT=ROOT/'reports/nov23_matched_learning_curve/conditional_spending_tabicl_n28.json'
PARENT='proposal_n27_directional_foundation'


def raw13(frame):
    x=frame.drop(columns=['Cabin','PassengerId']).copy()
    x[['Deck','CabinNumber','Side']]=frame.Cabin.str.split('/',expand=True)
    x.CabinNumber=pd.to_numeric(x.CabinNumber)
    cols=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
    x[cols]=x[cols].fillna('__MISSING__').astype(str)
    assert len(x.columns)==13 and not any('Name' in c for c in x)
    return x


def percentiles(frame,t,groups):
    # Midrank numerators and denominators for each hierarchy level. Withdraw the
    # context passenger's entire booking from every level, for both training and prediction.
    origin=frame.HomePlanet.fillna('?').astype(str).to_numpy()
    age=np.digitize(frame.Age.fillna(-100),[0,13,19,30,45,65])
    active=frame[FEES].gt(0).sum(1).to_numpy()
    n=len(frame)
    levels=[np.zeros(n,int),origin,np.array([f'{p}/{k}' for p,k in zip(origin,active)]),
            np.array([f'{p}/{k}/{b}' for p,k,b in zip(origin,active,age)])]
    def counts(values,codes):
        num=np.zeros(n); den=np.zeros(n)
        for cell in np.unique(codes):
            use=t[(codes[t]==cell)&np.isfinite(values[t])]
            q=np.flatnonzero((codes==cell)&np.isfinite(values))
            comparison=np.sort(values[use])
            num[q]=(np.searchsorted(comparison,values[q],side='left')+
                    np.searchsorted(comparison,values[q],side='right'))*.5
            den[q]=len(comparison)
            # Group-disjoint query orders require no subtraction. For training
            # members, subtract only their own booking members in this same cell.
            if len(use):
                for _,member in pd.Series(use,index=groups[use]).groupby(level=0,sort=False):
                    idx=member.to_numpy(); a=values[idx]
                    num[idx]-=((a[:,None]>a)+(a[:,None]==a)*.5).sum(1)
                    den[idx]-=len(idx)
        return num,den
    columns=[]
    for c in FEES:
        values=frame[c].to_numpy(float); prior=np.full(n,.5)
        for j,level in enumerate(levels):
            num,den=counts(values,level)
            mass=2. if j==0 else 24.
            prior=(num+mass*prior)/(den+mass)
        prior[~np.isfinite(values)]=np.nan
        columns.append(prior)
    return np.column_stack(columns)


def generate(test_only=False):
    import torch
    from tabicl_routed_context_n23 import infer,DIGEST,WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2); CACHE.mkdir(exist_ok=True)
    raw,y,g,fold=data();n=len(y)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True) if test_only else raw
    x=raw13(frame); groups=frame.PassengerId.str[:4].to_numpy()
    for f in ([8] if test_only else range(8)):
        t=np.flatnonzero(fold!=f)
        q=np.flatnonzero(fold==f) if f<8 else np.arange(n,len(frame))
        assert not np.isin(groups[t],groups[q]).any()
        path=CACHE/f'fold_{f}.npz'
        if path.exists():
            with np.load(path) as z:
                for k,v in [('query',q),('train',t),('truth',y[t]),('ids',frame.PassengerId.iloc[q].to_numpy(str)),('digest',DIGEST)]:
                    np.testing.assert_array_equal(z[k],v)
            continue
        ranks=percentiles(frame,t,groups)
        aug=x.copy()
        for j,c in enumerate(FEES):aug[c+'_percentile']=ranks[:,j]
        p=infer(aug,y,t,q,SEED+f)
        record=dict(query=q,train=t,truth=y[t],ids=frame.PassengerId.iloc[q].to_numpy(str),digest=DIGEST,p=p)
        if f==0 or test_only:
            direct=infer(x,y,t,q,SEED+f)
            if not test_only:
                with np.load(BASE_CACHE/'tabicl.npz') as z:
                    np.testing.assert_allclose(direct,z['p'][q],atol=2e-6,rtol=0)
            record['direct']=direct
        np.savez_compressed(path,**record)
        print(f,len(t),len(q),float(p.mean()),flush=True)


def rule(parent,reference,p,q):
    odds=lambda s:logit(s.clip(.003,.997))
    sign=np.where(parent,1.,-1.)
    strength=np.maximum(1.e-8,abs(odds(reference)))
    opp=(q>=.5)!=parent
    return {'standalone':opp,'new_dissent':opp&((p>=.5)==parent),
            'increment':opp&((sign*strength+odds(q)-odds(p)>=0)!=parent)}


def assess():
    from archive_source_harm_veto_n28 import inputs
    from export_budget_quarter_n23 import source
    from repeated_archive_misses_n26 import cohort,count
    frame,y,g,f,ids,_,_,seven,versions=inputs();n=len(y)
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    parent=versions['random'][:,0]
    _,test_dec,ph=source(PARENT,frame.iloc[n:]);np.testing.assert_array_equal(parent[n:],test_dec)
    with np.load(BASE_CACHE/'tabicl.npz') as z:
        for k,v in [('passenger',ids[:n]),('fold',f),('y',y)]:np.testing.assert_array_equal(z[k],v)
        plain=z['p']
    pred=np.empty(n)
    for k in range(8):
        with np.load(CACHE/f'fold_{k}.npz') as z:
            q=np.flatnonzero(f==k);np.testing.assert_array_equal(z['query'],q)
            t = np.flatnonzero(f != k)
            np.testing.assert_array_equal(z['train'], t)
            np.testing.assert_array_equal(z['truth'], y[t])
            np.testing.assert_array_equal(z['ids'], ids[q])
            assert not np.isin(g[t], g[q]).any()
            pred[q] = z['p']
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    same7=((seven>=.5)==parent[:,None]).all(1)
    same4=np.logical_and.reduce([(b==parent[:,None]).all(1) for b in versions.values()])
    err=parent[:n]!=y
    scopes=dict(all=known,fresh=known&fresh,seven=known&same7[:n],four=known&same4[:n],
                four_and_seven=known&same7[:n]&same4[:n])
    score=lambda p: float(np.mean((p>=.5)==y))
    report=dict(n=n,known_rows=int(known.sum()),base_13=score(plain),augmented_18=score(pred),
                vs_13=count((plain>=.5)!=(pred>=.5),np.ones(n,bool),(plain>=.5)!=y),parent_sha=ph,
                parent_known=float(np.mean(parent[:n][known]==y[known])),rules={},
                provenance={'backbone':'Frozen TabICL v2, two estimators, CPU',
                    'raw_inputs':13,'additional_inputs':5,'name_used':False,
                    'comparison':'Same eight shuffled whole-booking folds, seeds and contexts. Original 13-field fold 0 recomputed and matched cache within 2e-6.',
                    'parent':PARENT,'known_subset':'8,532 rows with verified incumbent local decisions',
                    'rank_reference':'Training bookings only. All members of the same booking removed for context ranks. Global, origin, origin/nonzero-service-count, then origin/count/age-band smoothed midranks.',
                    'rules':'standalone uses the 18 fields alone; new_dissent only for disagreements introduced by the five new fields; increment applies the new-minus-old logit change to a signed median reference confidence.',
                    'caution':'Single eight-fold split, previously examined competition labels. Reuse of the same folds does not constitute an independent second validation.'})
    report['cohorts']={scope:{'rows':int(mask.sum()),'parent_errors':int((mask&err).sum()),
                'base13_vs_parent':count((plain>=.5)!=parent[:n],mask,err),
                'aug18_vs_parent':count((pred>=.5)!=parent[:n],mask,err),
                'net_changes_18_vs_13':count((pred>=.5)!=(plain>=.5),mask,(plain>=.5)!=y)} for scope,mask in scopes.items()}
    report['standalone_fold_scores']=[]
    for k in range(8):
        m=f==k
        report['standalone_fold_scores'].append({'fold':k,'n':int(m.sum()),'base13':int(((plain[m]>=.5)==y[m]).sum()),
                'aug18':int(((pred[m]>=.5)==y[m]).sum())})
    ref=np.median(seven[:,1:4],axis=1)
    for k,v in rule(parent[:n],ref[:n],plain,pred).items():
        changes=np.where(err,1,-1)*v
        report['rules'][k]={s:count(v,m,err) for s,m in scopes.items()}
        report['rules'][k].update(known_accuracy=float(np.mean((parent[:n]^v)[known]==y[known])),
               fold_net=[int(changes[known&(f==i)].sum()) for i in range(8)])
    np.savez_compressed(CACHE/'aligned_local.npz',passenger=ids[:n],fold=f,y=y,plain=plain,augmented=pred)
    REPORT.write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('stage',choices=['train','test','assess']);stage=arg.parse_args().stage
    if stage=='assess':assess()
    else: generate(stage=='test')
