"""Fixed label-free party cabin-position features with a matched CPU residual control."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import signal


def runtime():
    if any(n in sys.modules for n in ['numpy','scipy','sklearn','lightgbm','torch']):raise RuntimeError('Entry before numeric imports required.')
    if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):raise RuntimeError('Duplicate runtime tolerance forbidden.')
    env={**os.environ,'KMP_DUPLICATE_LIB_OK':'FALSE'}
    if sys.platform=='darwin':
        library=Path(importlib.util.find_spec('sklearn').origin).parent/'.dylibs'
        if not (library/'libomp.dylib').is_file() or not library.resolve().is_relative_to(Path(sys.prefix).resolve()):raise RuntimeError('Existing environment libomp required.')
        current=env.get('DYLD_FALLBACK_LIBRARY_PATH','')
        if str(library) not in current.split(':'):env['DYLD_FALLBACK_LIBRARY_PATH']=str(library)+((':'+current) if current else '')
    if any(os.environ.get(k)!=v for k,v in env.items()):os.execve(sys.executable,[sys.executable,*sys.argv],env)


if __name__=='__main__':runtime()

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import nonlinear_error_corrector as residual
import exception_error_detector_nov18 as reporting

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/party_position'
PLAN=REPORT/'protocol.json'
OUT=ROOT/'artifacts/party_position'
NAMES=['PartyOrderFraction','PartySameCabinOrderFraction','PartyCabinLeader',
       'PartyCabinBlockFraction','PartyKnownCabinFraction','PartyCabinFragmentation',
       'PartyOtherDeckFraction','PartyOtherSideFraction','PartyCabinNumberRankFraction',
       'PartyCabinDistanceFromMinLog','PartyCabinDistanceToMaxLog',
       'PartyPreviousSameCabin','PartyNextSameCabin']
sha,write=residual.sha,residual.write


def positions(roster):
    if set(roster.columns)!={'PassengerId','Cabin'} or not roster.PassengerId.is_unique:raise ValueError('Only unique PassengerId and Cabin attributes allowed.')
    parsed=roster.PassengerId.str.extract(r'^(\d+)_(\d+)$')
    if parsed.isna().any().any():raise ValueError('Invalid passenger identifiers.')
    group=parsed[0].to_numpy();member=parsed[1].astype(int).to_numpy()
    cabins=roster.Cabin.to_numpy();known=roster.Cabin.notna().to_numpy()
    parts=roster.Cabin.str.split('/',expand=True).reindex(columns=[0,1,2])
    deck=parts[0].to_numpy();side=parts[2].to_numpy();number=pd.to_numeric(parts[1],errors='coerce').to_numpy(float)
    if np.any(known&~np.isfinite(number)):raise ValueError('Malformed known cabin.')
    out=np.full((len(roster),len(NAMES)),np.nan)
    groups={}
    for i,key in enumerate(group):groups.setdefault(key,[]).append(i)
    for indices in groups.values():
        ix=np.asarray(sorted(indices,key=lambda i:member[i]));n=len(ix)
        if len(np.unique(member[ix]))!=n:raise ValueError('Duplicate member positions.')
        valid=ix[known[ix]];distinct=list(dict.fromkeys(cabins[valid].tolist()))
        for rank,i in enumerate(ix):
            out[i,0]=rank/(n-1) if n>1 else 0.
            out[i,4]=len(valid)/n
            out[i,5]=len(distinct)/len(valid) if len(valid) else np.nan
            if not known[i]:continue
            same=valid[cabins[valid]==cabins[i]];at=int(np.flatnonzero(same==i)[0])
            out[i,1]=at/(len(same)-1) if len(same)>1 else 0.
            out[i,2]=float(at==0)
            out[i,3]=distinct.index(cabins[i])/(len(distinct)-1) if len(distinct)>1 else 0.
            others=valid[valid!=i]
            if len(others):
                out[i,6]=np.mean(deck[others]!=deck[i]);out[i,7]=np.mean(side[others]!=side[i])
            lane=valid[(deck[valid]==deck[i])&(side[valid]==side[i])]
            numbers=number[lane]
            ranknumber=np.sum(numbers<number[i])+.5*(np.sum(numbers==number[i])-1)
            out[i,8]=ranknumber/(len(lane)-1) if len(lane)>1 else 0.
            out[i,9]=np.log1p(number[i]-numbers.min());out[i,10]=np.log1p(numbers.max()-number[i])
            if rank>0 and known[ix[rank-1]]:out[i,11]=float(cabins[ix[rank-1]]==cabins[i])
            if rank+1<n and known[ix[rank+1]]:out[i,12]=float(cabins[ix[rank+1]]==cabins[i])
    if np.isinf(out).any():raise ValueError('Infinite derived attribute.')
    return pd.DataFrame(out,columns=NAMES,index=roster.index)


def check():
    frame=pd.DataFrame({'PassengerId':['0001_01','0001_02','0001_03','0002_01','0002_02','0003_01'],
                        'Cabin':['F/1/P','F/2/P','G/3/P','F/1/S','F/1/S',None]})
    f=positions(frame)
    np.testing.assert_array_equal(f.PartyOrderFraction,[0,.5,1,0,1,0])
    assert f.PartyOtherDeckFraction.iloc[0]==.5 and f.PartyOtherDeckFraction.iloc[2]==1.
    assert f.PartySameCabinOrderFraction.iloc[4]==1 and f.PartyCabinLeader.iloc[4]==0
    assert f.PartyNextSameCabin.iloc[3]==1 and f.PartyPreviousSameCabin.iloc[1]==0
    assert f.PartyKnownCabinFraction.iloc[5]==0 and np.isnan(f.PartyCabinBlockFraction.iloc[5])
    order=[3,0,5,2,4,1];permuted=frame.iloc[order].reset_index(drop=True)
    a=f.assign(pid=frame.PassengerId).set_index('pid').sort_index()
    b=positions(permuted).assign(pid=permuted.PassengerId).set_index('pid').sort_index()
    pd.testing.assert_frame_equal(a,b)
    try:positions(frame.assign(Transported=True))
    except ValueError:pass
    else:raise AssertionError('Label-containing roster accepted.')
    print('Attribute-only position/crossdeck/cabin-order/permutation/missing checks passed.')


def check_protocol():
    plan=json.loads(PLAN.read_text())
    if plan['new_columns']!=NAMES or plan['parameters']!=residual.PARAMS or plan['rounds']!=residual.ROUNDS:raise RuntimeError('Fixed recipe differs.')
    for name,digest in {**plan['source_hashes'],**plan['historical_hashes']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    return plan


def run():
    plan=check_protocol()
    if OUT.exists() or (REPORT/'result.json').exists():raise FileExistsError('Fresh fixed experiment only.')
    if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('FALSE runtime required.')
    OUT.mkdir(parents=True);started=time.monotonic()
    def timeout(*_):raise TimeoutError('CPU300seconds exceeded.')
    oldhandler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,300)
    try:
        import lightgbm as lgb
        inputs,splits,fold_ids,bases,evidence=residual.overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra']
        x=residual.features()
        roster=pd.concat([pd.read_csv(ROOT/'data/raw'/n,usecols=['PassengerId','Cabin'],dtype={'PassengerId':str}) for n in ['train.csv','test.csv']],ignore_index=True)
        extra=positions(roster);augmented=pd.concat([x,extra],axis=1)
        description={col:dict(known_rows=int(extra[col].notna().sum()),unique_known_values=int(extra[col].nunique()),
            minimum=None if extra[col].notna().sum()==0 else float(extra[col].min()),maximum=None if extra[col].notna().sum()==0 else float(extra[col].max())) for col in NAMES}
        write(REPORT/'unlabeled_feature_summary.json',description)
        oldpath=ROOT/'reports/error_targeted_20261117/residual_runtime_fixed_result.json';oldrecord=json.loads(oldpath.read_text())
        oldnpz=ROOT/oldrecord['oof_path']
        if oldrecord['status']!='complete' or sha(oldnpz)!=oldrecord['oof_sha256']:raise RuntimeError('Historical matched residual changed.')
        with np.load(oldnpz,allow_pickle=False) as z:
            for key,expected in [('dev_ids',dev),('fold_id',fold_ids),('raw',raw)]:np.testing.assert_array_equal(z[key],expected)
            oldcontrol=z['nonlinear'].copy()
        frozen=residual.relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json');residual.relational.nested.verify_hashes(frozen['sources'])
        evidence.update(frozen['sources']);evidence.update(plan['historical_hashes'])
        control=np.full(len(dev),np.nan);candidate=control.copy();folds=[]
        for fold,(a,b) in enumerate(splits):
            inner=residual.relational.load_inner(fold,a,b,dev,y,evidence)
            models={}
            for name,features,target in [('control',x,control),('party_position',augmented,candidate)]:
                train,query,vocab=residual.categorical_frames(features.iloc[dev[a]],features.iloc[dev[b]])
                dataset=lgb.Dataset(train,label=y[a],init_score=residual.relational.logit(inner),categorical_feature=residual.CATS,free_raw_data=True)
                model=lgb.train({**residual.PARAMS,'seed':20261112+fold},dataset,num_boost_round=residual.ROUNDS)
                p=expit(residual.relational.logit(raw[b])+model.predict(query,raw_score=True));target[b]=p
                path=OUT/f'{name}_fold{fold}.txt';model.save_model(str(path));schema=OUT/f'{name}_fold{fold}_schema.json'
                write(schema,dict(columns=list(features.columns),categories=vocab))
                saved=json.loads(schema.read_text());_,again,_=residual.categorical_frames(features.iloc[dev[a]],features.iloc[dev[b]],saved['categories'])
                reload=lgb.Booster(model_file=str(path));q=expit(residual.relational.logit(raw[b])+reload.predict(again,raw_score=True))
                np.testing.assert_array_equal(p,q)
                models[name]=dict(model_sha256=sha(path),schema_sha256=sha(schema),reload_difference=0.,iterations=model.current_iteration(),
                    gain_importance=dict(zip(features.columns,model.feature_importance(importance_type='gain').tolist())))
                del model,reload,dataset
            np.testing.assert_array_equal(control[b],oldcontrol[b])
            np.savez_compressed(OUT/f'fold{fold}_rows.npz',train_ids=dev[a],query_ids=dev[b],inner_raw=inner)
            row=dict(fold=fold,models=models,exact_old_control=True,metrics={k:residual.relational.score(v[b],y[b]) for k,v in [('raw',raw),('control',control),('party',candidate)]},
                paired={k:residual.relational.pair(candidate[b],v[b],y[b]) for k,v in [('raw',raw),('control',control)]})
            folds.append(row);write(REPORT/f'fold{fold}.json',row);print(json.dumps(row),flush=True)
        if not np.isfinite(candidate).all():raise RuntimeError('Incomplete OOF.')
        check_protocol();masks=reporting.reporting_masks(dev,fold_ids,y,inputs,bases,evidence)
        attr=x.iloc[dev];masks.update(Earth_sleep=(attr.HomePlanet.eq('Earth')&attr.CryoSleep.eq('True')).to_numpy(),Earth_awake=(attr.HomePlanet.eq('Earth')&attr.CryoSleep.eq('False')).to_numpy())
        probs={'raw':raw,'control':control,'party':candidate}
        subsets={n:dict(rows=int(m.sum()),correct={k:int(((p[m]>=.5)==y[m]).sum()) for k,p in probs.items()},
            paired={k:residual.relational.pair(candidate,p,y,m) for k,p in [('raw',raw),('control',control)]},
            confusion={k:reporting.confusion(p,y,m) for k,p in probs.items()}) for n,m in masks.items()}
        np.savez_compressed(OUT/'oof.npz',dev_ids=dev,fold_id=fold_ids,passenger_ids=np.asarray(inputs['passengers'],dtype=str),**probs,**masks)
        check_protocol()
        for n,d in evidence.items():
            if sha(ROOT/n)!=d:raise RuntimeError('Evidence changed: '+n)
        result=dict(status='complete_fixed_adaptive_threefold',folds=folds,metrics={k:residual.relational.score(p,y) for k,p in probs.items()},
            paired={k:residual.relational.pair(candidate,p,y) for k,p in [('raw',raw),('control',control)]},subsets=subsets,
            old_control_bitwise_equal=True,feature_summary=description,oof_path=str((OUT/'oof.npz').relative_to(ROOT)),oof_sha256=sha(OUT/'oof.npz'),
            protocol_sha256=sha(PLAN),seconds=time.monotonic()-started,no_Name=True,no_base_retraining=True,no_holdout_targets=True,no_upload=True,limitations=plan['limitations'])
        write(REPORT/'evidence.json',evidence);write(REPORT/'result.json',result)
        print(json.dumps({k:result[k] for k in ['metrics','paired','seconds']},indent=2))
    except BaseException as error:
        if (OUT/'oof.npz').exists():(OUT/'oof.npz').rename(OUT/'oof.partial.npz')
        write(REPORT/'failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldhandler)


if __name__=='__main__':
    if sys.argv[1:]==['--check']:check()
    elif sys.argv[1:]:raise ValueError('Only --check supported.')
    else:run()
