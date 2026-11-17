"""v99: original15 inputs plus two self-excluded peer CryoSleep statistics, fixed15 full-data updates, direct export.

15 is inherited from the already submitted v81 final refit, not a new selection
result. No inner split, validation loss, CV, resource probe or upload is run.
"""
import os
import sys
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):
    raise RuntimeError('Fresh FALSE-only process required.')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import torch
import mitra_candidate as base
import mitra_full_episode as training
import prepare_v84_online_first as common
import mitra_counts_fixed15_online_first as prior

ROOT=Path(__file__).resolve().parent
NAME='v99_Mitra_peercryo_fixed15_ready'
PROPOSAL='proposal_ready_peercryo17'
POLICY='reports/online_iteration_20261117/next_round_peercryo_policy.json'
EXTRA=['GroupSize','CabinOccupants','PeerCryoSleepRate','PeerKnownCryoCount']
COLUMNS=list(base.COMPACT)+EXTRA
SEED,STEPS,BUDGET=20261112,15,7200
INHERITED={'mitra_candidate.py','mitra_full_episode.py','mitra_full_episode_probe.py','iterate.py','experiment.py'}
REFERENCE='artifacts/proposal_nov16_full_episode'
sha,read,write=common.sha,common.read,common.write
if __name__=='__main__':
    sys.modules['mitra_peercryo_fixed15_ready']=sys.modules[__name__]


def require_false():
    if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('FALSE runtime changed.')


class PeerCryoRoster:
    """Fixed public-roster membership; PassengerIds only align rows, never features."""
    def __init__(self, raw, plain):
        raw=raw[['PassengerId','Cabin','CryoSleep']].copy()
        if (len(raw)!=len(plain) or not raw.index.equals(plain.index) or not raw.index.is_unique
                or raw.PassengerId.isna().any() or not raw.PassengerId.is_unique):
            raise ValueError('Unique aligned complete roster IDs required.')
        self.passenger_ids=raw.PassengerId.astype(str).copy()
        self.plain=plain[base.COMPACT].copy(deep=True)
        self.raw=raw.copy(deep=True)
        groups=self.passenger_ids.str.split('_').str[0]
        self.counts=pd.DataFrame({'GroupSize':groups.map(groups.value_counts()).astype(float),
            'CabinOccupants':raw.Cabin.map(raw.Cabin.value_counts(dropna=True)).astype(float)},index=raw.index)
        if not raw.CryoSleep.dropna().isin([True,False]).all():
            raise ValueError('CryoSleep must be observed boolean or missing.')
        sleeping=raw.CryoSleep.astype(float)
        known=sleeping.notna().astype(int)
        numerator=sleeping.fillna(0).groupby(groups).transform('sum')-sleeping.fillna(0)
        denominator=known.groupby(groups).transform('sum')-known
        self.counts['PeerCryoSleepRate']=numerator/denominator.replace(0,np.nan)
        self.counts['PeerKnownCryoCount']=denominator.astype(float)
        self.roster_sha256=hashlib.sha256(pd.util.hash_pandas_object(raw,index=True).to_numpy().tobytes()).hexdigest()

    def transform(self,x):
        if not x.index.is_unique or not x.index.isin(self.plain.index).all():
            raise ValueError('Query must retain unique fixed global roster indices.')
        pd.testing.assert_frame_equal(x[base.COMPACT],self.plain.loc[x.index],check_exact=True)
        return pd.concat([x[base.COMPACT].copy(),self.counts.loc[x.index]],axis=1)

    def frame_by_ids(self,ids):
        ids=pd.Index(np.asarray(ids).astype(str))
        lookup=pd.Index(self.passenger_ids.to_numpy());positions=lookup.get_indexer(ids)
        if (positions<0).any() or not ids.is_unique:raise ValueError('Unknown or repeated PassengerId.')
        return self.plain.iloc[positions]


def load_roster():
    import iterate
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda name:name!='Transported')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda name:name!='Transported')
    if len(train)!=8693 or len(test)!=4277:raise RuntimeError('Complete public roster required.')
    raw=pd.concat([train,test],ignore_index=True)
    plain,_=iterate.build_features(raw,False)
    return PeerCryoRoster(raw,plain)


class PeerCryoPreprocessor(base.MitraPreprocessor):
    def __init__(self,roster=None):
        self.roster=roster

    def fit(self,x,y,seed=SEED):
        y=base.binary_labels(y,len(x))
        if self.roster is None:self.roster=load_roster()
        self.reference=prior.CountsPreprocessor(self.roster).fit(x,y,seed)
        self.columns=list(COLUMNS);self.categories=list(base.CATEGORICAL)
        self.vocab={k:dict(v) for k,v in self.reference.vocab.items()}
        array=self.encode(x);self.all_missing=np.isnan(array).all(axis=0)
        array[:,self.all_missing]=0.
        self.upstream=base.Preprocessor(dim_embedding=None,n_classes=2,dim_output=10,
            use_quantile_transformer=False,use_feature_count_scaling=False,
            use_random_transforms=False,shuffle_classes=False,shuffle_features=False,
            random_mirror_regression=False,random_mirror_x=base.SETTINGS['random_mirror_x'],task=base.Task.CLASSIFICATION)
        previous=np.random.get_state()
        try:
            np.random.seed(seed);self.upstream.fit(array.copy(),y.copy())
        finally:np.random.set_state(previous)
        self.active_columns=np.asarray(self.columns)[~self.upstream.singular_features].tolist()
        self.n_features=self.transform(x).shape[1]
        return self

    def encode(self,x):
        return super().encode(self.roster.transform(x))

    def transform(self,x):
        result=super().transform(x)
        expected=self.reference.transform(x)
        original=[c for c in self.active_columns if c in prior.COLUMNS]
        indices=[self.active_columns.index(c) for c in original]
        if len(indices)!=expected.shape[1] or not np.array_equal(result[:,indices],expected):
            raise RuntimeError('Original15 active values/vocabulary/imputation/mirror changed.')
        return result


PeerCryoPreprocessor.__module__='mitra_peercryo_fixed15_ready'
PeerCryoRoster.__module__='mitra_peercryo_fixed15_ready'


class PeerCryoPredictor(base.MitraPredictor):
    def predict_proba_by_passenger_ids(self,ids):
        return self.predict_proba(self.preprocessor.roster.frame_by_ids(ids))


PeerCryoPredictor.__module__='mitra_peercryo_fixed15_ready'


def fit_fixed(x,y,ids,device='mps',log_path=None):
    y=base.binary_labels(y,len(x));ids=np.asarray(ids,dtype=np.int64)
    if len(x)!=8693 or ids.shape!=y.shape or not np.array_equal(ids,np.arange(8693)):
        raise ValueError('Fixed full8693 training context required.')
    require_false();base.seed_everything(SEED,device)
    prep=PeerCryoPreprocessor().fit(x,y,SEED)
    support=prep.transform(x)
    if support.shape!=(8693,17) or prep.active_columns!=COLUMNS:
        raise RuntimeError('Formal recipe requires all17 actual model input columns.')
    model=optimizer=data=targets=None
    meta=dict(status='incomplete',candidate=NAME,seed=SEED,training_rows=8693,
        training_ids_sha256=base.ids_hash(ids),labels_sha256=base.ids_hash(y),
        fixed_steps=STEPS,step_source=REFERENCE+'/fit_metadata.json',inner_selection_performed=False,
        feature_count=17,columns=COLUMNS,original15_encoding_bitwise_equal=True,
        roster_sha256=prep.roster.roster_sha256,roster_rows=len(prep.roster.plain),
        training_passenger_ids=prep.roster.passenger_ids.loc[x.index].tolist(),
        checkpoint_sha256=base.WEIGHT_SHA,query_chunk=2318,refit_history=[])
    try:
        model,config=training.fresh_model(device)
        data=torch.as_tensor(support,device=device);targets=torch.as_tensor(y,device=device)
        optimizer=base.optimizer_for(model);rng=np.random.default_rng(SEED)
        for step in range(1,STEPS+1):
            require_false()
            row=training.episode_step(model,optimizer,data,targets,ids,rng,step)
            meta['refit_history'].append(row);base.save_training_log(log_path,meta)
            print(f'{NAME} fixed update {step}/{STEPS}',flush=True)
        if len(meta['refit_history'])!=15:raise RuntimeError('Partial refit forbidden.')
        optimizer=data=targets=None;base.release_memory()
        meta.update(status='fit_complete',refit_optimizer_steps=15,full_prediction_support=True,
            parameters=sum(p.numel() for p in model.parameters()))
        predictor=PeerCryoPredictor(prep,config,model.state_dict(),support,y,device,meta,query_chunk=2318)
        base.save_training_log(log_path,meta)
        return predictor
    except BaseException as error:
        meta.update(status='failed_incomplete',error_type=type(error).__name__,error=str(error))
        base.save_training_log(log_path,meta);raise
    finally:
        model=optimizer=data=targets=None;base.release_memory()


def verify_reference():
    manifest=common.pair('proposal_nov16_full_episode','manifest.json')
    fit=read(ROOT/REFERENCE/'fit_metadata.json');reload=read(ROOT/REFERENCE/'reload_metadata.json')
    for value in (fit,reload):
        if any(value.get(k)!=v for k,v in dict(seed=SEED,training_rows=8693,selected_steps=15,refit_optimizer_steps=15,inner_optimizer_steps=50).items()):
            raise RuntimeError('Historical v81 did not use the declared final15 updates.')
    if manifest.get('export_reload_inference_verified') is not True:raise RuntimeError('Original full model incomplete.')
    for path,expected in manifest['source_hashes'].items():
        if (path in INHERITED or path.startswith('third_party/mitra/')) and sha(ROOT/path)!=expected:
            raise RuntimeError('Inherited numerical source changed: '+path)
    expected=manifest['environment_sha256']
    lock=base.environment_lock()
    if hashlib.sha256(lock.encode()).hexdigest()!=expected:raise RuntimeError('Use original v81 .venv-mitra environment.')
    base.verify_checkpoint()
    return manifest,lock


def export():
    out,report=ROOT/'artifacts'/PROPOSAL,ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():raise RuntimeError('Existing output immutable; no retry/resume.')
    policy=read(ROOT/POLICY)
    if (policy.get('versions')!=[NAME] or policy.get('allow_full_export') is not True or policy.get('allow_direct_submit') is not False
            or policy.get('requires_fresh_primary_account_quota') is not True
            or policy.get('fixed_optimizer_steps')!=15):raise RuntimeError('Exact local-ready-only export policy required.')
    previous,lock=verify_reference()
    for path,expected in policy['evidence_files'].items():
        if sha(ROOT/path)!=expected:raise RuntimeError('Fixed step/formula evidence changed: '+path)
    inherited={name for name in previous['source_hashes'] if name in INHERITED or name.startswith('third_party/mitra/')}
    names=inherited|{Path(__file__).name,'test_mitra_peercryo_fixed15_ready.py',POLICY,
        'prepare_v84_online_first.py','mitra_counts_fixed15_online_first.py','structure_counts.py','kaggle_workflow.py','requirements-mitra.lock.txt','reports/proposal_nov16_full_episode/manifest.json',
        'reports/proposal_online_first_counts15/manifest.json'}
    tracked=subprocess.check_output(['git','ls-files','-z','--',*sorted(names)],cwd=ROOT).decode()
    if set(filter(None,tracked.split('\0')))!=names or subprocess.check_output(['git','status','--porcelain','--',*sorted(names)],cwd=ROOT).strip():
        raise RuntimeError('Commit fixed sources/policy before full labels or model fit.')
    sources={name:sha(ROOT/name) for name in names}
    sources.update(policy['evidence_files'])
    for name in ('fit_metadata.json','reload_metadata.json','environment.lock.txt','model.joblib'):
        path=REFERENCE+'/'+name;sources[path]=sha(ROOT/path)
    for name in ('train','test','sample_submission'):
        path=f'data/raw/{name}.csv';sources[path]=sha(ROOT/path)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    out.mkdir(parents=True);report.mkdir(parents=True)
    started=time.monotonic();base.DEADLINE=started+BUDGET
    def timeout(*_):raise TimeoutError('Fixed15 export exceeded7200s; no fallback.')
    handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    model=restored=None
    try:
        require_false()
        if not torch.backends.mps.is_available():raise RuntimeError('Original MPS required.')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        protocol=dict(version=NAME,source_commit=commit,source_hashes=sources,policy=policy,
            checkpoint_sha256=base.WEIGHT_SHA,environment_sha256=previous['environment_sha256'],budget_seconds=BUDGET,
            algorithm='Fixed15 fresh refit; not original50step inner selection',source_final_steps_verified=True)
        write(out/'protocol.json',protocol);write(report/'protocol.json',protocol)
        (out/'environment.lock.txt').write_text(lock)
        import iterate
        import kaggle_workflow
        train,test,_=iterate.base.load_data()
        frame,_=iterate.build_features(pd.concat([train.drop(columns='Transported'),test],ignore_index=True),False)
        x,q=frame.iloc[:8693],frame.iloc[8693:]
        y=train.Transported.astype(np.int64).to_numpy();ids=np.arange(8693,dtype=np.int64)
        model=fit_fixed(x,y,ids,log_path=out/'training_log.json')
        expected_ids=test.PassengerId.astype(str).to_numpy()
        if not np.array_equal(model.preprocessor.roster.passenger_ids.loc[q.index].to_numpy(),expected_ids):
            raise RuntimeError('Query indices do not identify exact test PassengerIds.')
        first=np.asarray(model.predict_proba_by_passenger_ids(expected_ids));common.compare(first[:,1].tolist(),first[:,1].tolist())
        write(out/'fit_metadata.json',model.metadata);joblib.dump(model,out/'model.joblib')
        model.release();model=None;base.release_memory();require_false()
        restored=joblib.load(out/'model.joblib')
        if (restored.device!='cpu' or restored._engine is not None or restored.support.shape!=(8693,17)
                or restored.query_chunk!=2318 or not np.array_equal(restored.labels,y)
                or restored.preprocessor.active_columns!=COLUMNS or restored.metadata['refit_optimizer_steps']!=15):
            raise RuntimeError('Reloaded full17 context/recipe differs.')
        if not np.array_equal(restored.preprocessor.roster.passenger_ids.loc[q.index].to_numpy(),expected_ids):
            raise RuntimeError('Reloaded roster/PassengerIds changed.')
        restored.device='mps';again=np.asarray(restored.predict_proba_by_passenger_ids(expected_ids))
        difference=common.compare(first[:,1].tolist(),again[:,1].tolist())
        write(out/'reload_metadata.json',restored.metadata)
        restored.release();restored=None;base.release_memory();require_false()
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=first[:,1],reload_probability=again[:,1])).to_csv(out/'test_probabilities.csv',index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=first[:,1]>=.5)).to_csv(out/'submission.csv',index=False)
        check=kaggle_workflow.validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
        for path in (ROOT/'reports').rglob('online_attempts/*.json'):
            if read(path).get('file_sha256')==check['sha256']:raise RuntimeError('Duplicate submission already attempted.')
        for path,expected in sources.items():
            if sha(ROOT/path)!=expected:raise RuntimeError('Frozen evidence/source/data changed: '+path)
        if base.environment_lock()!=lock:raise RuntimeError('Environment changed.')
        base.verify_checkpoint();base.check_deadline()
        selection=dict(version=NAME,components=[NAME],threshold=.5,stage='ready_pending_fresh_primary_account_quota',A_eligible=None,S_supported=False,
            fixed_optimizer_steps=15,inner_selection_performed=False)
        manifest=dict(**check,status='complete',allow_direct_submit=False,requires_fresh_primary_account_quota=True,version=NAME,proposal=PROPOSAL,training_rows=8693,fit_rows=8693,
            feature_count=17,roster_rows=12970,prediction_ids_sha256=hashlib.sha256(json.dumps(expected_ids.tolist()).encode()).hexdigest(),
            threshold=.5,model_sha256=sha(out/'model.joblib'),selection=selection,
            export_reload_inference_verified=True,reload_rows=4277,reload_max_absolute_difference=difference,
            reload_threshold_decisions_equal=True,policy_path=POLICY,policy_sha256=sha(ROOT/POLICY),
            source_commit=commit,source_hashes=sources,environment_sha256=previous['environment_sha256'],
            inherited_v81_final_steps=15,inner_selection_performed=False,no_new_local_score=True,
            seconds=time.monotonic()-started,completed_at_utc=datetime.now(timezone.utc).isoformat())
        for name,value in [('selection.json',selection),('manifest.json',manifest)]:write(out/name,value);write(report/name,value)
        print(json.dumps(dict(status='complete',proposal=PROPOSAL,reload_difference=difference,sha256=check['sha256'])),flush=True)
        return manifest
    except BaseException as error:
        for directory in (out,report):
            for name in ('manifest.json','selection.json','submission.csv'):(directory/name).unlink(missing_ok=True)
            write(directory/'failure.json',dict(status='failed_not_upload_eligible',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        try:
            for active in (model,restored):
                if active is not None:active.release()
            base.release_memory()
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();export()
