"""EXAONE_v91_CAST_default8 portable CPU-context adapter; no formal screen/upload."""
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import exaone_mps_compat as compat
from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.checkpoint import load_classifier_checkpoint
from exaonetabular.presets import released_manifest
from exaonetabular._execution import _CudaExecutionPlan

torch=compat.torch
ROOT=Path(__file__).resolve().parent
NAME='EXAONE_v91_CAST_default8'
CHECKPOINT=ROOT/'artifacts/pretrained/exaone/exaone-tabular-classifier-v1_default.safetensors'
WEIGHT_SHA='2aa723e024e5c56c778d6c79cc3cee7675d1b6a03c06fa08c80e620e228b224d'
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CATEGORICAL=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
MASK=np.asarray([name in CATEGORICAL for name in COMPACT],dtype=bool)
SETTINGS=dict(seed=0,ensemble_count=8,parameter_dtype='float16',input_dtype='float32',
    preprocessing_dtype='float64',preprocessing_rng_device='cpu',aggregation_device='cpu',
    ensemble_step=1,query_step=1024,feedforward_token_chunk=524288,feature_attention_row_chunk=2147483647,
    cross_query_support_cache=False,sdpa_query_chunk=256,sdpa_long_k_boundary=256,
    sdpa_long_k_batch=1,sdpa_short_k_batch=512,mps_memory_fraction=.65,threads=4,
    support_row_limit=100000,feature_limit=500,full_support=True,threshold=.5,columns=COMPACT)
DEADLINE=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def array_hash(values):
    a=np.ascontiguousarray(values);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()


def environment_lock():
    return ''.join(f'{d.metadata["Name"]}=={d.version}\n' for d in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower()))


def check_deadline():
    if DEADLINE is not None and time.monotonic()>DEADLINE:raise TimeoutError('Frozen EXAONE resource budget exhausted.')


def verify_checkpoint():
    compat.verify_vendor()
    if CHECKPOINT.stat().st_size!=83286800 or sha(CHECKPOINT)!=WEIGHT_SHA:raise RuntimeError('Pinned EXAONE weight bytes/hash mismatch.')
    return replace(released_manifest('classification'),checkpoint_sha256=WEIGHT_SHA)


class CompactEncoder:
    def fit(self,x):
        self.columns=list(COMPACT);self.vocab={}
        for col in CATEGORICAL:
            values=x[col];missing=values.isna()|values.astype('string').eq('__MISSING__').fillna(False)
            self.vocab[col]={v:i for i,v in enumerate(sorted(set(values[~missing].astype(str))))}
        return self

    def transform(self,x):
        result=np.empty((len(x),13),dtype=np.float64)
        for i,col in enumerate(self.columns):
            if col in CATEGORICAL:
                # Unknown and missing share NaN before the unchanged official category encoder.
                values=x[col];missing=values.isna()|values.astype('string').eq('__MISSING__').fillna(False)
                result[:,i]=values.astype(str).map(self.vocab[col]).to_numpy(dtype=np.float64)
                result[missing.to_numpy(dtype=bool),i]=np.nan
            else:result[:,i]=pd.to_numeric(x[col],errors='raise').to_numpy(dtype=np.float64)
        if np.isinf(result).any():raise ValueError('Infinite attributes not supported.')
        return result


def valid_probabilities(p,rows):
    p=np.asarray(p)
    if p.shape!=(rows,2) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.allclose(p.sum(axis=1),1.,atol=2e-6,rtol=0):
        raise ValueError('Complete finite normalized binary probabilities required.')
    return p


class EXAONELocalClassifier(EXAONETabularClassifier):
    def _forward_chunked(self,support_batch,label_batch,query_batch):
        check_deadline();rows=len(self._state()['support_x'])
        if (support_batch.shape[0]!=8 or support_batch.shape[1]!=rows or label_batch.shape!=(8,rows)
                or query_batch.shape[0]!=8 or any(v.device.type!='cpu' or v.dtype!=torch.float32
                    for v in [support_batch,label_batch,query_batch])):
            raise RuntimeError('All eight full-context members must arrive from native CPU preprocessing as FP32.')
        self.forward_records=[]
        def guard(module,args,kwargs):
            check_deadline();xs,ys,xq=args[:3]
            if (xs.shape[0]!=1 or xs.shape[1]!=rows or ys.shape!=(1,rows) or xq.shape[0]!=1 or not 0<len(xq[0])<=1024
                    or any(v.dtype!=torch.float32 for v in [xs,ys,xq])
                    or any(kwargs.get(name,False) for name in ['cache_support','use_cached_support','cache_support_to_cpu','compact_cached_support'])):
                raise RuntimeError('Fixed no-cache single-view full-support execution changed.')
            self.forward_records.append(dict(support_rows=rows,query_rows=xq.shape[1],views=1,input_dtype='float32'))
        handle=self.model.register_forward_pre_hook(guard,with_kwargs=True)
        plan=_CudaExecutionPlan(ensemble_step=1,query_step=min(1024,query_batch.shape[1]),feedforward_token_chunk=524288)
        try:
            logits=self._executor()._run_execution_plan(self.model,support_batch,label_batch,query_batch,plan)
            if len(self.forward_records)!=8*((query_batch.shape[1]+1023)//1024):raise RuntimeError('Missing required view/query forward.')
            # Original inverse-class-permutation / mean-logit / softmax stays native on CPU.
            return logits.to(device='cpu')
        finally:handle.remove()


def new_classifier(*,local=True,device='cpu'):
    if device not in ['cpu','mps']:raise ValueError('Fixed CPU or MPS only.')
    manifest=verify_checkpoint();check_deadline()
    cls=EXAONELocalClassifier if local else EXAONETabularClassifier
    classifier=cls(manifest,device='cpu')
    load_classifier_checkpoint(CHECKPOINT,classifier.model,manifest,verify_checksum=True)
    if local:classifier.compatibility=compat.install_local(classifier.model)
    elif device!='cpu':raise ValueError('Unmodified baseline is CPU only.')
    classifier.model.to(device=device)
    return classifier


def fitted_identity(classifier,rows):
    state=classifier._state();prep=state['preprocessor']
    if len(state['support_x'])!=rows or len(state['support_y'])!=rows or prep.support_members.shape[:2]!=(8,rows):
        raise RuntimeError('Official preprocessing dropped views/support rows.')
    if prep.device.type!='cpu' or prep.support_members.dtype!=np.float32 or prep.signs.dtype!=np.float64:
        raise RuntimeError('Native CPU preprocessing dtype/device changed.')
    if not np.array_equal(classifier.classes_,[0,1]):raise RuntimeError('Original binary class order changed.')
    return dict(support_rows=rows,views=8,member_features=prep.support_members.shape[2],
        support_members_sha256=array_hash(prep.support_members),signs_sha256=array_hash(prep.signs),
        orders_sha256=array_hash(prep.orders),rng_sha256=array_hash(prep.rng_state.numpy()),
        target_rng_sha256=array_hash(prep.target_rng_state.numpy()))


class EXAONEPredictor:
    def __init__(self,x,y,ids,seed=0,device='cpu'):
        if seed!=0:raise ValueError('This resource recipe fixes released CPU ensemble seed0.')
        self.encoder=CompactEncoder().fit(x);self.support=self.encoder.transform(x)
        raw_y=np.asarray(y);self.labels=raw_y.astype(np.int64);self.ids=np.asarray(ids,dtype=np.int64).copy()
        if (raw_y.shape!=(len(x),) or not np.isin(raw_y,[0,1]).all() or len(np.unique(raw_y))!=2
                or self.ids.shape!=(len(x),) or len(np.unique(self.ids))!=len(x) or len(x)>100000):
            raise ValueError('Unique aligned full support and both binary classes required; no truncation.')
        self.seed=0;self.device=device;self._engine=None
        self.metadata=dict(candidate=NAME,settings=SETTINGS,training_rows=len(x),training_ids_sha256=array_hash(self.ids),
            labels_sha256=array_hash(self.labels),encoded_support_sha256=array_hash(self.support),
            categorical_mask=MASK.tolist(),checkpoint_sha256=WEIGHT_SHA,code_commit=compat.COMMIT,
            optimizer_steps=0,portable_state='CPU encoded full context, labels, ID/vocabulary and seed; reload pinned native model/preprocessing')

    def engine(self):
        if self._engine is None:
            classifier=new_classifier(local=True,device=self.device)
            classifier.fit(self.support,self.labels,categorical_mask=MASK)
            identity=fitted_identity(classifier,len(self.labels))
            if 'preprocessing' in self.metadata and self.metadata['preprocessing']!=identity:raise RuntimeError('Reconstructed native preprocessing/view state differs.')
            self.metadata['preprocessing']=identity;self.metadata['compatibility']=classifier.compatibility
            self._engine=classifier
        return self._engine

    def predict_proba(self,x):
        check_deadline();classifier=self.engine()
        if fitted_identity(classifier,len(self.labels))!=self.metadata['preprocessing']:raise RuntimeError('Context/views mutated.')
        p=valid_probabilities(classifier.predict_proba(self.encoder.transform(x)),len(x))
        self.metadata['last_prediction']=dict(query_rows=len(x),forward_records=list(getattr(classifier,'forward_records',[])),
            probabilities_sha256=array_hash(p),full_support_rows=len(self.labels),views=8)
        return p

    def release(self):
        self._engine=None;gc.collect()
        if self.device=='mps' and torch.backends.mps.is_available():torch.mps.empty_cache()

    def __getstate__(self):return {**self.__dict__,'device':'cpu','_engine':None}


def fit_context(x,y,ids,seed=0,device='cpu'):
    predictor=EXAONEPredictor(x,y,ids,seed,device)
    try:predictor.engine();return predictor
    except BaseException:predictor.release();raise
