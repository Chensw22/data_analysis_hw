"""Pinned Kumo Medium, unchanged official recipe; synthetic resource checks only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in [BASE, *BASE.parents] if (p/'iterate.py').is_file())
VENDOR = BASE/'third_party/kumo'
sys.path.insert(0, str(VENDOR))
import sdm
from sdm.models import KumoTabular, Callback
import kumo_mps_compat

sys.modules.setdefault('kumo_resource_probe', sys.modules[__name__])
CODE_COMMIT = '909d1d49a68e4d6bf71a468f9f813869f62ba970'
MODEL_REVISION = '3c3e10bbdb590ace29e7026847f92db3c603096d'
WEIGHT_SHA = '70d6aee8fd13416300afead71a5145a40edb790952e2417d87f27b3f7e57793d'
CHECKPOINT = ROOT/'artifacts/pretrained/kumo/medium-classifier.pt'
COMPACT = ['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService',
           'FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CATEGORICAL = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
STYPES = {name: 'categorical' if name in CATEGORICAL else 'numerical' for name in COMPACT}
SETTINGS = dict(size='medium',task='classification',parameters=61485274,
    n_estimators=8,estimator_batch_size=1,model_dtype='float32',amp=False,
    recipe='official_default_unchanged',preprocessing_device='cpu',
    model_device='mps',output_device='cpu',aggregation='official_average_logits_then_softmax',
    context='all_training_rows_no_sampling',kv_cache=False,threads=4,
    mps_memory_fraction=.65,threshold=.5,seed=20261116,
    cpu_budget_seconds=900,mps_budget_seconds=7200,
    mps_compatibility=kumo_mps_compat.COMPAT_ID)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def environment():
    return '\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n'


def verify_checkpoint():
    if CHECKPOINT.stat().st_size!=246117940 or sha(CHECKPOINT)!=WEIGHT_SHA:
        raise RuntimeError('Pinned Kumo Medium checkpoint missing or changed; no automatic download.')


def source_hashes():
    paths=[BASE/'kumo_resource_probe.py',BASE/'test_kumo_resource_probe.py',
           BASE/'kumo_mps_compat.py',BASE/'test_kumo_mps_compat.py',
           BASE/'requirements-kumo.lock.txt',BASE/'reports/kumo_strategy_review.json']
    paths += [p for p in VENDOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    return {str(p.relative_to(BASE)):sha(p) for p in sorted(paths)}


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if value.device.type!='cpu':raise ValueError('Serialized pretrained state must be CPU.')
        h.update(name.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def new_model(state,device):
    model=KumoTabular(task='classification',size='medium',pretrained=False,device='meta')
    model.models[sdm.Task.classification].load_state_dict(state,assign=True,strict=True)
    model.to(device).eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters())!=SETTINGS['parameters']:
        raise RuntimeError('Unexpected Kumo parameter count.')
    if any(p.dtype!=torch.float32 or p.device.type!=device for p in model.parameters()):
        raise RuntimeError('Expected all pretrained parameters in FP32 on requested device.')
    if kumo_mps_compat.install(model)!=1:
        raise RuntimeError('Expected exactly one Kumo compatibility binding.')
    return model


def features(frame):
    result=frame.loc[:,COMPACT].copy()
    for name in CATEGORICAL:
        result[name]=result[name].replace('__MISSING__',None)
    return sdm.TableTensor.from_pandas(result,stypes=STYPES,device='cpu')


def labels(y):
    y=np.asarray(y)
    if y.ndim!=1 or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('Require both binary classes in training context.')
    return sdm.TableTensor.from_pandas(pd.DataFrame({'target':y.astype(np.int64)}),
        stypes={'target':'categorical'},device='cpu')


class CpuRecipeDeviceBridge(Callback):
    """Only transfer finished official recipe tensors; do not change the recipe."""
    def __init__(self,device,support_rows):
        self.device=device;self.support_rows=int(support_rows);self.observed=[]

    def on_context_preprocessing_end(self,model,x,y,x_context_only,related_tables):
        if x_context_only is not None or related_tables is not None:
            raise ValueError('This candidate only supports one supervised table.')
        if (x.device.type!='cpu' or y.device.type!='cpu' or x.numerical.dtype!=torch.float32
                or len(x.categorical_columns)):
            raise RuntimeError('Official preprocessing must finish as CPU FP32 before transfer.')
        if x.shape[-2]!=self.support_rows or y.shape[-2]!=self.support_rows:
            raise RuntimeError('Training context was reduced or misaligned.')
        self.observed.append(dict(rows=int(x.shape[-2]),columns=int(x.shape[-1]),
            input_device=x.device.type,input_dtype=str(x.numerical.dtype),output_device=self.device))
        return x.to(self.device),y.to(self.device),None,None

    def on_query_preprocessing_end(self,model,x,related_tables):
        if (related_tables is not None or x.device.type!='cpu' or x.numerical.dtype!=torch.float32
                or len(x.categorical_columns)):
            raise RuntimeError('Query must use the context-fitted CPU recipe.')
        return x.to(self.device),None

    def on_model_forward_end(self,model,out):
        if out.device.type!=self.device:
            raise RuntimeError('Unexpected neural backend; automatic fallback is forbidden.')
        return out.to('cpu')


def probabilities(model,x,y,query,device,estimators=8,seed=20261116,bridge=True):
    if x.device.type!='cpu' or query.device.type!='cpu' or y.device.type!='cpu':
        raise ValueError('Keep raw tables on CPU to preserve official float64 preprocessing.')
    callback=CpuRecipeDeviceBridge(device,len(y))
    with torch.inference_mode(),torch.autocast(device_type=device,enabled=False):
        output=model(x_context=x,y_context=y,x_query=query,num_estimators=estimators,
            estimator_batch_size=1,callbacks=[callback] if bridge else [],
            generator=torch.Generator(device='cpu').manual_seed(seed))
    names=list(output.numerical_columns)
    if set(names)!={'0','1'}:raise RuntimeError(f'Unexpected output class identity: {names}')
    values=output.numerical.detach().cpu().numpy()
    if values.shape!=(len(query),2):raise RuntimeError(f'Unexpected output shape: {values.shape}')
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1) or not np.allclose(values.sum(-1),1,atol=2e-6):
        raise FloatingPointError('Invalid probability vector.')
    if bridge and len(callback.observed)!=estimators:
        raise RuntimeError('Not every declared estimator forwarded the full context.')
    return values[:,names.index('1')].copy(),callback.observed


class KumoPredictor:
    """Portable CPU pretrained state plus exact training context; no remote loader."""
    def __init__(self,x,y,device='cpu',estimators=8,seed=20261116):
        if len(x)!=len(y):raise ValueError('Training rows and labels differ.')
        verify_checkpoint()
        self.state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        self.state_sha=state_hash(self.state);self.checkpoint_sha=WEIGHT_SHA
        self.x=features(x);self.y=labels(y);self.device=device
        self.estimators=int(estimators);self.seed=int(seed);self._engine=None
        self.last_forward_contexts=[]

    def engine(self):
        if self._engine is None:
            if self.checkpoint_sha!=WEIGHT_SHA or state_hash(self.state)!=self.state_sha:
                raise RuntimeError('Persisted pretrained state changed.')
            self._engine=new_model(self.state,self.device)
        return self._engine

    def predict_proba(self,x):
        p,self.last_forward_contexts=probabilities(self.engine(),self.x,self.y,features(x),
            self.device,self.estimators,self.seed)
        return np.column_stack([1-p,p])

    def to(self,device):
        self._engine=None;self.device=str(device);return self

    def __getstate__(self):
        if any(t.device.type!='cpu' for t in self.state.values()):
            raise RuntimeError('GPU tensors cannot enter portable artifact.')
        result=self.__dict__.copy();result['_engine']=None;result['device']='cpu'
        return result

    def __setstate__(self,state):
        self.__dict__.update(state);self._engine=None;self.device='cpu'


for _cls in [CpuRecipeDeviceBridge,KumoPredictor]:_cls.__module__='kumo_resource_probe'


def synthetic(support_rows,query_rows,seed):
    rng=np.random.default_rng(seed);n=support_rows+query_rows
    frame=pd.DataFrame({
        'HomePlanet':rng.choice(['Earth','Europa','Mars'],n),
        'CryoSleep':rng.choice(['True','False'],n),'Destination':rng.choice(['A','B','C'],n),
        'Age':rng.integers(0,90,n).astype(float),'VIP':rng.choice(['False','True'],n),
        **{key:np.where(rng.random(n)<.65,0.,rng.lognormal(4.,1.3,n))
           for key in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']},
        'Deck':rng.choice(['B','C','F','G'],n),'CabinNumber':rng.integers(0,2000,n).astype(float),
        'Side':rng.choice(['P','S'],n)})
    for column in COMPACT:frame.loc[rng.random(n)<.03,column]=None if column in CATEGORICAL else np.nan
    # Unseen query category verifies context-fitted category handling.
    frame.loc[support_rows,'HomePlanet']='unseen_synthetic_planet'
    return frame.iloc[:support_rows].copy(),np.arange(support_rows,dtype=np.int64)%2,frame.iloc[support_rows:].copy()


def snapshot(device):
    result=dict(max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                measurement='RSS peak and MPS snapshots, not continuous accelerator peak')
    if device=='mps':
        torch.mps.synchronize()
        result.update(mps_current_bytes=torch.mps.current_allocated_memory(),
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
            mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return result


def cleanup(device):
    gc.collect()
    if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()


def compare(a,b,atol=2e-5):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Nonfinite or misaligned comparison.')
    difference=float(np.max(np.abs(a-b)))
    disagreements=int(np.sum((a>=.5)!=(b>=.5)))
    if difference>atol or disagreements:raise AssertionError(f'Probability/decision mismatch: {difference}, {disagreements}')
    return dict(max_abs_difference=difference,threshold_disagreements=disagreements,tolerance=atol)


def run(device,out):
    out=Path(out)
    if out.exists():raise RuntimeError('Probe output exists; no overwrite/resume.')
    if device not in ('cpu','mps'):raise ValueError('Explicit CPU or MPS required.')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('Disable automatic MPS fallback before the probe.')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback.')
    verify_checkpoint();sources=source_hashes()
    if device=='mps':
        if BASE!=ROOT:raise RuntimeError('Stage files must be reviewed, relocated and committed before MPS.')
        tracked=subprocess.check_output(['git','ls-files','-z','--',*sources],cwd=ROOT).decode()
        if set(filter(None,tracked.split('\0')))!=set(sources):raise RuntimeError('All probe sources must be committed.')
        if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=ROOT).strip():
            raise RuntimeError('Probe source files differ from committed versions.')
    torch.set_num_threads(4)
    if device=='mps':torch.mps.set_per_process_memory_fraction(.65)
    lock=environment()
    if lock!=(BASE/'requirements-kumo.lock.txt').read_text():raise RuntimeError('Environment differs from frozen lock.')
    cases=[(128,17,8)] if device=='cpu' else [(128,17,8),(4636,2318,1),(8693,4277,1)]
    budget=SETTINGS[f'{device}_budget_seconds'];out.mkdir(parents=True)
    (out/'environment.lock.txt').write_text(lock)
    write(out/'freeze.json',dict(synthetic_only=True,accuracy_computed=False,
        settings=SETTINGS,device=device,code_commit=CODE_COMMIT,model_revision=MODEL_REVISION,
        checkpoint_sha256=WEIGHT_SHA,sources=sources,environment_sha256=sha(out/'environment.lock.txt'),
        cases=[dict(support=s,query=q,views=e) for s,q,e in cases],budget_seconds=budget,
        comparison_atol=2e-5,threshold_decisions_must_match=True,
        large_probe_limit='Large cases test one view, not complete eight-view runtime or accuracy.',
        recorded_at_utc=datetime.now(timezone.utc).isoformat()))
    record=dict(status='incomplete',synthetic_only=True,accuracy_computed=False,device=device,
        freeze_sha256=sha(out/'freeze.json'),cases=[],before=snapshot(device))
    write(out/'result.json',record)
    def timeout(*_):raise TimeoutError('Fixed resource probe time budget exceeded.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,budget)
    started=time.monotonic();predictor=restored=None
    try:
        for s,q,e in cases:
            tick=time.monotonic();x,y,z=synthetic(s,q,SETTINGS['seed'])
            record['active_case']=dict(support=s,query=q,views=e,stage='load');write(out/'result.json',record)
            predictor=KumoPredictor(x,y,device,e,SETTINGS['seed'])
            record['active_case']['stage']='full_forward';write(out/'result.json',record)
            first=predictor.predict_proba(z)[:,1]
            contexts=predictor.last_forward_contexts
            after_forward=snapshot(device)
            record['active_case']['stage']='partition_check';write(out/'result.json',record)
            # Full context on both partitions; no query-dependent preprocessing fits.
            midpoint=q//2
            partitioned=np.concatenate([predictor.predict_proba(z.iloc[:midpoint])[:,1],
                                        predictor.predict_proba(z.iloc[midpoint:])[:,1]])
            partition_check=compare(first,partitioned)
            raw_check=None
            if device=='cpu':
                raw,_=probabilities(predictor.engine(),predictor.x,predictor.y,features(z),'cpu',e,
                    SETTINGS['seed'],bridge=False)
                raw_check=compare(first,raw,atol=0.)
            record['active_case']['stage']='cpu_save_reload';write(out/'result.json',record)
            path=out/'synthetic_predictor.joblib';joblib.dump(predictor,path)
            predictor=None;cleanup(device)
            restored=joblib.load(path)
            if restored.device!='cpu' or restored._engine is not None or len(restored.y)!=s:
                raise RuntimeError('Artifact did not restore CPU state and complete training context.')
            restored.to(device)
            record['active_case']['stage']='full_reload_forward';write(out/'result.json',record)
            second=restored.predict_proba(z)[:,1];reload_check=compare(first,second)
            restored=None;cleanup(device);path.unlink()
            record['cases'].append(dict(support_rows=s,query_rows=q,views=e,
                forwarded_contexts=contexts,partition_check=partition_check,
                raw_official_cpu_vs_callback=raw_check,reload_check=reload_check,
                probability_rows=len(first),seconds=time.monotonic()-tick,
                after_forward=after_forward,after_release=snapshot(device)))
            write(out/'result.json',record)
        if source_hashes()!=sources:raise RuntimeError('Probe source changed while running.')
        record.pop('active_case',None)
        record.update(status='complete',seconds=time.monotonic()-started,after=snapshot(device))
        write(out/'result.json',record)
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),
            seconds=time.monotonic()-started)
        write(out/'result.json',record);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        predictor=restored=None;cleanup(device)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','mps'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.device,args.out)
