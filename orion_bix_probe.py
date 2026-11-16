"""CPU-only official Orion-BiX v1.1 checkpoint diagnosis; not a candidate/resource screen.

Reproduce with PYTHONHASHSEED=0 .venv-orion/bin/python orion_bix_probe.py --output <new-directory>.
No MPS option, real dataset reads, training, probability accuracy, or compatibility patches.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'third_party/orion_bix'
CHECKPOINT=ROOT/'artifacts/pretrained/orion_bix/Orion-BiX-v1.1.ckpt'
WEIGHT_SHA='c53afd272e05e9bd6740698b6fa6f3d8789fc4f6da263415d3287b9ed51cd979'
MANIFEST_SHA='49d09c7857ad1536052204f0e3db667c2c6f9ada4b4df662f86b44a453006b65'
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CATEGORICAL=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def frame(rows,seed):
    rng=np.random.default_rng(seed)
    x=pd.DataFrame({name:rng.lognormal(2.,1.,rows) for name in COMPACT})
    for col in CATEGORICAL:x[col]=np.asarray(['A','B','C'],dtype=object)[rng.integers(3,size=rows)]
    x.loc[::13,'Age']=np.nan;x.loc[::17,'HomePlanet']=np.nan
    return x[COMPACT]


def plain_metadata(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,(list,tuple)) and all(isinstance(v,(int,float,str,bool)) for v in value):return list(value)
    return type(value).__name__


def run(output):
    output=Path(output)
    if output.exists():raise RuntimeError('Diagnostic output exists; no overwrite.')
    output.mkdir(parents=True)
    result=dict(status='incomplete',device='cpu',threads=4,budget_seconds=600,
        no_real_targets_read=True,no_accuracy_computed=True,no_training=True,MPS_executed=False,
        compatibility_executed=False,formal_screen_qualified=False,
        interpretation='Checkpoint identity/actual execution facts only; no claim this model is ineffective.')
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('CPU checkpoint diagnosis600s budget exceeded.')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    classifier=checkpoint=None
    try:
        if sha(VENDOR/'SOURCE.json')!=MANIFEST_SHA:raise RuntimeError('Pinned vendor manifest differs.')
        manifest=json.loads((VENDOR/'SOURCE.json').read_text())
        for name,digest in manifest['vendor_files_sha256'].items():
            if sha(VENDOR/name)!=digest:raise RuntimeError('Official vendor source changed: '+name)
        if sha(CHECKPOINT)!=WEIGHT_SHA:raise RuntimeError('Pinned checkpoint missing or hash differs.')
        for line in (ROOT/'requirements-orion-bix.lock.txt').read_text().splitlines():
            name,version=line.split('==')
            if importlib.metadata.version(name)!=version:raise RuntimeError('Independent dependency lock differs: '+name)
        sys.path.insert(0,str(VENDOR/'src'))
        from orion_bix.sklearn.classifier import OrionBixClassifier
        import orion_bix
        if not Path(orion_bix.__file__).resolve().is_relative_to((VENDOR/'src').resolve()):raise RuntimeError('Wrong Orion import.')
        torch.set_num_threads(4)
        checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        x,q=frame(64,42),frame(17,43);q.loc[0,'HomePlanet']='unseen'
        y=np.arange(64,dtype=np.int64)%2
        classifier=OrionBixClassifier(n_estimators=32,batch_size=1,use_amp=False,device='cpu',
            model_path=str(CHECKPOINT),allow_auto_download=False,checkpoint_version='Orion-BiX-v1.1.ckpt',random_state=42,n_jobs=4)
        classifier.fit(x,y)
        cleaned={k.replace('_orig_mod.','').replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
        actual=classifier.model_.state_dict()
        if set(cleaned)!=set(actual) or any(not torch.equal(cleaned[k],actual[k]) for k in actual):raise RuntimeError('Native strict state load mismatch.')
        views=sum(len(v) for v in classifier.ensemble_generator_.ensemble_configs_.values())
        if views!=32 or len(classifier.ensemble_generator_.X_)!=64:raise RuntimeError('Native context/views differ.')
        if not np.array_equal(classifier.classes_,[0,1]):raise RuntimeError('Binary class order differs.')
        tick=time.monotonic();p=classifier.predict_proba(q);forward=time.monotonic()-tick
        if p.shape!=(17,2) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1.,atol=1e-6,rtol=0):raise RuntimeError('Native CPU probability failure.')
        optimizer=checkpoint.get('optimizer',checkpoint.get('optimizer_state',checkpoint.get('optimizer_state_dict',{})))
        scheduler=checkpoint.get('scheduler',checkpoint.get('scheduler_state',checkpoint.get('scheduler_state_dict',{})))
        metadata={key:plain_metadata(value) for key,value in checkpoint.items() if key not in ['state_dict','config','optimizer','optimizer_state','optimizer_state_dict','scheduler','scheduler_state','scheduler_state_dict']}
        optimizer_steps=[float(state['step'].item() if isinstance(state['step'],torch.Tensor) else state['step'])
            for state in optimizer.get('state',{}).values() if 'step' in state]
        shape_manifest={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in actual.items()}
        (output/'state_shapes.json').write_text(json.dumps(shape_manifest,indent=2)+'\n')
        env=''.join(f'{d.metadata["Name"]}=={d.version}\n' for d in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower()))
        (output/'environment.lock.txt').write_text(env)
        result.update(status='native_cpu_complete_no_candidate_score',checkpoint_sha256=WEIGHT_SHA,checkpoint_bytes=CHECKPOINT.stat().st_size,
            checkpoint_config=checkpoint['config'],native_strict_state_all_keys_shapes_values_equal=True,
            actual_row_attention_type=classifier.model_.row_attention_type,
            instantiated_biaxial_module=isinstance(getattr(classifier.model_,'bi_axial_attention',None),torch.nn.Module),
            top_modules={name:type(module).__module__+'.'+type(module).__name__ for name,module in classifier.model_.named_children()},
            manager_modules={name:module.inference_mgr.enc_name for name,module in classifier.model_.named_modules() if hasattr(module,'inference_mgr')},
            parameters=sum(v.numel() for v in classifier.model_.parameters()),buffers_after_forward=sum(v.numel() for v in classifier.model_.buffers()),
            state_elements=sum(v.numel() for v in cleaned.values()),state_keys=len(actual),state_shapes_sha256=sha(output/'state_shapes.json'),
            checkpoint_extra_metadata=metadata,optimizer_present=bool(optimizer),scheduler_present=bool(scheduler),
            optimizer_groups=[{k:plain_metadata(v) for k,v in group.items() if k!='params'} for group in optimizer.get('param_groups',[])],
            optimizer_step_summary=dict(entries=len(optimizer_steps),unique_steps=sorted(set(optimizer_steps))),
            scheduler_metadata={k:plain_metadata(v) for k,v in scheduler.items()},
            saved_step_interpretation='These are the saved segment optimizer/scheduler steps; they do not establish total pretraining duration or the origin of initial weights.',
            native_cpu=dict(support_rows=64,query_rows=17,views=views,batch_size=1,forward_seconds=forward,
                probability_sha256=hashlib.sha256(p.tobytes()).hexdigest()),
            sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),ROOT/'requirements-orion-bix.txt',ROOT/'requirements-orion-bix.lock.txt',VENDOR/'SOURCE.json']},
            environment_sha256=hashlib.sha256(env.encode()).hexdigest())
        return result
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        result['seconds']=time.monotonic()-started
        (output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        classifier=checkpoint=None;gc.collect()
        print(json.dumps({k:result.get(k) for k in ['status','actual_row_attention_type','instantiated_biaxial_module','parameters','seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
