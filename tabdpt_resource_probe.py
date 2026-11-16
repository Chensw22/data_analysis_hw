"""Pinned TabDPT-v1.3 setup and synthetic-only resource probe; never score data."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
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

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT/'third_party/tabdpt'
sys.path.insert(0, str(VENDOR))
from tabdpt import TabDPTClassifier

CHECKPOINT = ROOT/'artifacts/pretrained/tabdpt-v1.3/tabdpt1_3.safetensors'
WEIGHT_SHA = '97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'
CODE_COMMIT = '3cef828a0bbfe8818bcdd6d855940e1949f2b4cd'
MODEL_REVISION = 'a5ca6e01c0fa09ec68c73e958e5199d1932abb3a'
COMPACT = ['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService',
           'FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CATEGORICAL = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
PREDICT = dict(n_ensembles=8, temperature=1., context_size=None, batch_size=4277,
               permute_classes=True)
SETTINGS = dict(device='mps', threads=4, mps_memory_fraction=.65, dtype='float32',
    use_flash=False, compile=False, normalizer='standard', clip_sigma=8.,
    feature_reduction='pca', context_reduction='subsample', missing_indicators=False,
    predict=PREDICT, parameters=63040528, state_elements=63040624, buffers=96, threshold=.5)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def environment():
    return '\n'.join(sorted(f"{d.metadata['Name']}=={d.version}"
                           for d in importlib.metadata.distributions()))+'\n'


class CompactEncoder:
    """Train-only category maps; unknown=-1, original missing remains NaN."""
    def fit(self, x):
        self.maps = {}
        for column in CATEGORICAL:
            values = x[column].replace('__MISSING__', np.nan).dropna().astype(str)
            self.maps[column] = {value:index for index,value in enumerate(sorted(values.unique()))}
        self.columns = list(COMPACT)
        return self

    def transform(self, x):
        output = np.empty((len(x),len(self.columns)), dtype=np.float32)
        for index, column in enumerate(self.columns):
            values = x[column].replace('__MISSING__', np.nan)
            if column in self.maps:
                missing = values.isna().to_numpy()
                values = values.astype(str).map(self.maps[column]).fillna(-1).to_numpy(dtype=np.float32)
                values[missing] = np.nan
            else:
                values = pd.to_numeric(values, errors='raise').to_numpy(dtype=np.float32, na_value=np.nan)
            output[:,index] = values
        return np.ascontiguousarray(output)


def verify_checkpoint():
    if CHECKPOINT.stat().st_size != 252233296 or sha(CHECKPOINT) != WEIGHT_SHA:
        raise RuntimeError('Pinned complete checkpoint missing or changed.')


def new_model(device):
    verify_checkpoint()
    result = TabDPTClassifier(device=device, use_flash=False, compile=False,
        model_weight_path=str(CHECKPOINT), verbose=False)
    if (sum(p.numel() for p in result.model.parameters()) != SETTINGS['parameters']
            or sum(p.numel() for p in result.model.buffers()) != SETTINGS['buffers']
            or any(p.dtype != torch.float32 for p in result.model.parameters())):
        raise RuntimeError('Expected complete FP32 v1.3 model.')
    return result


def probabilities(model, x, ensembles=8, seed=20261116):
    result = np.asarray(model.ensemble_predict_proba(x, **{**PREDICT, 'n_ensembles':ensembles}, seed=seed))
    if (result.shape != (len(x),2) or not np.isfinite(result).all()
            or (result<0).any() or (result>1).any()
            or not np.allclose(result.sum(axis=1),1,atol=2e-6,rtol=0)):
        raise RuntimeError('Invalid/incomplete probability matrix.')
    return result


def compare(first, second, atol=2e-6):
    if first.shape != second.shape:
        raise RuntimeError('Prediction shape differs.')
    difference = float(np.max(np.abs(first-second)))
    if (not np.isfinite(second).all() or difference > atol
            or not np.array_equal(first[:,1]>=.5,second[:,1]>=.5)):
        raise RuntimeError('Prediction probabilities/threshold decisions differ.')
    return difference


def release(model=None, device='mps'):
    if model is not None:
        model.to('cpu')
    gc.collect()
    if device == 'mps':
        torch.mps.synchronize(); torch.mps.empty_cache()


def snapshot(device):
    record = dict(max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  platform=sys.platform, memory_measurement='RSS high-water; MPS point snapshots, not peak')
    if device == 'mps':
        torch.mps.synchronize()
        record.update(mps_current_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return record


def synthetic(support, query, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(support+query,13)).astype(np.float32)
    x[rng.random(x.shape)<.025] = np.nan
    y = np.arange(support,dtype=np.int64)%2
    return x[:support], y, x[support:]


def source_hashes():
    paths = [ROOT/name for name in ['tabdpt_resource_probe.py','test_tabdpt_resource_probe.py',
        'reports/tabdpt_strategy_review.json','requirements-tabdpt.lock.txt']]
    paths += [p for p in VENDOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def run(device, out):
    out = Path(out)
    if out.exists():
        raise RuntimeError('Probe destination exists; no overwrite or implicit resume.')
    verify_checkpoint()
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no CPU fallback.')
    sources = source_hashes()
    if device == 'mps':
        tracked = subprocess.check_output(['git','ls-files','-z','--',*sources],cwd=ROOT).decode()
        if set(filter(None,tracked.split('\0'))) != set(sources):
            raise RuntimeError('Commit all frozen probe sources before MPS execution.')
        if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=ROOT).strip():
            raise RuntimeError('Probe sources differ from committed files.')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    if device == 'mps':torch.mps.set_per_process_memory_fraction(.65)
    budget = 6000 if device == 'mps' else 600
    lock = environment(); (out/'environment.lock.txt').write_text(lock)
    cases = [(128,17,8)] if device == 'cpu' else [(128,17,8),(4636,2318,1),(8693,4277,1)]
    frozen = dict(synthetic_only=True,accuracy_computed=False,settings=SETTINGS,device=device,
        sources=sources,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        checkpoint_sha256=WEIGHT_SHA,code_commit=CODE_COMMIT,model_revision=MODEL_REVISION,
        cases=[dict(support=s,query=q,ensembles=e) for s,q,e in cases],budget_seconds=budget,
        reload_tolerance=2e-6,query_partition_tolerance=2e-5,threshold_decisions_must_match=True,
        interpretation='Large cases check memory for one permutation; not full8ensemble runtime or accuracy.',
        recorded_at_utc=datetime.now(timezone.utc).isoformat())
    write(out/'freeze.json',frozen)
    record = dict(status='incomplete',synthetic_only=True,accuracy_computed=False,device=device,
                  freeze_sha256=sha(out/'freeze.json'),cases=[],before=snapshot(device))
    write(out/'result.json',record)
    def timeout(*_):raise TimeoutError('Fixed synthetic resource budget exceeded.')
    previous = signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,budget)
    start=time.monotonic(); model=restored=None
    try:
        for support, query, ensembles in cases:
            tick=time.monotonic(); x,y,z=synthetic(support,query,20261116)
            record['active_case']=dict(support_rows=support,query_rows=query,ensembles=ensembles,stage='load_and_fit')
            write(out/'result.json',record)
            model=new_model(device); model.fit(x,y)
            # Observe actual forwarded support length; no hidden reduction allowed.
            seen=[]
            def capture(_module,args,kwargs):seen.append(int(kwargs['y_src'].shape[1]))
            handle=model.model.register_forward_pre_hook(capture,with_kwargs=True)
            record['active_case']['stage']='full_prediction';write(out/'result.json',record)
            first=probabilities(model,z,ensembles=ensembles)
            handle.remove()
            if not seen or set(seen)!={support}:
                raise RuntimeError('Actual model context was reduced.')
            before_reload=snapshot(device)
            # External query partition must not alter full-context predictions.
            record['active_case']['stage']='query_partition_check';write(out/'result.json',record)
            short=probabilities(model,z[:256 if query>=256 else 7],ensembles=ensembles)
            batching_difference=compare(first[:len(short)],short,atol=2e-5)
            model.to('cpu')
            if any(p.device.type!='cpu' for p in model.model.parameters()):
                raise RuntimeError('Portable artifact must hold CPU weights.')
            path=out/'synthetic_model.joblib'; joblib.dump(model,path)
            release(model,device); model=None; release(None,device)
            restored=joblib.load(path); restored.to(device)
            if restored.n_instances!=support:raise RuntimeError('Reloaded training context differs.')
            record['active_case']['stage']='full_reload_prediction';write(out/'result.json',record)
            second=probabilities(restored,z,ensembles=ensembles)
            difference=compare(first,second)
            release(restored,device); restored=None; release(None,device)
            path.unlink()  # Synthetic weights have no subsequent modeling use.
            row=dict(support_rows=support,query_rows=query,ensembles=ensembles,
                actual_forward_support_rows=sorted(set(seen)),forward_calls=len(seen),
                probability_rows=len(first),full_reload_rows=len(second),
                reload_max_abs_difference=difference,batching_max_abs_difference=batching_difference,
                reload_threshold_identical=True,seconds=time.monotonic()-tick,
                resources_after_prediction=before_reload,resources_after_release=snapshot(device))
            record['cases'].append(row); write(out/'result.json',record)
            print(json.dumps(row),flush=True)
        if source_hashes()!=sources or environment()!=lock:
            raise RuntimeError('Frozen source/environment changed during probe.')
        verify_checkpoint()
        record.pop('active_case',None)
        record.update(status='complete',seconds=time.monotonic()-start,after=snapshot(device))
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),
                      seconds=time.monotonic()-start)
        raise
    finally:
        write(out/'result.json',record)
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        release(model,device); release(restored,device)
        (out/'synthetic_model.joblib').unlink(missing_ok=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','mps'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.device,args.out)
