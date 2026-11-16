"""Fixed Kumo Medium eight-view screen; original development rows, no upload CLI."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in [BASE,*BASE.parents] if (p/'iterate.py').is_file())
if str(ROOT) not in sys.path:sys.path.append(str(ROOT))
import iterate
import kumo_adapter as adapter
import kumo_full8_probe as full8
backend=adapter.backend
NAME=adapter.NAME
BASE_FIT,BASE_EVALUATE=iterate.fit,iterate.evaluate
PROBE=ROOT/'reports/kumo_resources/full8_mps'
SETTINGS=dict(**backend.SETTINGS,screen_budget_seconds=7200,export_budget_seconds=7200,
    weekly_reserve_percent=30,seed_policy='passed outer CV seed plus fold',
    inference_seed='fold seed; no fixed synthetic-probe seed in real fits')
OWN_SOURCES=['kumo_candidate.py','test_kumo_candidate.py']
RUN_OUT=None
DEADLINE=None
environment_lock=backend.environment


def release_memory():backend.cleanup('mps')


def check_deadline():
    if DEADLINE is not None and time.monotonic()>DEADLINE:
        raise TimeoutError('Fixed Kumo stage budget exceeded.')


def verify_resource():
    full8.verify_base_resource()
    frozen=json.loads((PROBE/'freeze.json').read_text())
    result=json.loads((PROBE/'result.json').read_text())
    env_sha=hashlib.sha256(environment_lock().encode()).hexdigest()
    if (result.get('status')!='complete' or result.get('device')!='mps'
            or result.get('synthetic_only') is not True or result.get('accuracy_computed') is not False
            or result.get('freeze_sha256')!=backend.sha(PROBE/'freeze.json')
            or frozen.get('device')!='mps' or frozen.get('settings')!=full8.SETTINGS
            or frozen.get('backend_settings')!=backend.SETTINGS
            or frozen.get('checkpoint_sha256')!=backend.WEIGHT_SHA
            or frozen.get('sources')!=full8.source_hashes()
            or frozen.get('environment_sha256')!=env_sha
            or backend.sha(PROBE/'environment.lock.txt')!=env_sha
            or frozen.get('base_probe_freeze_sha256')!=backend.sha(full8.PROBE/'freeze.json')
            or frozen.get('base_probe_result_sha256')!=backend.sha(full8.PROBE/'result.json')
            or [result.get(k) for k in ('support_rows','query_rows','views')]!=[8693,4277,8]
            or result.get('first_probability_rows')!=4277 or result.get('reload_probability_rows')!=4277
            or not full8.compare_record(result.get('reload_check',{}))):
        raise RuntimeError('Complete matching full-eight-view/reload resource evidence required.')
    for key in ('forwarded_contexts','reload_forwarded_contexts'):
        contexts=result.get(key,[])
        if len(contexts)!=8 or any(c.get('rows')!=8693 or c.get('input_device')!='cpu'
                or c.get('input_dtype')!='torch.float32' or c.get('output_device')!='mps' for c in contexts):
            raise RuntimeError('Every full8/reloaded estimator must preserve complete context.')
    return frozen


def source_hashes():
    sources=full8.source_hashes()
    for name in OWN_SOURCES:sources[name]=backend.sha(BASE/name)
    for name in ('iterate.py','experiment.py','requirements.lock.txt'):
        sources[name]=backend.sha(ROOT/name)
    for name in ('freeze.json','result.json','environment.lock.txt'):
        path=PROBE/name;sources[str(path.relative_to(ROOT))]=backend.sha(path)
    return sources


def configure_runtime(out,stage='screen'):
    global RUN_OUT,DEADLINE
    if stage not in ('screen','export'):raise ValueError(stage)
    verify_resource();full8.require_committed(source_hashes())
    if not backend.torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback.')
    if full8.os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('Disable automatic MPS fallback.')
    RUN_OUT=Path(out);DEADLINE=time.monotonic()+SETTINGS[stage+'_budget_seconds']
    backend.torch.set_num_threads(4);backend.torch.mps.set_per_process_memory_fraction(.65)


def fit(name,matrices,train_idx,y,predict_idx,seed):
    check_deadline()
    if name==iterate.BASELINE:return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name!=NAME or RUN_OUT is None:raise RuntimeError('Unknown candidate or unconfigured runtime.')
    a,b=np.asarray(train_idx),np.asarray(predict_idx)
    x=matrices[False][0]
    if (a.ndim!=1 or b.ndim!=1 or not len(a) or not len(b)
            or not np.issubdtype(a.dtype,np.integer) or not np.issubdtype(b.dtype,np.integer)
            or len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b)
            or np.intersect1d(a,b).size or min(a.min(),b.min())<0 or max(a.max(),b.max())>=len(x)):
        raise ValueError('Unique disjoint in-range training/query row IDs required.')
    a=a.astype(np.int64);b=b.astype(np.int64)
    path=RUN_OUT/f'{NAME}_seed{seed}_{adapter.array_sha(a)[:12]}.json'
    if path.exists():raise RuntimeError('Fit log exists; no overwrite/resume.')
    started=time.monotonic();model=None
    backend.write(path,dict(status='fitting_context',train_rows=len(a),query_rows=len(b),seed=int(seed)))
    try:
        model=adapter.fit_context(x.iloc[a],y,a,seed,device='mps')
        probability=model.predict_proba(x.iloc[b])[:,1]
        check_deadline()
        model.metadata.update(status='complete',seconds=time.monotonic()-started,
            prediction_rows=len(b),prediction_ids_sha256=adapter.array_sha(b))
        backend.write(path,model.metadata)
        return probability,model
    except BaseException as error:
        backend.write(path,dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if model is not None:model.release()


def record_protocol(out):
    path=Path(out)/'protocol.json';record=json.loads(path.read_text())
    if 'kumo_policy' in record:assert_frozen(out);return
    verify_resource();sources=source_hashes();commit=full8.require_committed(sources)
    for relative,digest in sources.items():
        record['sources'][relative]=digest
        target=Path(out)/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    record['kumo_policy']=dict(settings=copy.deepcopy(SETTINGS),source_commit=commit,
        automatic_eligibility=False,source_environment_sha256=hashlib.sha256(environment_lock().encode()).hexdigest(),
        comparison='Original CatBoost v6 rerun; old Mitra/v44 remain descriptive historical references.',
        disclosure='Original6954development rows reused; no oldholdout scoring. Pretraining provenance is author-reported synthetic, not an independent contamination audit.',
        selection='Fixed eight official views, threshold0.5; no tuning of recipe/weights/views/threshold.')
    backend.write(path,record)


def evaluate(name,matrices,train,dev,seed,out):
    if len(dev)!=6954 or seed!=iterate.SEEDS[0]:
        raise ValueError('Original6954development rows and first screening seed required.')
    record_protocol(out)
    return BASE_EVALUATE(name,matrices,train,dev,seed,out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:return
    if iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE or BASE_FIT.__module__!='iterate':
        raise RuntimeError('Use a fresh process without another candidate wrapper.')
    iterate.CONFIGS={iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
        NAME:{**SETTINGS,'context':False,'model':'kumo_medium8',
              'support_context':SETTINGS['context']}}
    iterate.BLEND=[];iterate.fit,iterate.evaluate=fit,evaluate


def assert_frozen(out):
    configure();iterate.assert_frozen(Path(out));verify_resource()
    policy=json.loads((Path(out)/'protocol.json').read_text())['kumo_policy']
    if (policy['settings']!=SETTINGS
            or policy['source_environment_sha256']!=hashlib.sha256(environment_lock().encode()).hexdigest()):
        raise RuntimeError('Frozen Kumo settings/environment changed.')


def main():
    global DEADLINE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen']);parser.add_argument('--run',default='run_048')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.run):raise ValueError('Single safe run name required.')
    out=ROOT/'artifacts'/args.run
    if out.exists():raise RuntimeError('Screen destination exists; no overwrite.')
    configure_runtime(out);configure()
    def timeout(*_):raise TimeoutError('Fixed7200second Kumo screen budget exceeded.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,7200)
    try:
        iterate.screen(out);assert_frozen(out)
        selection=json.loads((out/'selection.json').read_text())
        selection.update(stage='screen_only_pending_review',automatic_eligibility=False)
        backend.write(out/'selection.json',selection)
    except BaseException as error:
        if out.exists():
            backend.write(out/'screen_failure.json',dict(status='failed_inconclusive',error=repr(error)))
            (out/'selection.json').unlink(missing_ok=True)
            if (out/'screen_results.csv').exists():
                (out/'screen_results.csv').replace(out/'screen_results.partial.csv')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);DEADLINE=None


if __name__=='__main__':main()
