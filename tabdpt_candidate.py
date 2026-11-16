"""Fixed full-context TabDPT v1.3 screen on the original development rows."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

import iterate
import tabdpt_adapter as adapter
import tabdpt_resource_probe as backend

ROOT = Path(__file__).resolve().parent
NAME = 'v82_TabDPT13_full8'
BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
PROBE = ROOT/'reports/tabdpt_resources/full_mps'
OWN_SOURCES = ['tabdpt_candidate.py', 'test_tabdpt_candidate.py',
               'tabdpt_adapter.py', 'test_tabdpt_adapter.py']
SETTINGS = dict(**backend.SETTINGS, screen_budget_seconds=7200,
                export_budget_seconds=7200, weekly_reserve_percent=30)
RUN_OUT = None
DEADLINE = None
environment_lock = backend.environment


def release_memory():
    backend.release(None, 'mps')


def check_deadline():
    if DEADLINE is not None and time.monotonic() > DEADLINE:
        raise TimeoutError('Fixed TabDPT stage budget exceeded.')


def verify_resource():
    freeze = json.loads((PROBE/'freeze.json').read_text())
    result = json.loads((PROBE/'result.json').read_text())
    expected = [(128,17,8), (4636,2318,1), (8693,4277,1)]
    if (result['status'] != 'complete' or result['device'] != 'mps'
            or not result['synthetic_only'] or result['accuracy_computed']
            or result['freeze_sha256'] != backend.sha(PROBE/'freeze.json')
            or freeze['settings'] != backend.SETTINGS
            or freeze['checkpoint_sha256'] != backend.WEIGHT_SHA):
        raise RuntimeError('Complete matching synthetic MPS resource evidence required.')
    rows = result['cases']
    if [(r['support_rows'],r['query_rows'],r['ensembles']) for r in rows] != expected:
        raise RuntimeError('Required resource cases missing or changed.')
    for row in rows:
        if (row['actual_forward_support_rows'] != [row['support_rows']]
                or row['full_reload_rows'] != row['query_rows']
                or not row['reload_threshold_identical']
                or not 0 <= row['reload_max_abs_difference'] <= 2e-6
                or not 0 <= row['batching_max_abs_difference'] <= 2e-5):
            raise RuntimeError('Incomplete context/reload/batching verification.')
    if (freeze['sources'] != backend.source_hashes()
            or freeze['environment_sha256'] != hashlib.sha256(environment_lock().encode()).hexdigest()
            or backend.sha(PROBE/'environment.lock.txt') != freeze['environment_sha256']):
        raise RuntimeError('Resource-probed source/environment changed.')
    adapter.verify_sources(); backend.verify_checkpoint()
    return freeze


def configure_runtime(out, stage='screen'):
    global RUN_OUT, DEADLINE
    if stage not in ('screen','export'):
        raise ValueError(stage)
    RUN_OUT = Path(out)
    DEADLINE = time.monotonic()+SETTINGS[stage+'_budget_seconds']
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required; no CPU fallback.')
    torch.set_num_threads(4); torch.mps.set_per_process_memory_fraction(.65)


def fit(name, matrices, train_idx, y, predict_idx, seed):
    if name == iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name != NAME or RUN_OUT is None:
        raise RuntimeError('Unknown candidate or unconfigured output.')
    a,b = np.asarray(train_idx,dtype=np.int64),np.asarray(predict_idx,dtype=np.int64)
    if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b) or np.intersect1d(a,b).size:
        raise ValueError('Training/query row IDs must be unique and disjoint.')
    path = RUN_OUT/f'{NAME}_seed{seed}_{adapter.array_sha(a)[:12]}.json'
    if path.exists():
        raise RuntimeError('Existing fit log; no overwrite.')
    check_deadline(); started=time.monotonic()
    model=None
    try:
        x=matrices[False][0]
        model=adapter.fit_context(x.iloc[a],y,a,seed,device='mps')
        backend.write(path,dict(status='predicting',train_rows=len(a),query_rows=len(b)))
        probability=model.predict_proba(x.iloc[b])[:,1]
        check_deadline()
        model.metadata.update(status='complete',seconds=time.monotonic()-started,
            prediction_ids_sha256=adapter.array_sha(b),prediction_rows=len(b))
        backend.write(path,model.metadata)
        return probability,model
    except Exception as error:
        backend.write(path,dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if model is not None:model.release()


def record_protocol(out):
    path=Path(out)/'protocol.json'
    record=json.loads(path.read_text())
    if 'tabdpt_policy' in record:
        assert_frozen(out);return
    freeze=verify_resource()
    sources={**freeze['sources'],**{p:backend.sha(ROOT/p) for p in OWN_SOURCES}}
    for name in ['freeze.json','result.json','environment.lock.txt']:
        p=PROBE/name;sources[str(p.relative_to(ROOT))]=backend.sha(p)
    for relative,digest in sources.items():
        record['sources'][relative]=digest
        target=Path(out)/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    record['tabdpt_policy']=dict(settings=SETTINGS,automatic_eligibility=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        comparison='Original CatBoost baseline rerun; original Mitra/v44 are descriptive historical references.',
        disclosure='Reused6954development rows; real-data pretrained model, no claim of proven contamination-free evaluation.')
    backend.write(path,record)


def evaluate(name,matrices,train,dev,seed,out):
    if len(dev)!=6954 or seed!=iterate.SEEDS[0]:
        raise ValueError('Original development partition and screen seed required.')
    record_protocol(out)
    return BASE_EVALUATE(name,matrices,train,dev,seed,out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:return
    if iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE or BASE_FIT.__module__!='iterate':
        raise RuntimeError('Use a fresh process without another patched wrapper.')
    iterate.CONFIGS={iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
                     NAME:dict(context=False,model='tabdpt13',**SETTINGS)}
    iterate.BLEND=[];iterate.fit,iterate.evaluate=fit,evaluate


def assert_frozen(out):
    configure();iterate.assert_frozen(Path(out));verify_resource()
    if json.loads((Path(out)/'protocol.json').read_text())['tabdpt_policy']['settings']!=SETTINGS:
        raise RuntimeError('Frozen TabDPT settings changed.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen']);parser.add_argument('--run',default='run_047')
    args=parser.parse_args();out=ROOT/'artifacts'/args.run
    if out.exists():raise RuntimeError('Existing run; no overwrite.')
    verify_resource();configure();configure_runtime(out)
    def timeout(*_):raise TimeoutError('Fixed screen time budget exceeded.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,7200)
    try:
        iterate.screen(out);assert_frozen(out)
        selection=json.loads((out/'selection.json').read_text())
        selection.update(stage='screen_only_pending_review',automatic_eligibility=False)
        backend.write(out/'selection.json',selection)
    except Exception as error:
        if out.exists():
            backend.write(out/'screen_failure.json',dict(status='failed_inconclusive',error=repr(error)))
            (out/'selection.json').unlink(missing_ok=True)
            if (out/'screen_results.csv').exists():
                (out/'screen_results.csv').replace(out/'screen_results.partial.csv')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':main()
