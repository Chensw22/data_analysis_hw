"""One full8693/4277 eight-view Kumo resource/reload test, after the base probe."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import joblib
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in [BASE,*BASE.parents] if (p/'iterate.py').is_file())
if str(ROOT) not in sys.path:sys.path.append(str(ROOT))
import kumo_adapter as adapter
backend=adapter.backend
PROBE=ROOT/'reports/kumo_resources/full_mps'
SETTINGS=dict(support_rows=8693,query_rows=4277,views=8,seed=20261116,
    device='mps',threads=4,mps_memory_fraction=.65,budget_seconds=7200,
    comparison_atol=2e-5,full_cpu_serialization_and_reload=True)


def compare_record(record):
    return (record.get('threshold_disagreements')==0
        and isinstance(record.get('max_abs_difference'),(int,float))
        and 0<=record['max_abs_difference']<=2e-5
        and isinstance(record.get('tolerance'),(int,float))
        and 0<=record['tolerance']<=2e-5)


def verify_base_resource():
    freeze=json.loads((PROBE/'freeze.json').read_text())
    result=json.loads((PROBE/'result.json').read_text())
    expected=[(128,17,8),(4636,2318,1),(8693,4277,1)]
    if (result.get('status')!='complete' or result.get('device')!='mps'
            or result.get('synthetic_only') is not True or result.get('accuracy_computed') is not False
            or result.get('freeze_sha256')!=backend.sha(PROBE/'freeze.json')
            or freeze.get('device')!='mps' or freeze.get('settings')!=backend.SETTINGS
            or freeze.get('checkpoint_sha256')!=backend.WEIGHT_SHA
            or freeze.get('code_commit')!=backend.CODE_COMMIT
            or freeze.get('model_revision')!=backend.MODEL_REVISION
            or freeze.get('cases')!=[dict(support=s,query=q,views=v) for s,q,v in expected]):
        raise RuntimeError('Complete matching formal base MPS probe required before full8.')
    rows=result.get('cases',[])
    if [(r.get('support_rows'),r.get('query_rows'),r.get('views')) for r in rows]!=expected:
        raise RuntimeError('Required base resource cases missing or changed.')
    for row in rows:
        contexts=row.get('forwarded_contexts',[])
        if (len(contexts)!=row['views'] or row.get('probability_rows')!=row['query_rows']
                or not compare_record(row.get('partition_check',{}))
                or not compare_record(row.get('reload_check',{}))
                or any(c.get('rows')!=row['support_rows'] or c.get('input_device')!='cpu'
                    or c.get('input_dtype')!='torch.float32' or c.get('output_device')!='mps' for c in contexts)):
            raise RuntimeError('Incomplete base context, partition or reload checks.')
    env_sha=hashlib.sha256(backend.environment().encode()).hexdigest()
    if (freeze.get('sources')!=backend.source_hashes()
            or freeze.get('environment_sha256')!=env_sha
            or backend.sha(PROBE/'environment.lock.txt')!=env_sha
            or (ROOT/'requirements-kumo.lock.txt').read_text()!=backend.environment()):
        raise RuntimeError('Base-probed source or environment changed.')
    backend.verify_checkpoint();adapter.verify_sources()
    return freeze


def source_hashes():
    sources=adapter.source_hashes()
    for name in ('kumo_full8_probe.py','test_kumo_full8_probe.py'):
        sources[name]=backend.sha(BASE/name)
    for name in ('freeze.json','result.json','environment.lock.txt'):
        path=PROBE/name;sources[str(path.relative_to(ROOT))]=backend.sha(path)
    return sources


def require_committed(sources):
    if BASE!=ROOT:raise RuntimeError('Review, relocate and commit staged files before formal execution.')
    tracked=subprocess.check_output(['git','ls-files','-z','--',*sources],cwd=ROOT).decode()
    if set(filter(None,tracked.split('\0')))!=set(sources):
        raise RuntimeError('All formal sources and resource evidence must be committed.')
    if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=ROOT).strip():
        raise RuntimeError('Formal sources/evidence differ from committed versions.')
    for relative,expected in sources.items():
        if backend.sha(ROOT/relative)!=expected:raise RuntimeError('Source digest differs: '+relative)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def run(out):
    out=Path(out)
    if out.exists():raise RuntimeError('Full8 destination exists; no overwrite or resume.')
    verify_base_resource();sources=source_hashes();commit=require_committed(sources)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):
        raise RuntimeError('Disable automatic MPS fallback.')
    if not backend.torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback.')
    backend.torch.set_num_threads(4);backend.torch.mps.set_per_process_memory_fraction(.65)
    out.mkdir(parents=True);lock=backend.environment()
    (out/'environment.lock.txt').write_text(lock)
    backend.write(out/'freeze.json',dict(synthetic_only=True,accuracy_computed=False,
        settings=SETTINGS,backend_settings=backend.SETTINGS,device='mps',source_commit=commit,
        checkpoint_sha256=backend.WEIGHT_SHA,sources=sources,
        environment_sha256=backend.sha(out/'environment.lock.txt'),
        base_probe_freeze_sha256=backend.sha(PROBE/'freeze.json'),
        base_probe_result_sha256=backend.sha(PROBE/'result.json'),
        recorded_at_utc=datetime.now(timezone.utc).isoformat()))
    started=time.monotonic();model=restored=None
    path=out/'synthetic_full8.joblib'
    result=dict(status='incomplete',synthetic_only=True,accuracy_computed=False,device='mps',
        freeze_sha256=backend.sha(out/'freeze.json'),support_rows=8693,query_rows=4277,views=8,
        before=backend.snapshot('mps'))
    backend.write(out/'result.json',result)
    def timeout(*_):raise TimeoutError('Fixed7200second full8 resource budget exceeded.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,7200)
    try:
        x,y,z=backend.synthetic(8693,4277,SETTINGS['seed'])
        result['stage']='full8_forward';backend.write(out/'result.json',result)
        model=adapter.fit_context(x,y,np.arange(len(y)),SETTINGS['seed'],'mps')
        first=model.predict_proba(z)
        result.update(first_probability_rows=len(first),forwarded_contexts=model.last_forward_contexts,
            after_forward=backend.snapshot('mps'),stage='cpu_save_reload')
        backend.write(out/'result.json',result)
        joblib.dump(model,path);model.release();model=None;backend.cleanup('mps')
        restored=joblib.load(path)
        if (restored.device!='cpu' or restored._engine is not None or len(restored.labels)!=8693
                or restored.x.device.type!='cpu' or restored.y.device.type!='cpu'
                or any(t.device.type!='cpu' for t in restored.state.values())):
            raise RuntimeError('CPU artifact did not retain complete context and portable state.')
        restored.device='mps'
        result['stage']='full8_reload_forward';backend.write(out/'result.json',result)
        second=restored.predict_proba(z)
        check=backend.compare(first,second)
        result.update(reload_check=check,reload_probability_rows=len(second),
            reload_forwarded_contexts=restored.last_forward_contexts)
        if source_hashes()!=sources or backend.environment()!=lock:
            raise RuntimeError('Sources/environment changed during full8 probe.')
        backend.verify_checkpoint();verify_base_resource()
        result.pop('stage',None)
        result.update(status='complete',seconds=time.monotonic()-started,after=backend.snapshot('mps'))
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),
            seconds=time.monotonic()-started)
        raise
    finally:
        backend.write(out/'result.json',result)
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if model is not None:model.release()
        if restored is not None:restored.release()
        model=restored=None;backend.cleanup('mps');path.unlink(missing_ok=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
