"""Bounded dependency-only reproduction; no weights, model, GPU or dataset."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

CODE = '''import numpy as np, torch, faiss
print("imports_ok", flush=True)
torch.set_num_threads(4)
faiss.omp_set_num_threads(4)
print("thread_initialization_ok", torch.get_num_threads(), faiss.omp_get_max_threads(), flush=True)
'''


def reproduce():
    started=time.monotonic()
    result=dict(scope='CPU dependency import and declared thread initialization only',
        model_loaded=False,weights_read=False,gpu_used=False,
        versions={name:importlib.metadata.version(name) for name in ['numpy','torch','faiss-cpu']},
        child_source=CODE,timeout_seconds=15)
    try:
        child=subprocess.run([sys.executable,'-u','-c',CODE],capture_output=True,text=True,timeout=15)
        result.update(returncode=child.returncode,stdout=child.stdout,stderr=child.stderr,
            status='dependency_initialization_passed' if child.returncode==0 else 'dependency_initialization_failed')
    except subprocess.TimeoutExpired as error:
        decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
        result.update(status='dependency_initialization_timeout',returncode=None,
            stdout=decode(error.stdout),stderr=decode(error.stderr))
    result['seconds']=time.monotonic()-started
    result['resource_or_model_qualified']=False
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out is not None and args.out.exists():
        raise SystemExit('Output already exists; refusing overwrite.')
    result=reproduce()
    text=json.dumps(result,indent=2)+'\n'
    if args.out is not None:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    print(text,end='')
    raise SystemExit(0 if result['status']=='dependency_initialization_passed' else 2)
