"""Prepare a private, no-network GPU probe locally. Never push or execute it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = ['kaggle_gpu_readiness.py', 'prepare_kaggle_gpu_readiness.py',
           'test_kaggle_gpu_readiness.py']
SLUG = 'spaceship-gpu-readiness-20261116'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_username():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        username = client.username
    if not isinstance(username, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', username):
        raise RuntimeError('Official authenticated client has no usable username; do not guess.')
    return username


def committed_sources():
    tracked = subprocess.check_output(['git','ls-files','-z','--',*SOURCES],cwd=ROOT).decode()
    if set(filter(None, tracked.split('\0'))) != set(SOURCES):
        raise RuntimeError('Commit the readiness implementation and tests before preparation.')
    if subprocess.check_output(['git','status','--porcelain','--',*SOURCES],cwd=ROOT).strip():
        raise RuntimeError('Readiness sources differ from committed versions.')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def prepare(out):
    out = Path(out).resolve()
    if not out.is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Metadata containing the account slug must stay under ignored artifacts/.')
    if out.exists():
        raise RuntimeError('Preparation already exists; do not overwrite or silently resubmit.')
    commit = committed_sources()
    owner = official_username()
    metadata = dict(id=f'{owner}/{SLUG}', title='Spaceship GPU Readiness 20261116',
        code_file='kaggle_gpu_readiness.py', language='python', kernel_type='script',
        is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
        machine_shape='NvidiaL4', dataset_sources=[], competition_sources=[],
        kernel_sources=[], model_sources=[])
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT/SOURCES[0], out/SOURCES[0])
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cli = str(Path(sys.executable).with_name('kaggle'))
    record = dict(status='prepared_not_pushed', prepared_from_commit=commit,
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={p:sha(ROOT/p) for p in SOURCES},
        code_sha256=sha(out/SOURCES[0]), metadata_sha256=sha(out/'kernel-metadata.json'),
        private=True, requested_accelerator='NvidiaL4', kernel_timeout_seconds=180,
        own_timeout_seconds=120, paid_resources_authorized=False,
        competition_submission=False, upload_attempted=False,
        commands=dict(
            push=[cli,'kernels','push','-p',str(out),'--accelerator','NvidiaL4','--timeout','180'],
            status=[cli,'kernels','status',metadata['id']],
            logs=[cli,'kernels','logs',metadata['id']],
            output=[cli,'kernels','output',metadata['id'],'-p',str(out/'outputs'),
                    '--file-pattern','^gpu_readiness\\.json$']))
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'], directory=str(out),
                         source_commit=commit, code_sha256=record['code_sha256'])))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/kaggle_gpu_readiness')
    prepare(parser.parse_args().out)
