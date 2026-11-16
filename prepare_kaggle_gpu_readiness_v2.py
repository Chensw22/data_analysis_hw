"""Prepare/finalize the last private GPU diagnostic locally; never push or run it."""
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
SOURCES = ['kaggle_gpu_readiness_v2.py', 'prepare_kaggle_gpu_readiness_v2.py',
           'test_kaggle_gpu_readiness_v2.py']
SLUG = 'spaceship-gpu-readiness-v2-20261116'
ACCELERATOR = 'NvidiaTeslaT4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_username():
    from kaggle.api.kaggle_api_extended import KaggleApi
    if 'latest' not in KaggleApi.valid_push_pinning_types:
        raise RuntimeError('Installed official API does not support latest image pinning.')
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        username = client.username
    if not isinstance(username, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', username):
        raise RuntimeError('Official authenticated client has no usable username; do not guess.')
    return username


def current_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def committed_sources():
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *SOURCES], cwd=ROOT).decode()
    if set(filter(None, tracked.split('\0'))) != set(SOURCES):
        raise RuntimeError('Commit all v2 implementation and test files before finalization.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *SOURCES], cwd=ROOT).strip():
        raise RuntimeError('V2 sources differ from committed versions.')
    return current_commit()


def prepare(out, finalize=False):
    out = Path(out).resolve()
    if not out.is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Account metadata must stay under ignored artifacts/.')
    ignored = subprocess.run(['git', 'check-ignore', '-q', str(out/'kernel-metadata.json')], cwd=ROOT)
    if ignored.returncode != 0:
        raise ValueError('Output metadata is not ignored by Git.')
    record_path = out/'preparation.json'
    if finalize:
        commit = committed_sources()
        record = json.loads(record_path.read_text())
        if record['status'] != 'draft_not_pushable':
            raise RuntimeError('Only an unfinalized draft can be finalized; no silent repeat.')
        if record['source_hashes'] != {p:sha(ROOT/p) for p in SOURCES}:
            raise RuntimeError('Draft sources changed; review a new output directory.')
        if (record['code_sha256'] != sha(out/SOURCES[0]) or
                record['code_sha256'] != record['source_hashes'][SOURCES[0]] or
                record['metadata_sha256'] != sha(out/'kernel-metadata.json')):
            raise RuntimeError('Prepared code or metadata changed.')
        metadata = json.loads((out/'kernel-metadata.json').read_text())
        cli = str(Path(sys.executable).with_name('kaggle'))
        record.update(status='prepared_not_pushed', prepared_from_commit=commit,
            finalized_at_utc=datetime.now(timezone.utc).isoformat(), commands=dict(
                push=[cli,'kernels','push','-p',str(out),'--accelerator',ACCELERATOR,'--timeout','180'],
                status=[cli,'kernels','status',metadata['id']],
                logs=[cli,'kernels','logs',metadata['id']],
                output=[cli,'kernels','output',metadata['id'],'-p',str(out/'outputs'),
                        '--file-pattern',r'^gpu_readiness_v2\.json$']))
    else:
        if out.exists():
            raise RuntimeError('Draft exists; use --finalize after review and commit, or a new directory.')
        commit = current_commit()
        owner = official_username()
        metadata = dict(id=f'{owner}/{SLUG}', title='Spaceship GPU Readiness V2 20261116',
            code_file=SOURCES[0], language='python', kernel_type='script', is_private=True,
            enable_gpu=True, enable_tpu=False, enable_internet=False,
            machine_shape=ACCELERATOR, docker_image_pinning_type='latest',
            dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(ROOT/SOURCES[0], out/SOURCES[0])
        (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        record = dict(status='draft_not_pushable', base_commit=commit, prepared_from_commit=None,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(),
            source_hashes={p:sha(ROOT/p) for p in SOURCES}, code_sha256=sha(out/SOURCES[0]),
            metadata_sha256=sha(out/'kernel-metadata.json'), private=True,
            requested_accelerator=ACCELERATOR, image_policy='latest',
            kernel_timeout_seconds=180, own_timeout_seconds=120,
            paid_resources_authorized=False, competition_submission=False, upload_attempted=False,
            diagnostic_attempt_number=2, further_diagnostic_retries_authorized=False,
            hypothesis='Supported T4 accelerator plus explicit latest image may avoid the v1 CPU image assignment.',
            interface_evidence=dict(kaggle_version='2.2.5',
                accelerator='Installed kernels push --help and official docs/kernels.md: NvidiaTeslaT4',
                image='Installed KaggleApi.valid_push_pinning_types and kernels_push: docker_image_pinning_type=latest'))
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(status=record['status'], directory=str(out),
        source_commit=record['prepared_from_commit'], code_sha256=record['code_sha256'])))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/kaggle_gpu_readiness_v2')
    parser.add_argument('--finalize', action='store_true', help='Verify reviewed draft against committed sources; never upload.')
    args = parser.parse_args()
    prepare(args.out, args.finalize)
