"""Official Kaggle CLI workflow with commit/file provenance and no automatic retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
COMPETITION = 'spaceship-titanic'
CLI = Path(sys.executable).with_name('kaggle')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args, capture=False):
    return subprocess.run([str(CLI), *args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None, check=False)


def validate_submission(path, test_path, expected_rows=4277):
    submission = pd.read_csv(path, dtype=str, keep_default_na=False)
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    if list(submission) != ['PassengerId', 'Transported']:
        raise ValueError('Submission columns must be PassengerId,Transported in that order.')
    if len(submission) != expected_rows or len(test) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows.')
    if not submission.PassengerId.is_unique or not submission.PassengerId.equals(test.PassengerId):
        raise ValueError('PassengerId values/order must match the test data exactly.')
    if not submission.Transported.isin(['True', 'False']).all():
        raise ValueError('Transported must contain only literal True/False, no missing values.')
    return {'rows': len(submission), 'sha256': sha256(path), 'valid': True}


def download():
    destination = ROOT/'data/official'
    destination.mkdir(parents=True, exist_ok=True)
    # The bundle endpoint can return 403 while authorized single-file downloads work.
    for name in ['train.csv', 'test.csv', 'sample_submission.csv']:
        result = execute(['competitions', 'download', COMPETITION, '-f', name,
                          '-p', str(destination), '--unzip'])
        if result.returncode:
            raise SystemExit(result.returncode)
    raw = ROOT/'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in ['train.csv', 'test.csv', 'sample_submission.csv']:
        official, local = destination/name, raw/name
        if not official.exists():
            raise RuntimeError(f'Official download missing {name}; no local file was replaced.')
        if local.exists():
            pd.testing.assert_frame_equal(pd.read_csv(official), pd.read_csv(local))
        records[name] = {'official_sha256': sha256(official),
                         'local_sha256_before': sha256(local) if local.exists() else None,
                         'same_table': True if local.exists() else None}
    # Compare every existing file before creating missing local files.
    for name in records:
        if not (raw/name).exists():
            shutil.copy2(destination/name, raw/name)
    report = ROOT/'reports/official_data_verification.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                                 'competition': COMPETITION, 'files': records}, indent=2)+'\n')
    print(f'Official data available and compared: {report}')


def latest_record(run):
    records = sorted((ROOT/'reports'/run/'online_attempts').glob('*.json'))
    if not records:
        raise ValueError('No submission attempt recorded for this run.')
    return records[-1]


def status(run, ref=None):
    record_path = latest_record(run)
    record = json.loads(record_path.read_text())
    ref = ref or record.get('submission_ref')
    if not ref:
        raise ValueError('No submission ref. Check Kaggle submission history; do not resubmit blindly. Then use status --ref ID.')
    result = execute(['competitions', 'submission', str(ref)], capture=True)
    print(result.stdout, end='')
    if result.returncode:
        raise SystemExit(result.returncode)
    record['submission_ref'] = str(ref)
    record['last_checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    # Keep only the fixed public result fields, never credentials or CLI auth diagnostics.
    fields = {'Submission Ref', 'Status', 'Public Score', 'Private Score', 'Submission Date'}
    record['evaluation'] = {k.strip(): v.strip() for line in result.stdout.splitlines()
                            if ':' in line for k, v in [line.split(':', 1)] if k.strip() in fields}
    record_path.write_text(json.dumps(record, indent=2)+'\n')


def submit(run, wait):
    path = ROOT/'artifacts'/run/'submission.csv'
    check = validate_submission(path, ROOT/'data/raw/test.csv')
    # Prevent duplicate attempts even if the last process failed after an accepted upload.
    for old in (ROOT/'reports').glob('*/online_attempts/*.json'):
        if json.loads(old.read_text()).get('file_sha256') == check['sha256']:
            raise ValueError(f'This exact file already has an attempt: {old}. Use status; no duplicate upload.')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError('Commit completed changes before submitting so the result maps to a clean commit.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    selection = json.loads((ROOT/'artifacts'/run/'selection.json').read_text())
    now = datetime.now(timezone.utc)
    attempt = now.strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT/'reports'/run/'online_attempts'
    destination.mkdir(parents=True, exist_ok=True)
    record_path = destination/(attempt+'.json')
    record = dict(competition=COMPETITION, run=run, version=selection['version'], commit=commit,
                  file_sha256=check['sha256'], rows=check['rows'],
                  started_at_utc=now.isoformat(), state='attempt_started')
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    message = f"{run} {selection['version']} git:{commit[:12]} sha:{check['sha256'][:12]}"
    logs = ROOT/'submission_logs'; logs.mkdir(exist_ok=True)
    print(f'Submitting {message}; this command will NOT retry the upload.', flush=True)
    # Stream to a local ignored log so a killed process still leaves evidence of its ref.
    with (logs/(attempt+'.log')).open('w') as log:
        result = subprocess.run([str(CLI), 'competitions', 'submit', COMPETITION,
            '-f', str(path), '-m', message, '--wait', str(wait), '--poll-interval', '20'],
            cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT,
            env={**__import__('os').environ, 'PYTHONUNBUFFERED': '1'}, check=False)
    output = (logs/(attempt+'.log')).read_text()
    print(output, end='')
    match = re.search(r'Submission ref:\s*(\d+)', output)
    record.update(cli_returncode=result.returncode,
                  state='submitted' if match else 'needs_manual_reconciliation')
    if match:
        record['submission_ref'] = match.group(1)
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    if match:
        status(run)
    elif result.returncode:
        raise SystemExit(result.returncode)
    else:
        raise RuntimeError('No submission ref found. Check history before any further upload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['login', 'download', 'check', 'submit', 'status'])
    parser.add_argument('--run', default='run_002')
    parser.add_argument('--wait', type=int, default=180)
    parser.add_argument('--ref', type=int)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run):
        parser.error('Invalid run identifier.')
    if args.wait < 1:
        parser.error('--wait must be positive.')
    if args.action == 'login':
        raise SystemExit(execute(['auth', 'login']).returncode)
    if args.action == 'download':
        download()
    elif args.action == 'check':
        print(json.dumps(validate_submission(ROOT/'artifacts'/args.run/'submission.csv', ROOT/'data/raw/test.csv')))
    elif args.action == 'submit':
        submit(args.run, args.wait)
    elif args.action == 'status':
        status(args.run, args.ref)


if __name__ == '__main__':
    main()
