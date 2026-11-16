"""One no-retry upload within the user's bounded November 16 iteration batch."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

import kaggle_workflow as workflow
from quota_guard import latest_weekly, assess

ROOT = Path(__file__).resolve().parent
BATCH = 'online_iteration_20261116'
AUTH = ROOT/'reports'/BATCH/'authorization.json'


def read(path):
    return json.loads(Path(path).read_text())


def count_attempts(reports, batch):
    count = 0
    for path in Path(reports).glob('*/authorization.json'):
        authorization = read(path)
        if authorization.get('batch_id') == batch:
            # Includes unresolved/failed attempts: a missing score is not a refund.
            count += len(list((path.parent/'online_attempts').glob('*.json')))
    return count


def verify(proposal):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', proposal):
        raise ValueError('Invalid proposal name.')
    batch = read(AUTH)
    if (batch['batch_id'] != BATCH or batch['maximum_upload_attempts'] != 10
            or batch['weekly_reserve_percent'] != 30):
        raise RuntimeError('Batch scope changed.')
    out, report = ROOT/'artifacts'/proposal, ROOT/'reports'/proposal
    bound, manifest = read(out/'authorization.json'), read(out/'manifest.json')
    for name in ['authorization.json', 'manifest.json', 'selection.json']:
        if read(out/name) != read(report/name):
            raise RuntimeError('Committed report and local proposal differ.')
    if (bound.get('batch_id') != BATCH or bound.get('proposal') != proposal
            or bound.get('source_authorization_sha256') != workflow.sha256(AUTH)
            or bound.get('maximum_upload_attempts_for_this_file') != 1):
        raise RuntimeError('Proposal lacks this batch authorization.')
    if manifest.get('status') != 'complete' or not manifest.get('export_reload_inference_verified'):
        raise RuntimeError('Complete export and reload verification required.')
    check = workflow.validate_submission(out/'submission.csv', ROOT/'data/raw/test.csv')
    if bound.get('file_sha256') != check['sha256'] or manifest.get('sha256') != check['sha256']:
        raise RuntimeError('Bound CSV changed.')
    if bound.get('model_sha256') != workflow.sha256(out/'model.joblib'):
        raise RuntimeError('Bound model changed.')
    selection = read(out/'selection.json')
    if bound.get('version') != selection.get('version'):
        raise RuntimeError('Bound model version differs.')
    provenance = manifest['provenance']
    if not provenance['source_hashes']:
        raise RuntimeError('Missing source provenance.')
    for path, digest in provenance['source_hashes'].items():
        if workflow.sha256(ROOT/path) != digest:
            raise RuntimeError('Prepared source changed: '+path)
    decision_path = ROOT/provenance['decision_path']
    if (workflow.sha256(decision_path) != provenance['decision_sha256']
            or provenance['decision_sha256'] != bound.get('decision_sha256')):
        raise RuntimeError('Prepared candidate decision changed.')
    decision = read(decision_path)
    if (decision['evidence_files'] != provenance['evidence_files']
            or decision['proposal'] != proposal or decision['version'] != bound['version']):
        raise RuntimeError('Decision evidence or identity differs.')
    for path, digest in decision['evidence_files'].items():
        if workflow.sha256(ROOT/path) != digest:
            raise RuntimeError('Prepared validation evidence changed: '+path)
    for path in (ROOT/'reports').rglob('online_attempts/*.json'):
        if read(path).get('file_sha256') == check['sha256']:
            raise RuntimeError('Identical CSV already attempted; reconcile status instead.')
    used = count_attempts(ROOT/'reports', BATCH)
    if used >= batch['maximum_upload_attempts']:
        raise RuntimeError('All ten authorized attempts are consumed.')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit changes before uploading.')
    return bound, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True)
    parser.add_argument('--session-file', type=Path, required=True)
    parser.add_argument('--wait', type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.wait <= 180:
        parser.error('--wait must be between 1 and 180 seconds.')
    lock_path = ROOT/'artifacts'/BATCH/'submission.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bound, used = verify(args.proposal)
        quota = assess(latest_weekly(args.session_file), reserve=30)
        if not quota['can_start_more_work']:
            raise RuntimeError('Fresh weekly usage must be below 70%; stop/unknown.')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        limits = json.loads(str(api.competition_get_submission_limits('spaceship-titanic')))
        if limits['numAllowedNow'] < 1:
            raise RuntimeError('No Kaggle submissions available now.')
        print(json.dumps(dict(batch_attempt_number=used+1, daily_available=limits['numAllowedNow'],
                              proposal=args.proposal)), flush=True)
        message = f"{bound['version']}：本轮第 {used+1}/10 次探索性测试，即将上传。"
        subprocess.Popen(['osascript', str(ROOT/'remind_exploratory_submission.applescript'), message],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        workflow.submit(args.proposal, args.wait)


if __name__ == '__main__':
    main()
