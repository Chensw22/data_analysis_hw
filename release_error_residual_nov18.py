"""Release the already exported v101 once, preserving its frozen export policy."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

import kaggle_workflow as workflow

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / 'reports/error_targeted_20261118/residual_online_release.json'


def read(path):
    return json.loads(path.read_text())


def verify():
    release = read(RELEASE)
    assert release['allow_one_exploratory_submission'] is True
    assert release['maximum_upload_attempts_for_this_file'] == 1
    proposal = 'proposal_ready_nonlinear_residual_v2'
    assert release['proposal'] == proposal
    out, public = ROOT / 'artifacts' / proposal, ROOT / 'reports' / proposal
    for relative, expected in release['bound_files'].items():
        assert workflow.sha256(ROOT / relative) == expected, relative
    manifest = read(out / 'manifest.json')
    assert workflow.sha256(out / 'manifest.json') == workflow.sha256(public / 'manifest.json')
    assert workflow.sha256(out / 'selection.json') == workflow.sha256(public / 'selection.json')
    assert manifest['status'] == 'complete' and manifest['training_rows'] == 8693
    assert manifest['rows'] == manifest['reload_rows'] == 4277
    assert manifest['export_reload_inference_verified'] is True
    assert manifest['reload_max_absolute_difference'] == 0
    assert manifest['version'] == release['version'] == read(out / 'selection.json')['version']
    assert manifest['artifact_kind'] == 'composite_mixed_runtime'
    for name, expected in {**manifest['component_files'], **manifest['raw_component']['files']}.items():
        assert workflow.sha256(out / name) == expected, name
    for name, expected in manifest['source_hashes'].items():
        assert workflow.sha256(ROOT / name) == expected, name
    assert workflow.sha256(out / 'model.joblib') == manifest['model_sha256']
    assert workflow.sha256(out / 'test_probabilities.csv') == manifest['test_probabilities_sha256']
    assert workflow.sha256(ROOT / manifest['policy_path']) == manifest['policy_sha256']
    check = workflow.validate_submission(out / 'submission.csv', ROOT / 'data/raw/test.csv')
    assert check['sha256'] == manifest['sha256'] == release['csv_sha256']
    for attempt in (ROOT / 'reports').rglob('online_attempts/*.json'):
        if read(attempt).get('file_sha256') == check['sha256']:
            raise RuntimeError('Existing attempt: reconcile status; do not repeat upload.')
    return proposal, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    lock = ROOT / 'artifacts/online_iteration_20261117/submission.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proposal, check = verify()
        if not args.submit:
            print(json.dumps({'status': 'offline_release_verified', **check}))
            return
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Commit completed changes before submission.')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        limits = json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        if limits['numAllowedNow'] < 1:
            raise RuntimeError('No submissions available.')
        print(json.dumps({'fresh_daily_limits': limits}), flush=True)
        message = f"v101 非线性纠错：按已有自主提交授权测试一次，当前可用 {limits['numAllowedNow']} 次。"
        subprocess.Popen(['osascript', str(ROOT / 'remind_online_first_submission.applescript'), message],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        workflow.submit(proposal, 120)


if __name__ == '__main__':
    main()
