"""Run one primary-account follow-up after the queued 160-change file has been scored."""
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone

from await_service_pair_slot_n23 import DIGEST, OUT, REPORT, ROOT, SSH, available, write
from kaggle_workflow import COMPETITION, sha256, validate_submission
from submit_committed_export import submit

WORK = ROOT/'artifacts/nov23_forward_release_transfer'
BASELINE = .81692
LINEAR = 'proposal_n23_service_bag_quarter'
UNION = 'proposal_n23_all_booking_union'
FILES = {
    LINEAR: ('service_bag.npz', 'aea3384033faa17529f5f42809b8534ef386ce1bc78c8c6f6cc53160b0e8c987',
             'export_service_bag_n23.py', 'e902dba0d2b6f1fd171ad5c06613baceff5a578a'),
    UNION: ('recipe.json', '76ea7ffbfa1e444307bd3d523b23e922723fa37cc7638b73df8a56b41174d3a5',
            'export_large_booking_union_n23.py', 'a25cce1171fedebefc43c583be2afc0eb06fdb05'),
}


def read(path):
    return json.loads(path.read_text())


def state(phase, **info):
    write(WORK/'followup_status.json', dict(phase=phase, at_utc=datetime.now(timezone.utc).isoformat(), **info))


def verify(name):
    model, digest, _, revision = FILES[name]
    folder = ROOT/'artifacts'/name
    manifest = read(folder/'manifest.json')
    for file in ['manifest.json', 'selection.json']:
        expected = subprocess.check_output(['git', 'show', f'{revision}:reports/{name}/{file}'], cwd=ROOT)
        assert expected == (folder/file).read_bytes()
    assert manifest['sha256'] == digest and manifest['model_sha256'] == sha256(folder/model)
    assert validate_submission(folder/'submission.csv', ROOT/'data/raw/test.csv')['sha256'] == digest
    return manifest


def choice(score, rank):
    if score > .85 and rank is not None and 0 < rank <= 3:
        return None
    return UNION if score >= BASELINE else LINEAR


def record(name, manifest, receipt, first, third):
    model, digest, source, revision = FILES[name]
    report = receipt.parent.parent
    attempt = read(receipt)
    assert attempt['file_sha256'] == digest and attempt['evaluation']['Status'] == 'COMPLETE'
    score = float(attempt['evaluation']['Public Score'])
    rank = attempt.get('leaderboard_snapshot', {})
    result = report/'followup_result.json'
    write(result, dict(ref=attempt['submission_ref'], score=score, rank=rank, third=third,
                       first_ref=first['ref'], first_score=first['score']))
    paths = [receipt, result]
    best_file = ROOT/'reports/current_best_submission.json'
    best = read(best_file)
    if score > best['public_score']:
        dependencies = (read(ROOT/'artifacts'/name/model)['members'] if name == UNION
                        else {v:manifest[v] for v in ['v108', 'v131']})
        write(best_file, dict(version=manifest['version'], proposal=name, submission_ref=attempt['submission_ref'],
                              public_score=score, upload_commit=attempt['commit'], csv_sha256=digest,
                              source_model_sha256=manifest['model_sha256'], source_model_path=f'artifacts/{name}/{model}',
                              source_file=source, export_recipe_commit=revision, artifact_directory=f'artifacts/{name}',
                              report_directory=f'reports/{name}', training_rows=manifest.get('training_rows', 0),
                              full_roster_training_rows=8693, query_rows=manifest['rows'], leaderboard_snapshot=rank,
                              goal_top5_met=bool(rank.get('rank_fraction') and rank['rank_fraction'] <= .05),
                              S_supported=False, previous_best_public_score=best['public_score'],
                              online_record=str(receipt.relative_to(ROOT)), dependencies=dependencies))
    if read(best_file)['csv_sha256'] == digest:
        paths.append(best_file)
    names = [str(p.relative_to(ROOT)) for p in paths]
    subprocess.run(['git', 'add', '--', *names], cwd=ROOT, check=True)
    diff = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *names], cwd=ROOT)
    if diff.returncode == 1:
        subprocess.run(['git', 'commit', '--only', '-m', f'Record {name} public score {score:.5f}', '--', *names],
                       cwd=ROOT, check=True)
    else:
        diff.check_returncode()
    subprocess.run(['git', '-c', f'core.sshCommand={SSH}', 'push', 'origin', 'main'], cwd=ROOT, check=True)
    state('completed', proposal=name, ref=attempt['submission_ref'], score=score)


def run():
    WORK.mkdir(exist_ok=True)
    with (WORK/'followup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = WORK/'followup_status.json'
        phase = read(status)['phase'] if status.exists() else None
        if phase in ('completed', 'goal_reached'):
            return
        for name in FILES:
            verify(name)
        while read(OUT/'queue_status.json')['phase'] != 'completed':
            state('awaiting_first_score', first_sha256=DIGEST)
            time.sleep(300)
        first = read(REPORT/'queued_result.json')
        first_receipt, = (REPORT/'online_attempts').glob('*.json')
        raw = read(first_receipt)
        assert raw['file_sha256'] == DIGEST and raw['evaluation']['Status'] == 'COMPLETE'
        assert raw['submission_ref'] == first['ref'] and float(raw['evaluation']['Public Score']) == first['score']
        name = choice(first['score'], first['rank'].get('account_best_rank'))
        best = read(ROOT/'reports/current_best_submission.json')
        if name is None or choice(best['public_score'], best['leaderboard_snapshot'].get('account_best_rank')) is None:
            state('goal_reached', first_ref=first['ref'])
            return
        manifest = verify(name)
        report = ROOT/'reports'/name
        while True:
            receipts = list((report/'online_attempts').glob('*.json'))
            if not receipts and phase in ('submitting', 'needs_reconciliation'):
                raise RuntimeError('Inspect previous upload before a retry.')
            try:
                api, limits = available()
                if receipts:
                    third = float(api.competition_leaderboard_view(COMPETITION, page_size=3)[2].score)
            except PermissionError:
                raise
            except Exception as error:
                state('awaiting_api', proposal=name, error_type=type(error).__name__)
                time.sleep(300)
                continue
            if receipts:
                receipt, = receipts
                record(name, manifest, receipt, first, third)
                return
            if limits['numAllowedNow'] <= 0:
                state('awaiting_slot', proposal=name, limits=limits)
                time.sleep(900)
                continue
            state('submitting', proposal=name, sha256=manifest['sha256'], limits=limits)
            try:
                submit(name, FILES[name][0], f'主账号首份评测为 {100*first["score"]:.3f}%。即将提交预选的后续版本 {name}。')
            except BaseException:
                state('needs_reconciliation', proposal=name, sha256=manifest['sha256'])
                raise
            # The wrapper preserves an official receipt before returning.
            receipt, = (report/'online_attempts').glob('*.json')
            assert read(receipt)['file_sha256'] == manifest['sha256']
            state('recording', proposal=name)
            phase = 'recording'


if __name__ == '__main__':
    run()
