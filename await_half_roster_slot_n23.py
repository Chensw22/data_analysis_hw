"""Submit the frozen half-roster gate once, after the previous two releases finish."""
import argparse
import fcntl
import re
import subprocess
import time
from datetime import datetime, timezone

from await_service_pair_followup_n23 import WORK as PREVIOUS, commit_records, persist_score, read
from await_service_pair_slot_n23 import ROOT, available, write
from kaggle_workflow import COMPETITION, sha256, validate_submission
from submit_committed_export import submit

NAME = 'proposal_n23_half_roster_disagreement'
MODEL = 'gate.joblib'
SOURCE = 'export_half_roster_disagreement_n23.py'
OUT = ROOT/'artifacts'/NAME
REPORT = ROOT/'reports'/NAME
STATUS = OUT/'queue_status.json'
STAGE = ROOT/'artifacts/online_releases'/NAME
DONE = {'completed', 'goal_reached', 'already_submitted', 'evaluation_failed'}


def verify(revision):
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Pass the full export commit SHA.')
    for path in [REPORT/'manifest.json', REPORT/'selection.json', ROOT/SOURCE]:
        snapshot = subprocess.check_output(['git', 'show', f'{revision}:{path.relative_to(ROOT)}'], cwd=ROOT)
        if path.read_bytes() != snapshot:
            raise ValueError(f'{path.name} differs from the frozen export.')
        if path.parent == REPORT and (OUT/path.name).read_bytes() != snapshot:
            raise ValueError(f'Artifact {path.name} differs from its committed copy.')
    manifest = read(REPORT/'manifest.json')
    checked = validate_submission(OUT/'submission.csv', ROOT/'data/raw/test.csv')
    assert checked['sha256'] == manifest['sha256'] and checked['rows'] == manifest['rows']
    assert manifest['model_sha256'] == sha256(OUT/MODEL) and manifest['reload_difference'] == 0
    assert read(OUT/'selection.json')['version'] == manifest['version']
    assert manifest['parents'] and manifest['training_rows'] > 0
    return manifest


def snapshot(api):
    competitions = api.competitions_list(search=COMPETITION).competitions
    competition, = [c for c in competitions if c.url == f'https://www.kaggle.com/competitions/{COMPETITION}']
    rank, teams = competition.user_rank, competition.team_count
    board = api.competition_leaderboard_view(COMPETITION, page_size=3)
    return dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), account_best_rank=rank or None,
                team_count=teams, rank_fraction=(rank/teams if rank and teams else None)), float(board[2].score)


def status_name(row):
    return str(row.status).rsplit('.', 1)[-1].upper()


def find_remote(rows, manifest):
    suffix = 'sha:' + manifest['sha256'][:12]
    matches = [r for r in rows if suffix in (r.description or '').split()]
    if len(matches) > 1:
        raise RuntimeError('Multiple official submissions match this export; reconcile the records.')
    return matches[0] if matches else None


def reconcile(row, manifest, rank):
    # The immutable digest and commit in the standard submission message also
    # identify an accepted upload if either process died before copying its receipt.
    prefix = NAME + ' ' + manifest['version'] + ' git:'
    match = re.fullmatch(re.escape(prefix) + r'([0-9a-f]{12}) sha:' + manifest['sha256'][:12],
                         row.description or '')
    if not match:
        raise ValueError('Official message does not identify the expected version and commit.')
    revision = subprocess.check_output(['git', 'rev-parse', '--verify', match[1]+'^{commit}'],
                                       cwd=ROOT, text=True).strip()
    committed = subprocess.check_output(['git', 'show', f'{revision}:reports/{NAME}/manifest.json'], cwd=ROOT)
    if committed != (REPORT/'manifest.json').read_bytes():
        raise RuntimeError('The uploaded commit differs from this export.')
    records = list((REPORT/'online_attempts').glob('*.json'))
    if not records:
        records = list((STAGE/'reports'/NAME/'online_attempts').glob('*.json'))
    if len(records) > 1:
        raise RuntimeError('Multiple local attempts need reconciliation.')
    if records:
        attempt = read(records[0])
        assert attempt['commit'] == revision and attempt['file_sha256'] == manifest['sha256']
        assert str(attempt.get('submission_ref', row.ref)) == str(row.ref)
        filename = records[0].name
    else:
        filename = f'official-{row.ref}.json'
        attempt = dict(competition=COMPETITION, run=NAME, version=manifest['version'], commit=revision,
                       rows=manifest['rows'], file_sha256=manifest['sha256'], reconciled_from_official=True)
    receipt = REPORT/'online_attempts'/filename
    receipt.parent.mkdir(exist_ok=True)
    if attempt.get('evaluation', {}).get('Status') == 'COMPLETE' and receipt.exists():
        assert float(attempt['evaluation']['Public Score']) == float(row.public_score)
        if attempt.get('leaderboard_snapshot', {}).get('account_best_rank'):
            return receipt
    attempt.update(submission_ref=str(row.ref), state='submitted', leaderboard_snapshot=rank,
                   last_checked_at_utc=rank['observed_at_utc'], evaluation={
                       'Submission Ref': str(row.ref), 'Status': status_name(row), 'Public Score': str(row.public_score),
                       'Private Score': str(row.private_score or ''), 'Submission Date': str(row.date)})
    write(receipt, attempt)
    return receipt


def run(revision):
    manifest = verify(revision)
    digest = manifest['sha256']
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = read(STATUS) if STATUS.exists() else {}
        if saved and (saved['sha256'] != digest or saved['export_revision'] != revision):
            raise RuntimeError('A different frozen export already owns this queue.')
        attempted = saved.get('attempted', False)

        def state(phase, **fields):
            write(STATUS, dict(phase=phase, attempted=attempted, export_revision=revision, sha256=digest,
                               at_utc=datetime.now(timezone.utc).isoformat(), **fields))

        if saved.get('phase') in DONE:
            return
        while True:
            previous_path = PREVIOUS/'followup_status.json'
            previous = read(previous_path) if previous_path.exists() else {}
            if previous.get('phase') == 'goal_reached':
                state('goal_reached', predecessor=previous)
                return
            if previous.get('phase') == 'completed':
                break
            state('awaiting_predecessor', predecessor_phase=previous.get('phase'))
            time.sleep(300)
        while True:
            try:
                api, limits = available()
                rows = api.competition_submissions(COMPETITION, page_size=100)
                rank, third = snapshot(api)
            except PermissionError:
                raise
            except Exception as error:
                state('awaiting_api', error_type=type(error).__name__)
                time.sleep(300)
                continue
            completed = [r for r in rows if status_name(r) == 'COMPLETE' and r.public_score not in ('', None)]
            first = next((r for r in completed if str(r.ref) == str(previous['ref'])), None)
            if first is None:
                state('awaiting_predecessor_receipt', ref=previous['ref'])
                time.sleep(300)
                continue
            assert float(first.public_score) == float(previous['score'])
            remote = find_remote(rows, manifest)
            if remote is not None:
                if not (remote.description or '').startswith(NAME+' '):
                    state('already_submitted', ref=str(remote.ref), description=remote.description)
                    return
                status = status_name(remote)
                if status in {'ERROR', 'CANCELLED', 'CANCELED'}:
                    receipt = reconcile(remote, manifest, rank)
                    commit_records([receipt], f'Record {NAME} official evaluation {status}')
                    state('evaluation_failed', ref=str(remote.ref), official_status=status)
                    return
                if status != 'COMPLETE':
                    state('awaiting_score', ref=str(remote.ref), official_status=status)
                    time.sleep(60)
                    continue
                receipt = reconcile(remote, manifest, rank)
                score, ref = persist_score(NAME, manifest, receipt, MODEL, digest, SOURCE, revision,
                                           manifest['parents'], third, previous)
                state('completed', ref=ref, score=score)
                return
            existing = [p for p in (ROOT/'reports').glob('*/online_attempts/*.json')
                        if read(p).get('file_sha256') == digest]
            for receipt in existing:
                attempt = read(receipt)
                known = next((r for r in completed if str(r.ref) == str(attempt.get('submission_ref'))), None)
                if known is not None and receipt.parent.parent != REPORT:
                    state('already_submitted', ref=str(known.ref), receipt=str(receipt.relative_to(ROOT)))
                    return
            if existing or attempted or STAGE.exists():
                # An accepted upload can take a little while to appear in official history.
                state('awaiting_reconciliation', receipts=[str(p.relative_to(ROOT)) for p in existing])
                time.sleep(300)
                continue
            if max(float(r.public_score) for r in completed) > .85:
                if not rank['account_best_rank']:
                    state('awaiting_rank')
                    time.sleep(300)
                    continue
                if rank['account_best_rank'] <= 3:
                    state('goal_reached', rank=rank, third=third)
                    return
            if limits['numAllowedNow'] <= 0:
                state('waiting_for_slot', limits=limits)
                time.sleep(900)
                continue
            verify(revision)
            attempted = True
            state('submitting', limits=limits)
            try:
                submit(NAME, MODEL, '前两份候选已评测。即将提交半量校准模型，验证共同错例的修正效果。')
            except (Exception, SystemExit) as error:
                if not STAGE.exists() and (isinstance(error, BlockingIOError) or
                                           str(error) == 'Official submission allowance exhausted.'):
                    attempted = False   # The shared wrapper raised before calling the upload command.
                state('rechecking_official_history', error_type=type(error).__name__)
            time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--check', action='store_true', help='Verify the committed export offline and exit.')
    args = parser.parse_args()
    if args.check:
        print(verify(args.revision)['sha256'])
    else:
        run(args.revision)
