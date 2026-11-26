"""Run a frozen release once after its predecessor receives an official primary-account score."""
import argparse
import fcntl
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from await_service_pair_followup_n23 import WORK as PREVIOUS, commit_records, persist_score, read
from await_service_pair_slot_n23 import ROOT, available, write
from kaggle_workflow import COMPETITION, sha256, validate_submission
from submit_committed_export import submit

MODEL = 'gate.joblib'
DONE = {'completed', 'goal_reached', 'already_submitted', 'evaluation_failed', 'predecessor_failed', 'members_below_reference', 'reserve_not_needed'}


@dataclass(frozen=True)
class QueuedExport:
    name: str
    source: str
    predecessor: Path
    notice: str
    predecessor_digest: str | None = None
    model: str = MODEL

    @property
    def out(self):
        return ROOT/'artifacts'/self.name

    @property
    def report(self):
        return ROOT/'reports'/self.name

    @property
    def stage(self):
        return ROOT/'artifacts/online_releases'/self.name


HALF = QueuedExport('proposal_n23_half_roster_disagreement', 'export_half_roster_disagreement_n23.py',
                    PREVIOUS/'followup_status.json',
                    '前两份候选已评测。即将提交半量校准模型，验证共同错例的修正效果。')
ADAPTIVE = QueuedExport('proposal_n23_adaptive_bill_prior', 'export_adaptive_bill_prior_n23.py',
                        HALF.out/'queue_status.json',
                        '109 人版已取得官方评分。即将提交消费先验模型，检验另一组乘客的改判。',
                        '58b759b4e767969020c0e06bfa975906e03898bba67825ec3fb2df4489bde05f')
SELECTED = QueuedExport('proposal_n24_selected_dissent', 'export_selected_dissent_n24.py',
                        ROOT/'reports/proposal_v131_party_consensus/online_attempts/20261121T061548706760Z.json',
                        '主账号已有提交额度，即将评测相对 v131 改判 10 人的九票树模型。',
                        'ed644344cf5df8d6e5c213b5c42b109d0efbee43404cb8b00e6bafc2fde187d4',
                        model='ensemble.joblib')
SOLO_RANDOM = QueuedExport('proposal_n24_solo_random_nine', 'export_solo_random_nine_n24.py',
                           SELECTED.out/'queue_status.json',
                           '九票版已取得官方成绩，即将评测加入独行乘客随机字段模型的 70 人版。',
                           '5ab295e3a24cd7b04ed92c66ccef6dea971866416437fa6f5d1af9cadc359223',
                           model='composition.json')
NET_EDITS = QueuedExport('proposal_n24_bagged_net_edits', 'export_bagged_net_edits_n24.py',
                         SOLO_RANDOM.out/'queue_status.json',
                         '70 人版已有官方成绩，即将提交按净纠正人数训练的 41 人版。',
                         '5dea4e65659fa7cb345c6238eedf0109d55e52afd4a4b1da88e523705b522871',
                         model='calibrators.json')
CONFIRMED_FIELDS = QueuedExport('proposal_n25_confirmed_random_fields', 'export_confirmed_random_fields_n25.py',
    ROOT/'reports/proposal_n24_selected_dissent/online_attempts/20261125T070137804381Z.json',
    '主账号额度已刷新，即将提交双分组确认版。与当前最佳相比，该版本改判 30 名乘客。',
    '5ab295e3a24cd7b04ed92c66ccef6dea971866416437fa6f5d1af9cadc359223', model='ensemble.joblib')
AWAKE_PERSONAL = QueuedExport('proposal_n25_awake_personal_tabicl', 'export_awake_personal_n25.py',
    CONFIRMED_FIELDS.out/'queue_status.json',
    '30 人版已获得官方评测，即将提交清醒乘客 TabICL。它相对原九票版改判 135 人。',
    'd023e5cea9086b1c5418fac089960865375ff55727e70fb8761d442d8dbaa582', model='training_context.csv')
CONFIRMED_UNION = QueuedExport('proposal_n25_confirmed_awake_union', 'export_confirmed_awake_union_n25.py',
    AWAKE_PERSONAL.out/'queue_status.json',
    '两个单独版本已有评测，至少一版持平或超过原九票版。即将提交保留任一版本改判的组合方案，共改判 154 人。',
    '9efc0f5ba7fa7a7169a88375e00297b2ebcefca60cdb8b0a95b4261e00d4eeeb', model='composition.json')
AWAKE_FILTER = QueuedExport('proposal_n25_awake_error_confirmation', 'export_awake_error_confirmation_n25.py',
    AWAKE_PERSONAL.out/'queue_status.json',
    '清醒乘客模型已取得正式成绩，即将提交错误模型筛选的 41 人版。',
    '9efc0f5ba7fa7a7169a88375e00297b2ebcefca60cdb8b0a95b4261e00d4eeeb', model='error_context.joblib')
DISPUTE_BAND = QueuedExport('proposal_n26_dispute_band', 'export_dispute_band_n26.py',
    AWAKE_FILTER.out/'queue_status.json',
    '即将向主账号提交 68 个配置的分歧验证版，检验树模型对中等分歧乘客的改判。',
    'ca69b47c31f92b2204ac7abbeaff8caba43ab83e43213ef20a9119f9c9f6464f', model='frozen_votes.npz')
JOINT_DISPUTE = QueuedExport('proposal_n26_joint_dispute_awake', 'export_joint_dispute_awake_n26.py',
    DISPUTE_BAND.out/'queue_status.json',
    '87 人版已有官方评测。即将提交合并分歧与清醒乘客改判的 214 人版。',
    'e375a1a35f90d51a2b01ca66a6f6f17ec5f97e95e05184c6bf2cd6deb4bad7c8', model='composition.json')
CONSENSUS_REVIEW = QueuedExport('proposal_n26_consensus_review', 'export_consensus_review_n26.py',
    JOINT_DISPUTE.out/'queue_status.json',
    '214 人组合已有官方成绩。即将提交神经网络双重确认版。',
    '1f7d0612ba4f1cef17723ae242dbba1787502e4bd69066e30eaca49345c59ef7', model='review_scores.npz')
FARE_DECK = QueuedExport('proposal_n26_fare_deck', 'export_fare_deck_n26.py',
    CONSENSUS_REVIEW.out/'queue_status.json',
    '主账号即将评测精确账单甲板版，该模型在整单验证中救回了此前一致判错的乘客。',
    '8df14a7bb4a404fcfceff3abf7f7c44e9b855b50fca72a2bb2b1d090514c4459', model='scores.npz')
SIX_WAY_DISPUTE = QueuedExport('proposal_n26_six_way_dispute', 'export_six_way_dispute_n26.py',
    FARE_DECK.out/'queue_status.json',
    '主账号即将评测登记与账单六模型一致改判版，上一份账单甲板版已有官方成绩。',
    'a8a864dacdedd6d720f581922b4c5247c7adc7b0636ec1a0aa7cc580b37f5aae', model='predictions.npz')
TABNET_ATTENTION = QueuedExport('proposal_n26_tabnet_attention', 'export_tabnet_attention_n26.py',
    SIX_WAY_DISPUTE.out/'queue_status.json',
    'Kaggle 主账号即将提交 TabNet 双重确认版。本版改判了此前十个版本都判断一致的 18 名乘客。',
    '222a2a6549b2b49a39452c58f203196ae20a150875766f330752bbf8de69bf2d', model='predictions.npz')
HISTORICAL_VETO = QueuedExport('proposal_n26_historical_veto', 'export_historical_veto_n26.py',
    TABNET_ATTENTION.out/'queue_status.json',
    '组合版已跳过，主账号仍有本轮额度。即将评测从两批历史订单选出的备用版。',
    '4f5d0389d8f01e1c375b41521a88c87849f7183106d2537eb12bb9cca168f943', model='composition.json')
CHAMPION_FARE_SIX = QueuedExport('proposal_n26_champion_fare_six', 'rebase_fare_six_n26.py',
    HISTORICAL_VETO.out/'queue_status.json',
    '主账号额度已刷新。即将评测直接从九票公榜最佳出发的账单与六模型版，共改判 111 人。',
    'de14c99a507fb7073023908d8a26987064d134c0e77eaec6c77302caf7a76461', model='scores.npz')
JOINT_FARE_FOUNDATIONS = QueuedExport('proposal_n26_joint_fare_foundations', 'export_joint_fare_foundations_n26.py',
    CHAMPION_FARE_SIX.out/'queue_status.json',
    '111 人版已有正式成绩。即将评测票价 TabICL 与 Nori 都支持的新增 50 人。',
    '9ba377aa456f04b417e0b78cfebc2009828a6c3cd434714d6a4a457876517f01', model='inputs.npz')
JOBS = {job.name: job for job in (HALF, ADAPTIVE, SELECTED, SOLO_RANDOM, NET_EDITS, CONFIRMED_FIELDS,
                                  AWAKE_PERSONAL, CONFIRMED_UNION, AWAKE_FILTER, DISPUTE_BAND, JOINT_DISPUTE, CONSENSUS_REVIEW, FARE_DECK, SIX_WAY_DISPUTE,
                                  TABNET_ATTENTION, HISTORICAL_VETO, CHAMPION_FARE_SIX, JOINT_FARE_FOUNDATIONS)}


def reserve_available(union, expected_digest, official_match, last_submission_date, today, remaining):
    return (union.get('phase') == 'members_below_reference' and union.get('attempted') is False
            and union.get('sha256') == expected_digest and not official_match
            and str(last_submission_date)[:10] == today.isoformat() and remaining > 0)


def verify(revision, job=HALF):
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Pass the full export commit SHA.')
    for path in [job.report/'manifest.json', job.report/'selection.json', ROOT/job.source]:
        snapshot = subprocess.check_output(['git', 'show', f'{revision}:{path.relative_to(ROOT)}'], cwd=ROOT)
        if path.read_bytes() != snapshot:
            raise ValueError(f'{path.name} differs from the frozen export.')
        if path.parent == job.report and (job.out/path.name).read_bytes() != snapshot:
            raise ValueError(f'Artifact {path.name} differs from its committed copy.')
    manifest = read(job.report/'manifest.json')
    checked = validate_submission(job.out/'submission.csv', ROOT/'data/raw/test.csv')
    assert checked['sha256'] == manifest['sha256'] and checked['rows'] == manifest['rows']
    assert manifest['model_sha256'] == sha256(job.out/job.model) and manifest['reload_difference'] == 0
    assert read(job.out/'selection.json')['version'] == manifest['version']
    if job.name == SELECTED.name:
        assert manifest['source_v131'] == job.predecessor_digest
        manifest = dict(manifest, parents={'v131': manifest['source_v131']}, training_rows=manifest['train_rows'])
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


def reconcile(row, manifest, rank, job=HALF):
    # The immutable digest and commit in the standard submission message also
    # identify an accepted upload if either process died before copying its receipt.
    prefix = job.name + ' ' + manifest['version'] + ' git:'
    match = re.fullmatch(re.escape(prefix) + r'([0-9a-f]{12}) sha:' + manifest['sha256'][:12],
                         row.description or '')
    if not match:
        raise ValueError('Official message does not identify the expected version and commit.')
    revision = subprocess.check_output(['git', 'rev-parse', '--verify', match[1]+'^{commit}'],
                                       cwd=ROOT, text=True).strip()
    committed = subprocess.check_output(['git', 'show', f'{revision}:reports/{job.name}/manifest.json'], cwd=ROOT)
    if committed != (job.report/'manifest.json').read_bytes():
        raise RuntimeError('The uploaded commit differs from this export.')
    records = list((job.report/'online_attempts').glob('*.json'))
    if not records:
        records = list((job.stage/'reports'/job.name/'online_attempts').glob('*.json'))
    if len(records) > 1:
        raise RuntimeError('Multiple local attempts need reconciliation.')
    if records:
        attempt = read(records[0])
        assert attempt['commit'] == revision and attempt['file_sha256'] == manifest['sha256']
        assert str(attempt.get('submission_ref', row.ref)) == str(row.ref)
        filename = records[0].name
    else:
        filename = f'official-{row.ref}.json'
        attempt = dict(competition=COMPETITION, run=job.name, version=manifest['version'], commit=revision,
                       rows=manifest['rows'], file_sha256=manifest['sha256'], reconciled_from_official=True)
    receipt = job.report/'online_attempts'/filename
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


def confirm_predecessor(job, saved, official):
    score = float(official.public_score)
    if status_name(official) != 'COMPLETE' or str(official.ref) != str(saved['ref']):
        raise ValueError('The official predecessor reference is incomplete or mismatched.')
    if 'score' in saved and float(saved['score']) != score:
        raise ValueError('The recorded predecessor score differs from its official score.')
    if job.predecessor_digest is not None:
        # Completion is tied to the exact 109-change file, even when the third
        # worker found that the same bytes had already been submitted under another name.
        if saved.get('sha256') != job.predecessor_digest:
            raise ValueError('The predecessor state describes a different export.')
        token = 'sha:' + job.predecessor_digest[:12]
        if token not in (official.description or '').split():
            # Older submissions can lack the digest in their public description.
            receipts = (ROOT/'reports').glob('*/online_attempts/*.json')
            if not any((record := read(path)).get('file_sha256') == job.predecessor_digest
                       and str(record.get('submission_ref')) == str(official.ref) for path in receipts):
                raise RuntimeError('Cannot link the predecessor official reference to the frozen 109-change export.')
    return {**saved, 'score': score}


def run(revision, job=HALF):
    status_path = job.out/'queue_status.json'
    manifest = verify(revision, job)
    digest = manifest['sha256']
    with (job.out/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = read(status_path) if status_path.exists() else {}
        if saved and (saved['sha256'] != digest or saved['export_revision'] != revision):
            raise RuntimeError('A different frozen export already owns this queue.')
        attempted = saved.get('attempted', False)

        def state(phase, **fields):
            write(status_path, dict(phase=phase, attempted=attempted, export_revision=revision, sha256=digest,
                               at_utc=datetime.now(timezone.utc).isoformat(), **fields))

        if saved.get('phase') in DONE:
            return
        while True:
            if job.name == AWAKE_FILTER.name:
                union_status = CONFIRMED_UNION.out/'queue_status.json'
                union = read(union_status) if union_status.exists() else {}
                if union.get('phase') not in DONE:
                    state('awaiting_union_decision', union_phase=union.get('phase'))
                    time.sleep(300)
                    continue
            previous_path = job.predecessor
            previous = read(previous_path) if previous_path.exists() else {}
            if previous.get('evaluation', {}).get('Status') == 'COMPLETE':
                previous = dict(phase='completed', ref=previous['submission_ref'], sha256=previous['file_sha256'],
                                score=float(previous['evaluation']['Public Score']))
            if previous.get('phase') == 'goal_reached':
                state('goal_reached', predecessor=previous)
                return
            if previous.get('phase') in ('evaluation_failed', 'predecessor_failed'):
                state('predecessor_failed', predecessor=previous)
                return
            if previous.get('phase') == 'completed' or (
                    job.predecessor_digest and previous.get('phase') == 'already_submitted'):
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
            previous = confirm_predecessor(job, previous, first)
            remote = find_remote(rows, manifest)
            if remote is not None:
                if not (remote.description or '').startswith(job.name+' '):
                    state('already_submitted', ref=str(remote.ref), description=remote.description)
                    return
                status = status_name(remote)
                if status in {'ERROR', 'CANCELLED', 'CANCELED'}:
                    receipt = reconcile(remote, manifest, rank, job)
                    commit_records([receipt], f'Record {job.name} official evaluation {status}')
                    state('evaluation_failed', ref=str(remote.ref), official_status=status)
                    return
                if status != 'COMPLETE':
                    state('awaiting_score', ref=str(remote.ref), official_status=status)
                    time.sleep(60)
                    continue
                receipt = reconcile(remote, manifest, rank, job)
                score, ref = persist_score(job.name, manifest, receipt, job.model, digest, job.source, revision,
                                           manifest['parents'], third, previous)
                state('completed', ref=ref, score=score)
                return
            existing = [p for p in (ROOT/'reports').glob('*/online_attempts/*.json')
                        if read(p).get('file_sha256') == digest]
            for receipt in existing:
                attempt = read(receipt)
                known = next((r for r in completed if str(r.ref) == str(attempt.get('submission_ref'))), None)
                if known is not None and receipt.parent.parent != job.report:
                    state('already_submitted', ref=str(known.ref), receipt=str(receipt.relative_to(ROOT)))
                    return
            if existing or attempted or job.stage.exists():
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
            if job.name == CONFIRMED_UNION.name:
                tree_state = read(AWAKE_PERSONAL.predecessor)
                nine_receipt = read(CONFIRMED_FIELDS.predecessor)
                refs = {str(r.ref): r for r in completed}
                tree = refs.get(str(tree_state['ref']))
                nine = refs.get(str(nine_receipt['submission_ref']))
                if tree is None or nine is None:
                    state('awaiting_member_scores')
                    time.sleep(300)
                    continue
                tree_score = confirm_predecessor(AWAKE_PERSONAL, tree_state, tree)['score']
                nine_score = float(nine.public_score)
                assert float(nine_receipt['evaluation']['Public Score']) == nine_score
                if max(tree_score, previous['score']) < nine_score:
                    state('members_below_reference', tree_score=tree_score, neural_score=previous['score'],
                          reference_score=nine_score)
                    return
            if job.name == HISTORICAL_VETO.name:
                union = read(CONFIRMED_UNION.out/'queue_status.json')
                original = read(CONFIRMED_UNION.report/'manifest.json')
                if not reserve_available(union, original['sha256'], find_remote(rows, original) is not None,
                                         first.date, datetime.now(timezone.utc).date(), limits['numAllowedNow']):
                    state('reserve_not_needed', union_phase=union.get('phase'), predecessor_ref=previous['ref'],
                          limits=limits)
                    return
            if limits['numAllowedNow'] <= 0:
                state('waiting_for_slot', limits=limits)
                time.sleep(900)
                continue
            verify(revision, job)
            attempted = True
            state('submitting', limits=limits)
            try:
                submit(job.name, job.model, job.notice)
            except (Exception, SystemExit) as error:
                if not job.stage.exists() and (isinstance(error, BlockingIOError) or
                                           str(error) == 'Official submission allowance exhausted.'):
                    attempted = False   # The shared wrapper raised before calling the upload command.
                state('rechecking_official_history', error_type=type(error).__name__)
            time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--proposal', choices=JOBS, default=HALF.name)
    parser.add_argument('--check', action='store_true', help='Verify the committed export offline and exit.')
    args = parser.parse_args()
    if args.check:
        print(verify(args.revision, JOBS[args.proposal])['sha256'])
    else:
        run(args.revision, JOBS[args.proposal])
