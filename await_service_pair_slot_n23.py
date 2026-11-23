"""One primary-account submission of the frozen 160-change candidate after capacity returns."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_workflow import COMPETITION, sha256, validate_submission
from submit_committed_export import submit

ROOT=Path(__file__).resolve().parent
NAME='proposal_n23_service_pair_zero_donor'
OUT=ROOT/'artifacts'/NAME
REPORT=ROOT/'reports'/NAME
DIGEST='b93762aed4453db157f1f5143339c855a6c151ce06c259125f4e4ea5fd23ac4e'
MODEL='paid_bills.cbm'
REFERENCE='58653601'
SSH='ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=18 -o ConnectionAttempts=2'


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def state(phase,**fields):
    write(OUT/'queue_status.json',dict(phase=phase,at_utc=datetime.now(timezone.utc).isoformat(),**fields))
    print(phase,flush=True)


def check_file():
    manifest=json.loads((REPORT/'manifest.json').read_text())
    assert manifest['sha256']==DIGEST and manifest['model_sha256']==sha256(OUT/MODEL)
    assert manifest['reload_difference']==0 and manifest['v131_changes']==160
    assert validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')['sha256']==DIGEST
    assert (OUT/'manifest.json').read_bytes()==(REPORT/'manifest.json').read_bytes()
    return manifest


def available():
    api=KaggleApi()
    api.authenticate()
    rows=api.competition_submissions(COMPETITION,page_size=100)
    if not any(str(r.ref)==REFERENCE for r in rows):
        raise PermissionError('Primary-account reference was not found.')
    limits=json.loads(str(api.competition_get_submission_limits(COMPETITION)))
    return api,limits


def record(api, receipt, manifest):
    attempt=json.loads(receipt.read_text())
    assert attempt['file_sha256']==DIGEST and attempt['evaluation']['Status']=='COMPLETE'
    score=float(attempt['evaluation']['Public Score'])
    leaders=api.competition_leaderboard_view(COMPETITION,page_size=3)
    rank=attempt.get('leaderboard_snapshot',{})
    result=REPORT/'queued_result.json'
    write(result,dict(ref=attempt['submission_ref'],score=score,rank=rank,
                      third=float(leaders[2].score),above_85=score>.85))
    paths=[receipt,result]
    best_file=ROOT/'reports/current_best_submission.json'
    old=json.loads(best_file.read_text())
    if score>float(old['public_score']):
        write(best_file,dict(version=manifest['version'],proposal=NAME,submission_ref=attempt['submission_ref'],
                             public_score=score,upload_commit=attempt['commit'],csv_sha256=DIGEST,
                             source_model_sha256=manifest['model_sha256'],source_model_path=str((OUT/MODEL).relative_to(ROOT)),
                             artifact_directory=str(OUT.relative_to(ROOT)),report_directory=str(REPORT.relative_to(ROOT)),
                             source_file='export_zero_donor_pairs_n23.py',export_recipe_commit='ae233e0bfae7db383da7ef9fecdced578e3a51c6',
                             training_rows=manifest['model_training_rows'],full_roster_training_rows=8693,query_rows=manifest['rows'],
                             leaderboard_snapshot=rank,goal_top5_met=bool(rank.get('rank_fraction') and rank['rank_fraction']<=.05),
                             S_supported=False,online_record=str(receipt.relative_to(ROOT)),dependencies=manifest['parents'],
                             previous_best_public_score=old['public_score']))
    if json.loads(best_file.read_text()).get('csv_sha256')==DIGEST:
        paths.append(best_file)
    names=[str(p.relative_to(ROOT)) for p in paths]
    subprocess.run(['git','add','--',*names],check=True,cwd=ROOT)
    if subprocess.run(['git','diff','--quiet','HEAD','--',*names],cwd=ROOT).returncode:
        subprocess.run(['git','commit','--only','-m',f'Record 160-change candidate public score {score:.5f}','--',*names],check=True,cwd=ROOT)
    subprocess.run(['git','-c',f'core.sshCommand={SSH}','push','origin','main'],check=True,cwd=ROOT)
    state('completed',score=score,ref=attempt['submission_ref'])


def run():
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=check_file()
        status=OUT/'queue_status.json'
        old=json.loads(status.read_text()).get('phase') if status.exists() else None
        if old=='completed':
            return
        while True:
            receipts=list((REPORT/'online_attempts').glob('*.json'))
            if not receipts and old in ('submitting','needs_reconciliation'):
                raise RuntimeError('Reconcile the previous attempt before another upload.')
            try:
                api,limits=available()
            except PermissionError:
                raise
            except Exception as exc:
                state('waiting_for_api',error_type=type(exc).__name__)
                time.sleep(900)
                continue
            if receipts:
                assert len(receipts)==1
                break
            if limits['numAllowedNow']==0:
                state('waiting_for_slot',limits=limits,sha256=DIGEST)
                time.sleep(900)
                continue
            check_file()
            state('submitting',limits=limits,sha256=DIGEST)
            try:
                submit(NAME,MODEL,'主账号额度已刷新。即将提交已核验的 160 人改判版，完成后记录公开分。')
            except BaseException:
                state('needs_reconciliation',sha256=DIGEST)
                raise
            receipts=list((REPORT/'online_attempts').glob('*.json'))
            assert len(receipts)==1
            break
        state('recording_result')
        record(api,receipts[0],manifest)


if __name__=='__main__':
    run()
