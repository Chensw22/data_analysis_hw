"""One-shot release of the verified v115 file when the main Kaggle account has capacity."""
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from kaggle.api.kaggle_api_extended import KaggleApi

import kaggle_workflow as workflow
from export_consensus_routes import NAME,VERSION
from submit_committed_export import submit

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/"reports"/NAME
OUT=ROOT/"artifacts"/NAME
APPROVED_FILE="f4793198fa592727974757fd9e35783e7bfa37613a4c6c4b8e2a1f9055fcee69"
MAIN_REFERENCE="58503435"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes"


def write(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    tmp.replace(path)


def status(phase,**detail):
    write(OUT/"queue_status.json",dict(phase=phase,at=datetime.now(timezone.utc).isoformat(),**detail))


def wait_for_slot():
    while True:
        try:
            api=KaggleApi();api.authenticate()
            history=api.competition_submissions(workflow.COMPETITION,page_size=100)
            if not any(str(row.ref)==MAIN_REFERENCE for row in history):
                raise PermissionError("The current account does not own the reference submission.")
            limits=json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        except PermissionError:
            raise
        except Exception as error:
            status("waiting_for_official_api",error_type=type(error).__name__)
        else:
            status("slot_available" if limits["numAllowedNow"] else "waiting_for_daily_slot",limits=limits)
            if limits["numAllowedNow"]>0:
                return api
        time.sleep(900)


def publish(api,attempt):
    row=json.loads(attempt.read_text())
    if row["file_sha256"]!=APPROVED_FILE or row["evaluation"]["Status"]!="COMPLETE":
        raise ValueError("Pending attempt requires reconciliation.")
    score=float(row["evaluation"]["Public Score"])
    rank=row.get("leaderboard_snapshot",{})
    top=[float(e.score) for e in api.competition_leaderboard_view(workflow.COMPETITION,page_size=3)]
    result=dict(public_score=score,submission_ref=row["submission_ref"],leaderboard=rank,top_three=top,
                observed_utc=datetime.now(timezone.utc).isoformat(),over_85_percent=score>.85,
                top_three_account=rank.get("account_best_rank") in (1,2,3))
    result_path=REPORT/"queued_result.json"
    write(result_path,result)
    changes=[attempt,result_path]
    best_path=ROOT/"reports/current_best_submission.json"
    best=json.loads(best_path.read_text())
    if score>float(best["public_score"]):
        manifest=json.loads((REPORT/"manifest.json").read_text())
        best=dict(version=VERSION,proposal=NAME,submission_ref=row["submission_ref"],public_score=score,
            upload_commit=row["commit"],csv_sha256=APPROVED_FILE,source_model_sha256=manifest["model_sha256"],
            source_model_path=str((OUT/"router.json").relative_to(ROOT)),artifact_directory=str(OUT.relative_to(ROOT)),
            report_directory=str(REPORT.relative_to(ROOT)),source_file="export_consensus_routes.py",
            export_recipe_commit=manifest["recipe_commit"],training_rows=8693,query_rows=4277,leaderboard_snapshot=rank,
            goal_top5_met=bool(rank.get("rank_fraction") and rank["rank_fraction"]<=.05),S_supported=False,
            online_record=str(attempt.relative_to(ROOT)),dependencies=manifest["dependencies"])
        write(best_path,best);changes.append(best_path)
    # Avoid committing unrelated staged work in the shared checkout.
    paths=[str(v.relative_to(ROOT)) for v in changes]
    subprocess.run(["git","add","--",*paths],check=True,cwd=ROOT)
    if subprocess.run(["git","diff","--cached","--quiet","--",*paths],cwd=ROOT).returncode==1:
        subprocess.run(["git","commit","--only","-m",f"Record v115 main-account public score {score:.5f}","--",*paths],cwd=ROOT,check=True)
    subprocess.run(["git","-c",f"core.sshCommand={SSH}","push","origin","main"],cwd=ROOT,check=True)
    status("completed",**result)


def run():
    with (OUT/"quota_wait.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check=workflow.validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
        if check["sha256"]!=APPROVED_FILE:
            raise ValueError("Queued v115 differs from the approved export.")
        records=sorted((REPORT/"online_attempts").glob("*.json"))
        state_path=OUT/"queue_status.json"
        previous=json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous.get("phase")=="completed":
            return
        if not records:
            if previous.get("phase") in ("submitting","needs_reconciliation"):
                raise RuntimeError("Inspect prior attempt before any new upload.")
            api=wait_for_slot()
            status("submitting",sha256=APPROVED_FILE)
            try:
                submit(NAME,"router.json","v115 错例专家联合已就绪，主账号额度已刷新，即将提交一次。")
            except BaseException:
                status("needs_reconciliation",sha256=APPROVED_FILE)
                raise
            records=sorted((REPORT/"online_attempts").glob("*.json"))
        else:
            api=KaggleApi();api.authenticate()
        if len(records)!=1:
            raise RuntimeError("Expected a single v115 attempt for this release.")
        status("scored_awaiting_publication")
        publish(api,records[0])


if __name__=="__main__":
    run()
