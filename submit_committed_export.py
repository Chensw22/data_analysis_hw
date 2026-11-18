"""Submit one verified export from its committed revision while unrelated training runs."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

import kaggle_workflow as workflow

ROOT = Path(__file__).resolve().parent


def submit(name, model_filename, notice):
    if not re.fullmatch(r"proposal_[a-zA-Z0-9_]+", name) or Path(model_filename).name != model_filename:
        raise ValueError("Expected a proposal directory and local model filename.")
    base = ROOT / "artifacts" / name
    report = ROOT / "reports" / name
    digest = workflow.validate_submission(base/"submission.csv", ROOT/"data/raw/test.csv")["sha256"]
    releases = ROOT / "artifacts/online_releases"
    releases.mkdir(parents=True,exist_ok=True)
    stage = releases / name
    with (ROOT/"artifacts/online_iteration_20261117/submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record in (ROOT/"reports").glob("*/online_attempts/*.json"):
            if json.loads(record.read_text()).get("file_sha256") == digest:
                raise ValueError("Duplicate competition export: " + str(record))
        if stage.exists():
            raise FileExistsError(f"Previous release exists; reconcile its attempt before reuse: {stage}")
        revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        subprocess.run(["git","clone","--quiet","--shared","--no-checkout",str(ROOT),str(stage)],check=True)
        subprocess.run(["git","checkout","--quiet","--detach",revision],cwd=stage,check=True)
        frozen = stage / "reports" / name
        m = json.loads((frozen/"manifest.json").read_text())
        if (m["sha256"] != digest or m["reload_difference"] != 0 or workflow.sha256(base/model_filename) != m["model_sha256"]):
            raise ValueError("Model, reload verification or submission differs from committed manifest.")
        for file in ["manifest.json","selection.json"]:
            if (base/file).read_bytes() != (frozen/file).read_bytes():
                raise ValueError(f"Uncommitted change to exported {file}")
        p = pd.read_csv(base/"test_probabilities.csv")
        csv = pd.read_csv(base/"submission.csv")
        if not (p.PassengerId.equals(csv.PassengerId) and p.probability.between(0,1).all()
                and p.probability.equals(p.reload_probability) and (p.probability>=.5).equals(csv.Transported)):
            raise ValueError("Probability verification failed.")
        (stage/"artifacts").mkdir()
        (stage/"artifacts"/name).symlink_to(base, target_is_directory=True)
        (stage/"data/raw/test.csv").symlink_to(ROOT/"data/raw/test.csv")
        api = KaggleApi()
        api.authenticate()
        limits = json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        print(limits,flush=True)
        if limits["numAllowedNow"] <= 0:
            shutil.rmtree(stage)
            raise ValueError("Official submission allowance exhausted.")
        subprocess.Popen(["osascript",str(stage/"remind_online_first_submission.applescript"),notice],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        result = subprocess.run([sys.executable,str(stage/"kaggle_workflow.py"),"submit","--run",name,"--wait","120"],cwd=stage)
        attempts = list((frozen/"online_attempts").glob("*.json"))
        if len(attempts) != 1:
            raise RuntimeError(f"Reconcile release in {stage}; no automatic retry.")
        target = report/"online_attempts"
        target.mkdir(exist_ok=True)
        shutil.copy2(attempts[0],target/attempts[0].name)
        for log in (stage/"submission_logs").glob("*.log"):
            shutil.copy2(log,ROOT/"submission_logs"/log.name)
        record = json.loads(attempts[0].read_text())
        if result.returncode or record.get("evaluation",{}).get("Status") != "COMPLETE":
            raise RuntimeError(f"Preserved release in {stage}; reconcile via official status.")
        shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal")
    parser.add_argument("model_filename")
    parser.add_argument("notice")
    args = parser.parse_args()
    submit(args.proposal,args.model_filename,args.notice)
