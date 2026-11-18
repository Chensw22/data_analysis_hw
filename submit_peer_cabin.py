"""Check and submit the peer-cabin prediction file once."""
import fcntl
import json
import subprocess

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle_workflow as workflow
from peer_cabin_impute import PROPOSAL
from peer_planet_impute import ROOT, SOURCE

REPORT = ROOT/"reports"/PROPOSAL
OUT = ROOT/"artifacts"/PROPOSAL


def main():
    with (ROOT/"artifacts/online_iteration_20261117/submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = json.loads((REPORT/"manifest.json").read_text())
        check = workflow.validate_submission(OUT/"submission.csv", ROOT/"data/raw/test.csv")
        assert check["sha256"] == result["sha256"] and result["reload_max_difference"] == 0
        assert workflow.sha256(SOURCE/"model.joblib") == result["source_model_sha256"]
        assert result["changes_from_v96"] > 0
        table = pd.read_csv(OUT/"test_probabilities.csv")
        submission = pd.read_csv(OUT/"submission.csv")
        assert table.PassengerId.equals(submission.PassengerId)
        assert table.probability.equals(table.reload_probability)
        assert (table.probability >= .5).equals(submission.Transported)
        for path in (ROOT/"reports").rglob("online_attempts/*.json"):
            if json.loads(path.read_text()).get("file_sha256") == check["sha256"]:
                raise ValueError("Existing submission attempt for the same predictions")
        if subprocess.check_output(["git","status","--porcelain"], cwd=ROOT):
            raise ValueError("Commit pending changes before submission")
        api = KaggleApi()
        api.authenticate()
        limits = json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        print(limits, flush=True)
        if limits["numAllowedNow"] <= 0:
            raise ValueError("Kaggle daily limit reached")
        subprocess.Popen(["osascript", str(ROOT/"remind_online_first_submission.applescript"),
                          "v106 同行舱位补全已完成复核，按已有授权提交一次。"],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        workflow.submit(PROPOSAL, 120)


if __name__ == "__main__":
    main()
