"""Submit the completed v108 candidate once after verifying the official quota."""
import fcntl
import json
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import kaggle_workflow as workflow
from export_party_bill import OUT,REPORT,ROOT,NAME,SOURCE


def run():
    with (ROOT/"artifacts/online_iteration_20261117/submission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((REPORT/"manifest.json").read_text())
        check=workflow.validate_submission(OUT/"submission.csv",ROOT/"data/raw/test.csv")
        assert check["sha256"]==manifest["sha256"] and manifest["reload_difference"]==0
        assert workflow.sha256(SOURCE/"submission.csv")==manifest["source_csv"]
        assert workflow.sha256(OUT/"peer_coefficients.joblib")==manifest["correction_sha256"]
        probs=pd.read_csv(OUT/"test_probabilities.csv",float_precision="round_trip")
        csv=pd.read_csv(OUT/"submission.csv")
        assert probs.PassengerId.equals(csv.PassengerId) and probs.probability.equals(probs.reload_probability)
        assert (probs.probability>=.5).equals(csv.Transported)
        for attempt in (ROOT/"reports").rglob("online_attempts/*.json"):
            if json.loads(attempt.read_text()).get("file_sha256")==check["sha256"]:
                raise ValueError("Already submitted")
        if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT):
            raise ValueError("Commit pending work before upload")
        api=KaggleApi();api.authenticate()
        limits=json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        print(limits,flush=True)
        if limits["numAllowedNow"]<=0:
            raise ValueError("No available submissions")
        subprocess.Popen(["osascript",str(ROOT/"remind_online_first_submission.applescript"),
                          "v108 同行消费校正已复核，即将提交一次。"],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        workflow.submit(NAME,120)


if __name__=="__main__":
    run()
