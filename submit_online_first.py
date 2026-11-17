"""Direct exploratory Kaggle upload: file integrity, daily limits, reminder, no retry."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

import kaggle_workflow as workflow

ROOT=Path(__file__).resolve().parent
POLICY=ROOT/'reports/online_iteration_20261117/online_first_policy.json'


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):result.update(part)
    return result.hexdigest()


def verify(proposal):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',proposal):raise ValueError('Invalid proposal name.')
    policy=read(POLICY);out=ROOT/'artifacts'/proposal;public=ROOT/'reports'/proposal
    if policy.get('allow_direct_submit') is not True:raise RuntimeError('Direct upload not authorized.')
    for name in ['manifest.json','selection.json']:
        if digest(out/name)!=digest(public/name):raise RuntimeError('Export report/artifact mismatch.')
    manifest=read(out/'manifest.json');selection=read(out/'selection.json')
    if (manifest.get('status')!='complete' or manifest.get('export_reload_inference_verified') is not True
            or manifest.get('training_rows')!=8693 or manifest.get('rows')!=4277
            or selection.get('version') not in policy['versions'] or manifest.get('version')!=selection['version']):
        raise RuntimeError('Complete authorized full-data export required.')
    check=workflow.validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    if manifest.get('sha256')!=check['sha256'] or manifest.get('model_sha256')!=digest(out/'model.joblib'):
        raise RuntimeError('Prepared CSV/model identity differs.')
    sources=manifest.get('source_hashes')
    if not sources:raise RuntimeError('Missing source identity.')
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Prepared source changed: '+name)
    for path in (ROOT/'reports').rglob('online_attempts/*.json'):
        if read(path).get('file_sha256')==check['sha256']:raise RuntimeError('This CSV already has an upload attempt; reconcile status.')
    used=0
    for auth in (ROOT/'reports').glob('*/authorization.json'):
        if read(auth).get('batch_id')==policy['batch_id']:
            used+=len(list((auth.parent/'online_attempts').glob('*.json')))
    if used>=policy['maximum_new_upload_attempts_current_batch']:raise RuntimeError('Authorized batch attempts exhausted.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit completed changes before upload.')
    return policy,selection,check,used


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',required=True);parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    lock=ROOT/'artifacts/online_iteration_20261117/submission.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        policy,selection,check,used=verify(args.proposal)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        limits=json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        if limits['numAllowedNow']<1:raise RuntimeError('No Kaggle submissions available now.')
        print(json.dumps({'proposal':args.proposal,'version':selection['version'],'daily_limits':limits,
                          'batch_attempts_used':used,'csv_sha256':check['sha256']}),flush=True)
        if not args.submit:return
        auth=ROOT/'reports'/args.proposal/'authorization.json'
        if not auth.exists():raise RuntimeError('Commit per-file authorization before upload.')
        bound=read(auth)
        if (bound.get('batch_id')!=policy['batch_id'] or bound.get('file_sha256')!=check['sha256']
                or bound.get('policy_sha256')!=digest(POLICY) or bound.get('maximum_upload_attempts_for_this_file')!=1):
            raise RuntimeError('Per-file online-first authorization differs.')
        message=f"{selection['version']}：按你最新的线上优先授权，即将提交一次；当前可用{limits['numAllowedNow']}次。"
        subprocess.Popen(['osascript',str(ROOT/'remind_online_first_submission.applescript'),message],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        workflow.submit(args.proposal,120)


if __name__=='__main__':main()
