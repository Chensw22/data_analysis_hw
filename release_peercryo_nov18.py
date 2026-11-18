"""One independently released upload of the existing v99 peer-Cryo artifact."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess

import kaggle_workflow as workflow

ROOT=Path(__file__).resolve().parent
PROPOSAL='proposal_ready_peercryo17'
VERSION='v99_Mitra_peercryo_fixed15_ready'
RELEASE=ROOT/'reports/error_followup_20261118/peercryo_online_release.json'
EXPORT_POLICY='reports/online_iteration_20261117/next_round_peercryo_policy.json'
READY='reports/next_round_20261117/ready_candidate.json'


def read(path):return json.loads(Path(path).read_text())


def required_bound_files():
    manifest=read(ROOT/'artifacts'/PROPOSAL/'manifest.json')
    paths=set(manifest['source_hashes'])
    paths.update([Path(__file__).name,'kaggle_workflow.py','remind_online_first_submission.applescript',EXPORT_POLICY,READY,
        'reports/error_rescue_20261117/feature_probe_result.json'])
    for prefix in ('artifacts','reports'):
        paths.update(f'{prefix}/{PROPOSAL}/{n}' for n in ['manifest.json','selection.json','protocol.json'])
    paths.update(f'artifacts/{PROPOSAL}/{n}' for n in ['model.joblib','submission.csv','test_probabilities.csv',
        'fit_metadata.json','reload_metadata.json','training_log.json','environment.lock.txt'])
    return {name:workflow.sha256(ROOT/name) for name in sorted(paths)}


def verify():
    release=read(RELEASE)
    if (release.get('allow_one_exploratory_submission') is not True
            or release.get('maximum_upload_attempts_for_this_file')!=1
            or release.get('version')!=VERSION or release.get('proposal')!=PROPOSAL):
        raise RuntimeError('Exact single-file v99 release required.')
    if release.get('bound_files')!=required_bound_files():raise RuntimeError('Released source/artifact bindings changed.')
    out,public=ROOT/'artifacts'/PROPOSAL,ROOT/'reports'/PROPOSAL
    for name in ['manifest.json','selection.json','protocol.json']:
        if workflow.sha256(out/name)!=workflow.sha256(public/name):raise RuntimeError('Public/artifact mismatch: '+name)
    manifest=read(out/'manifest.json');selection=read(out/'selection.json');protocol=read(out/'protocol.json')
    fit=read(out/'fit_metadata.json');reload=read(out/'reload_metadata.json');policy=read(ROOT/EXPORT_POLICY);ready=read(ROOT/READY)
    if (manifest.get('status')!='complete' or manifest.get('training_rows')!=8693 or manifest.get('fit_rows')!=8693
            or manifest.get('rows')!=4277 or manifest.get('reload_rows')!=4277 or manifest.get('feature_count')!=17
            or manifest.get('roster_rows')!=12970 or manifest.get('export_reload_inference_verified') is not True
            or manifest.get('reload_max_absolute_difference')!=0 or manifest.get('reload_threshold_decisions_equal') is not True):
        raise RuntimeError('Complete8693/4277/17 exact-reload export required.')
    if (any(v.get('version')!=VERSION for v in [manifest,selection,protocol,ready])
            or manifest.get('proposal')!=PROPOSAL or ready.get('proposal')!=PROPOSAL
            or manifest.get('threshold')!=.5 or selection.get('threshold')!=.5
            or selection.get('fixed_optimizer_steps')!=15 or selection.get('inner_selection_performed') is not False):
        raise RuntimeError('Version/fixed-recipe mismatch.')
    for meta in (fit,reload):
        for name,value in dict(status='fit_complete',candidate=VERSION,seed=20261112,training_rows=8693,
                feature_count=17,roster_rows=12970,fixed_steps=15,refit_optimizer_steps=15,
                inner_selection_performed=False,query_chunk=2318).items():
            if meta.get(name)!=value:raise RuntimeError('Full fit/reload recipe mismatch: '+name)
        if len(meta['columns'])!=17 or meta['columns'][-2:]!=['PeerCryoSleepRate','PeerKnownCryoCount']:
            raise RuntimeError('Wrong17-column recipe.')
    if fit!=reload:raise RuntimeError('Full portable model metadata changed on reload.')
    if (policy.get('allow_full_export') is not True or policy.get('allow_direct_submit') is not False
            or policy.get('maximum_upload_attempts_authorized_now')!=0 or policy.get('versions')!=[VERSION]
            or manifest.get('policy_path')!=EXPORT_POLICY or workflow.sha256(ROOT/EXPORT_POLICY)!=manifest['policy_sha256']):
        raise RuntimeError('Original export-only policy must remain unchanged; only new release grants one attempt.')
    for name,digest in manifest['source_hashes'].items():
        if workflow.sha256(ROOT/name)!=digest:raise RuntimeError('Frozen source changed: '+name)
    if (protocol['source_hashes']!=manifest['source_hashes'] or protocol['source_commit']!=manifest['source_commit']
            or protocol['environment_sha256']!=manifest['environment_sha256']
            or workflow.sha256(out/'environment.lock.txt')!=manifest['environment_sha256']
            or fit['checkpoint_sha256']!=protocol['checkpoint_sha256']
            or workflow.sha256(out/'model.joblib')!=manifest['model_sha256']
            or ready['model_sha256']!=manifest['model_sha256'] or ready['reload_max_absolute_difference']!=0):
        raise RuntimeError('Model/checkpoint/environment identity mismatch.')
    check=workflow.validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    if len({check['sha256'],manifest['sha256'],release['csv_sha256'],ready['csv_sha256']})!=1:
        raise RuntimeError('Released CSV identity differs.')
    with (out/'test_probabilities.csv').open(newline='') as stream:probs=list(csv.DictReader(stream))
    with (out/'submission.csv').open(newline='') as stream:submission=list(csv.DictReader(stream))
    if len(probs)!=4277:raise RuntimeError('Complete probability/reload table required.')
    for p,s in zip(probs,submission):
        v,w=float(p['probability']),float(p['reload_probability'])
        if (p['PassengerId']!=s['PassengerId'] or not math.isfinite(v) or not 0<=v<=1
                or v!=w or s['Transported']!=str(v>=.5)):
            raise RuntimeError('Probability/reload/threshold/PassengerId mismatch.')
    for attempt in (ROOT/'reports').rglob('online_attempts/*.json'):
        if read(attempt).get('file_sha256')==check['sha256']:
            raise RuntimeError('CSV already attempted; reconcile status, never repeat upload.')
    return check


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    lock=ROOT/'artifacts/online_iteration_20261117/submission.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check=verify()
        if not args.submit:
            print(json.dumps(dict(status='offline_release_verified',version=VERSION,**check)));return
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Commit completed changes before upload.')
        for path in [RELEASE,Path(__file__).resolve()]:
            committed=subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
            if hashlib.sha256(committed).hexdigest()!=workflow.sha256(path):raise RuntimeError('Uncommitted release/entrypoint.')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        limits=json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        if limits['numAllowedNow']<1:raise RuntimeError('No current primary-account submission allowance.')
        print(json.dumps({'fresh_daily_limits':limits}),flush=True)
        message=f"v99 同行休眠17列：按今日独立释放授权探索提交一次，当前可用 {limits['numAllowedNow']} 次。"
        subprocess.Popen(['osascript',str(ROOT/'remind_online_first_submission.applescript'),message],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        verify()
        workflow.submit(PROPOSAL,120)


if __name__=='__main__':main()
