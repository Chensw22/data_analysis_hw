"""Release exactly one completed fixed K256/one-view retrieval submission."""
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
PROPOSAL='proposal_nov18_retrieval256'
VERSION='v104_TabDPT_numpy_retrieval256_oneview'
RELEASE=ROOT/'reports/error_followup_20261118/retrieval_online_release.json'
EXPORT_POLICY='reports/error_followup_20261118/retrieval_export_policy.json'
PREDICT=dict(n_ensembles=1,temperature=1.,context_size=256,batch_size=8,permute_classes=True,seed=20261112)
CHECKPOINT='artifacts/pretrained/tabdpt-v1.3/tabdpt1_3.safetensors'


def read(path):return json.loads(Path(path).read_text())


def required_bound_files():
    out=ROOT/'artifacts'/PROPOSAL;manifest=read(out/'manifest.json')
    if manifest.get('status')!='complete':raise RuntimeError('Completed export required before binding release.')
    paths=set(manifest['source_hashes'])
    paths.update([Path(__file__).name,'kaggle_workflow.py','remind_online_first_submission.applescript',
        'prepare_retrieval_nov18.py',EXPORT_POLICY,CHECKPOINT])
    for prefix in ['artifacts','reports']:
        paths.update(f'{prefix}/{PROPOSAL}/{name}' for name in ['manifest.json','protocol.json','selection.json'])
    paths.update(f'artifacts/{PROPOSAL}/{name}' for name in ['model.joblib','probabilities.npz',
        'test_probabilities.csv','submission.csv','worker_result.json'])
    return {name:workflow.sha256(ROOT/name) for name in sorted(paths)}


def verify():
    release=read(RELEASE)
    if (release.get('allow_one_exploratory_submission') is not True
            or release.get('maximum_upload_attempts_for_this_file')!=1
            or release.get('proposal')!=PROPOSAL or release.get('version')!=VERSION):
        raise RuntimeError('Exact one-file exploratory release required.')
    if release.get('bound_files')!=required_bound_files():raise RuntimeError('Immutable release files changed.')
    out,public=ROOT/'artifacts'/PROPOSAL,ROOT/'reports'/PROPOSAL
    for name in ['manifest.json','protocol.json','selection.json']:
        if workflow.sha256(out/name)!=workflow.sha256(public/name):raise RuntimeError('Public/local mismatch: '+name)
    manifest=read(out/'manifest.json');protocol=read(out/'protocol.json');selection=read(out/'selection.json')
    worker=read(out/'worker_result.json');policy=read(ROOT/EXPORT_POLICY)
    if any(value.get('version')!=VERSION for value in [manifest,protocol,selection,worker,policy]):
        raise RuntimeError('Version identities differ.')
    for record in [manifest,worker]:
        if (record.get('status')!='complete' or record.get('training_rows')!=8693
                or record.get('rows')!=4277 or record.get('feature_count')!=13 or record.get('K')!=256
                or record.get('predict')!=PREDICT or record.get('portable_cpu_state') is not True
                or record.get('export_reload_inference_verified') is not True or record.get('reload_rows')!=4277
                or record.get('reload_threshold_decisions_equal') is not True
                or not 0<=record.get('reload_max_absolute_difference',float('inf'))<=2e-6):
            raise RuntimeError('Incomplete/mismatched full retrieval and reload recipe.')
    for key,value in worker.items():
        if manifest.get(key)!=value:raise RuntimeError('Completed worker/manifest mismatch: '+key)
    if (manifest.get('proposal')!=PROPOSAL or manifest.get('fit_rows')!=8693
            or manifest.get('threshold')!=.5 or selection.get('threshold')!=.5
            or policy.get('proposal')!=PROPOSAL or policy.get('predict')!=PREDICT
            or policy.get('allow_full_export') is not True or policy.get('allow_direct_submit') is not False
            or policy.get('budget_seconds')!=1800 or protocol.get('budget_seconds')!=1800
            or manifest.get('policy_path')!=EXPORT_POLICY or protocol.get('policy_path')!=EXPORT_POLICY
            or manifest.get('policy_sha256')!=workflow.sha256(ROOT/EXPORT_POLICY)
            or protocol.get('policy_sha256')!=manifest['policy_sha256']):
        raise RuntimeError('Original export-only policy/threshold changed; release must be separate.')
    if (protocol['source_hashes']!=manifest['source_hashes'] or protocol['source_commit']!=manifest['source_commit']
            or protocol['environment_sha256']!=manifest['environment_sha256']
            or policy['environment_sha256']!=manifest['environment_sha256']
            or workflow.sha256(ROOT/CHECKPOINT)!=manifest['checkpoint_sha256']):
        raise RuntimeError('Source/environment/checkpoint provenance mismatch.')
    for name,digest in {**manifest['source_hashes'],**policy['evidence_sha256']}.items():
        if workflow.sha256(ROOT/name)!=digest:raise RuntimeError('Frozen source/evidence changed: '+name)
    for name,key in [('model.joblib','model_sha256'),('probabilities.npz','probabilities_sha256'),
            ('test_probabilities.csv','test_probabilities_sha256')]:
        if workflow.sha256(out/name)!=manifest[key]:raise RuntimeError('Export artifact changed: '+name)
    first_record,second_record=manifest['first_inference'],manifest['reload_inference']
    if (first_record['neighbor_sha256']!=second_record['neighbor_sha256']
            or first_record['calls']!=second_record['calls']):raise RuntimeError('Reload neighbor identity changed.')
    for record in [first_record,second_record]:
        if (record['support_pool']!=8693 or record['query_rows']!=4277 or record['native_faiss_search'] is not False
                or sum(c['batch'] for c in record['calls'])!=4277
                or any(not 1<=c['batch']<=8 or c['tokens']!=257 or c['labels']!=256 for c in record['calls'])):
            raise RuntimeError('Observed forward shapes/context/coverage differ.')
    check=workflow.validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    if len({check['sha256'],manifest['sha256'],release['csv_sha256']})!=1:raise RuntimeError('Released CSV differs.')
    with (out/'test_probabilities.csv').open(newline='') as stream:probabilities=list(csv.DictReader(stream))
    with (out/'submission.csv').open(newline='') as stream:submission=list(csv.DictReader(stream))
    if len(probabilities)!=4277:raise RuntimeError('Incomplete probability table.')
    for p,s in zip(probabilities,submission):
        a,b=float(p['probability']),float(p['reload_probability'])
        if (p['PassengerId']!=s['PassengerId'] or not math.isfinite(a) or not math.isfinite(b)
                or not 0<=a<=1 or not 0<=b<=1 or abs(a-b)>2e-6
                or (a>=.5)!=(b>=.5) or s['Transported']!=str(a>=.5)):
            raise RuntimeError('Probability/reload/CSV classification mismatch.')
    import numpy as np
    with np.load(out/'probabilities.npz',allow_pickle=False) as cache:
        a,b=cache['prediction'],cache['reload']
        if a.shape!=(4277,2) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError('Invalid complete probability matrices.')
        if np.any((a<0)|(a>1)|(b<0)|(b>1)):raise RuntimeError('Probability outside[0,1].')
        np.testing.assert_allclose(a.sum(1),1,rtol=0,atol=2e-6)
        np.testing.assert_allclose(b.sum(1),1,rtol=0,atol=2e-6)
        difference=float(np.max(np.abs(a-b)))
        if difference!=manifest['reload_max_absolute_difference'] or difference>2e-6:
            raise RuntimeError('NPZ reload measurement differs.')
        np.testing.assert_array_equal(a[:,1]>=.5,b[:,1]>=.5)
        np.testing.assert_array_equal(cache['train_ids'],np.arange(8693))
        np.testing.assert_array_equal(cache['query_ids'],np.arange(8693,12970))
        np.testing.assert_array_equal(cache['passenger_ids'],np.asarray([r['PassengerId'] for r in submission],dtype=str))
        # CSV float32 decimal text is restored to its saved native dtype for exact identity.
        np.testing.assert_array_equal(a[:,1],np.asarray([float(r['probability']) for r in probabilities],dtype=a.dtype))
        np.testing.assert_array_equal(b[:,1],np.asarray([float(r['reload_probability']) for r in probabilities],dtype=b.dtype))
    for attempt in (ROOT/'reports').rglob('online_attempts/*.json'):
        if read(attempt).get('file_sha256')==check['sha256']:
            raise RuntimeError('Existing upload attempt: reconcile status, do not retry.')
    # No model unpickling or inference; portableCPU support identity was verified
    # by the immutable completed exporter and is bound by model/source hashes.
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
            raise RuntimeError('Commit completed source/release/evidence before uploading.')
        for path in [RELEASE,Path(__file__).resolve()]:
            committed=subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)
            if hashlib.sha256(committed).hexdigest()!=workflow.sha256(path):raise RuntimeError('Uncommitted release or entrypoint.')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        limits=json.loads(str(api.competition_get_submission_limits(workflow.COMPETITION)))
        if limits['numAllowedNow']<1:raise RuntimeError('No currently available primary-account submissions.')
        print(json.dumps({'fresh_daily_limits':limits}),flush=True)
        message=f"v104 固定K256单视图检索：按独立授权探索提交一次，当前可用 {limits['numAllowedNow']} 次。"
        subprocess.Popen(['osascript',str(ROOT/'remind_online_first_submission.applescript'),message],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        verify()
        workflow.submit(PROPOSAL,120)


if __name__=='__main__':main()
