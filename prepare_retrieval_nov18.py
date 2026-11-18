"""Conditional full K256/one-view export; requires a separately committed decision."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import tabdpt_numpy_retrieval_nov18 as retrieval
import numpy_retrieval_diagnostic_nov18 as diagnostic

ROOT=Path(__file__).resolve().parent
NAME='v104_TabDPT_numpy_retrieval256_oneview'
PROPOSAL='proposal_nov18_retrieval256'
POLICY=ROOT/'reports/error_followup_20261118/retrieval_export_policy.json'
OUT=ROOT/'artifacts'/PROPOSAL
REPORT=ROOT/'reports'/PROPOSAL
BUDGET=1800
sha,read,write=retrieval.sha,retrieval.read,retrieval.write


def required_evidence():
    """Read-only policy-building helper; refuses incomplete diagnostic/probe."""
    evidence,envhash=diagnostic.resource_gate()
    for name in ['result.json','freeze.json']:
        local=diagnostic.OUT/name;public=diagnostic.REPORT/name
        diagnostic.committed(public)
        if sha(local)!=sha(public):raise RuntimeError('Diagnostic local/public mismatch.')
        evidence.update({str(p.relative_to(ROOT)):sha(p) for p in [local,public]})
    result=read(diagnostic.OUT/'result.json');frozen=read(diagnostic.OUT/'freeze.json')
    if (result.get('status')!='complete_single_fold_adaptive_diagnostic' or result.get('fold')!=0
            or result.get('training_rows')!=4636 or result.get('query_rows')!=2318
            or result.get('reload_threshold_equal') is not True
            or not 0<=result.get('reload_max_abs_difference',float('inf'))<=2e-6
            or result.get('freeze_sha256')!=sha(diagnostic.OUT/'freeze.json')
            or frozen['environment_sha256']!=envhash):
        raise RuntimeError('Complete exact-recipe diagnostic/reload required.')
    evidence.update(frozen['evidence_sha256'])
    for name,key in [('model.joblib','model_sha256'),('predictions.npz','predictions_sha256'),
            ('rows.npz','rows_sha256'),('input_evidence.json','input_evidence_sha256')]:
        path=diagnostic.OUT/name
        if sha(path)!=result[key]:raise RuntimeError('Diagnostic output changed: '+name)
        evidence[str(path.relative_to(ROOT))]=sha(path)
    for name,digest in evidence.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Qualification evidence changed: '+name)
    return evidence,envhash


def check_frozen():
    frozen=read(OUT/'protocol.json')
    for name,digest in frozen['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Export source/evidence changed: '+name)
    return frozen


def worker():
    retrieval.require_false();frozen=check_frozen()
    # Only metadata/hash checks above; numeric imports and full labels occur below.
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import tabdpt_adapter as old
    import exaone_validation_data as data
    import kaggle_workflow as workflow
    if hashlib.sha256(old.upstream.environment().encode()).hexdigest()!=frozen['environment_sha256']:
        raise RuntimeError('Original retrieval environment required.')
    old.verify_sources();old.upstream.verify_checkpoint()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
    train=pd.read_csv(ROOT/'data/raw/train.csv')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda name:name!='Transported')
    if len(train)!=8693 or len(test)!=4277:raise RuntimeError('Complete8693/4277 required.')
    frame=data.compact_features(pd.concat([train.drop(columns='Transported'),test],ignore_index=True))
    x,q=frame.iloc[:8693],frame.iloc[8693:]
    y=old.binary_labels(train.Transported.astype(np.int64).to_numpy(),8693)
    ids=np.arange(8693,dtype=np.int64);test_ids=test.PassengerId.astype(str).to_numpy()
    model=None
    try:
        encoder=old.CompactEncoder().fit(x)
        model=retrieval.RetrievalPredictor(encoder,encoder.transform(x),y,ids,device='mps')
        if model.support.shape!=(8693,13):raise RuntimeError('Original13 encoded inputs required.')
        write(OUT/'progress.json',dict(stage='full8693_pool_4277_query',predict=retrieval.PREDICT))
        first=model.predict_proba(q);first_record=model.last_record
        joblib.dump(model,OUT/'model.joblib');model=None;old.upstream.release(None,'mps')
        write(OUT/'progress.json',dict(stage='full4277_reload',first_inference=first_record))
        model=joblib.load(OUT/'model.joblib')
        if model.device!='cpu' or model.support.shape!=(8693,13):raise RuntimeError('PortableCPU full context required.')
        np.testing.assert_array_equal(model.ids,ids);np.testing.assert_array_equal(model.labels,y)
        np.testing.assert_array_equal(model.support,model.encoder.transform(x))
        model.device='mps';again=model.predict_proba(q);second_record=model.last_record
        difference=retrieval.comparison(first,again,2e-6)
        if first_record['neighbor_sha256']!=second_record['neighbor_sha256']:
            raise RuntimeError('Complete reload neighbor identities changed.')
        for record in [first_record,second_record]:
            if record['support_pool']!=8693 or record['query_rows']!=4277 or record['native_faiss_search'] is not False:
                raise RuntimeError('Fixed fullpool NumPy retrieval changed.')
        model=None;old.upstream.release(None,'mps')
        pd.DataFrame(dict(PassengerId=test_ids,probability=first[:,1],reload_probability=again[:,1])).to_csv(OUT/'test_probabilities.csv',index=False)
        pd.DataFrame(dict(PassengerId=test_ids,Transported=first[:,1]>=.5)).to_csv(OUT/'submission.csv',index=False)
        check=workflow.validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')
        for path in (ROOT/'reports').rglob('online_attempts/*.json'):
            if read(path).get('file_sha256')==check['sha256']:raise RuntimeError('CSV already attempted; no duplicate.')
        np.savez_compressed(OUT/'probabilities.npz',prediction=first,reload=again,train_ids=ids,
            passenger_ids=np.asarray(test_ids,dtype=str),query_ids=q.index.to_numpy(dtype=np.int64))
        check_frozen();retrieval.require_false()
        if hashlib.sha256(old.upstream.environment().encode()).hexdigest()!=frozen['environment_sha256']:
            raise RuntimeError('Environment changed.')
        metadata=dict(status='complete',version=NAME,training_rows=8693,rows=4277,feature_count=13,
            K=256,predict=dict(retrieval.PREDICT),checkpoint_sha256=old.WEIGHT_SHA,
            training_ids_sha256=old.array_sha(ids),training_labels_sha256=old.array_sha(y),
            first_inference=first_record,reload_inference=second_record,
            export_reload_inference_verified=True,reload_rows=4277,reload_max_absolute_difference=difference,
            reload_threshold_decisions_equal=True,portable_cpu_state=True,
            sha256=check['sha256'],model_sha256=sha(OUT/'model.joblib'),
            test_probabilities_sha256=sha(OUT/'test_probabilities.csv'),probabilities_sha256=sha(OUT/'probabilities.npz'))
        write(OUT/'worker_result.json',metadata)
    finally:
        model=None;old.upstream.release(None,'mps')


def export():
    if OUT.exists() or REPORT.exists():raise RuntimeError('Fresh output only; no retry/resume.')
    start=time.monotonic();deadline=start+BUDGET
    # Missing policy is a deliberate stop before labels/model construction.
    policy=read(POLICY);diagnostic.committed(POLICY)
    if (policy.get('version')!=NAME or policy.get('proposal')!=PROPOSAL
            or policy.get('allow_full_export') is not True or policy.get('allow_direct_submit') is not False
            or policy.get('predict')!=retrieval.PREDICT or policy.get('budget_seconds')!=BUDGET):
        raise RuntimeError('Explicit fixed one-view full-export decision required.')
    evidence,envhash=required_evidence()
    if policy.get('environment_sha256')!=envhash or policy.get('evidence_sha256')!=evidence:
        raise RuntimeError('Policy must bind complete qualified diagnostic/probe/environment.')
    names=[Path(__file__).name,str(POLICY.relative_to(ROOT)),*diagnostic.SOURCES,
           'kaggle_workflow.py','exaone_validation_data.py']
    for name in names:diagnostic.committed(ROOT/name)
    sources={**evidence,**{name:sha(ROOT/name) for name in names}}
    for filename in ['train','test','sample_submission']:
        name=f'data/raw/{filename}.csv';sources[name]=sha(ROOT/name)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True);child=None
    try:
        protocol=dict(version=NAME,proposal=PROPOSAL,source_hashes=sources,environment_sha256=envhash,
            policy_path=str(POLICY.relative_to(ROOT)),policy_sha256=sha(POLICY),policy=policy,budget_seconds=BUDGET,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        for folder in [OUT,REPORT]:write(folder/'protocol.json',protocol)
        with (OUT/'worker.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait(timeout=max(.01,deadline-time.monotonic()))
        if code:raise RuntimeError(f'Full export worker exit{code}; inspect preserved worker.log.')
        result=read(OUT/'worker_result.json')
        if result['status']!='complete' or time.monotonic()>deadline:raise RuntimeError('Incomplete/late full export.')
        check_frozen()
        for name,key in [('model.joblib','model_sha256'),('submission.csv','sha256'),
            ('test_probabilities.csv','test_probabilities_sha256'),('probabilities.npz','probabilities_sha256')]:
            if sha(OUT/name)!=result[key]:raise RuntimeError('Worker artifact changed.')
        selection=dict(version=NAME,components=[NAME],threshold=.5,stage='prepared_pending_separate_release',
            A_eligible=None,S_supported=False,automatic_upload_authorized=False)
        manifest=dict(**result,proposal=PROPOSAL,threshold=.5,fit_rows=8693,valid=True,
            source_commit=protocol['source_commit'],source_hashes=sources,environment_sha256=envhash,
            policy_path=protocol['policy_path'],policy_sha256=protocol['policy_sha256'],
            selection=selection,seconds=time.monotonic()-start,no_new_local_score=True)
        for name,value in [('selection.json',selection),('manifest.json',manifest)]:
            for folder in [OUT,REPORT]:write(folder/name,value)
        print(__import__('json').dumps(dict(status='complete_pending_release',proposal=PROPOSAL,sha256=result['sha256'])),flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for folder in [OUT,REPORT]:
            for name in ['manifest.json','selection.json','submission.csv']:(folder/name).unlink(missing_ok=True)
            write(folder/'failure.json',dict(status='failed_not_upload_eligible',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    retrieval.runtime_entry()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:
        try:worker()
        except BaseException as error:
            write(OUT/'worker_failure.json',dict(error_type=type(error).__name__,error=str(error)));raise
    else:export()
