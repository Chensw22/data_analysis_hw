"""One resource-qualified, fixed fold0 K256/one-view retrieval diagnostic; no export."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import tabdpt_numpy_retrieval_nov18 as retrieval

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'reports/error_followup_20261118/retrieval_diagnostic_protocol.json'
OUT=ROOT/'artifacts/error_followup_20261118/retrieval_fold0'
REPORT=ROOT/'reports/error_followup_20261118/retrieval_fold0'
BUDGET=1200
sha,read,write=retrieval.sha,retrieval.read,retrieval.write
SOURCES=['numpy_retrieval_diagnostic_nov18.py','tabdpt_numpy_retrieval_nov18.py',
         'error_overlap_diagnostic.py','exaone_validation_data.py','audit_top5_references.py',
         'three_family_mean_screen.py','prior_adapt_mean_screen.py']


def committed(path):
    name=str(Path(path).relative_to(ROOT))
    if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest()!=sha(path):
        raise RuntimeError('Commit evidence/source first: '+name)


def resource_gate():
    # Pure JSON/hash checks before any labels, model construction or numeric imports.
    files={};local=retrieval.OUT;public=retrieval.REPORT
    for name in ['result.json','freeze.json']:
        a=local/name;b=public/name;committed(b)
        if sha(a)!=sha(b):raise RuntimeError('Local/public resource evidence mismatch.')
        files.update({str(p.relative_to(ROOT)):sha(p) for p in [a,b]})
    result=read(local/'result.json');freeze=read(local/'freeze.json')
    if (result.get('status')!='resource_complete' or result.get('qualified_for_root_consideration') is not True
            or result.get('cost_passed') is not True or result.get('cv_authorized') is not False
            or result.get('elapsed_seconds',float('inf'))>600
            or result.get('projected_fold2318_two_predictions_seconds',float('inf'))>1200
            or result.get('real_data_read') is not False or result.get('accuracy_computed') is not False
            or result.get('freeze_sha256')!=sha(local/'freeze.json')):
        raise RuntimeError('Complete bounded resource qualification required.')
    for name,limit in [('cpu_mps',2e-5),('reload',2e-6),('reverse',2e-6),('subset19',2e-6)]:
        value=result.get('stages',{}).get(name)
        if not isinstance(value,(int,float)) or not 0<=value<=limit:raise RuntimeError('Resource numerical check invalid.')
    for name,key in [('probabilities.npz','probabilities_sha256'),('model.joblib','model_sha256')]:
        p=local/name
        if sha(p)!=result[key]:raise RuntimeError('Resource artifact changed.')
        files[str(p.relative_to(ROOT))]=sha(p)
    for name,digest in freeze['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Resource source changed: '+name)
    weight=ROOT/freeze['checkpoint_path']
    if sha(weight)!=freeze['checkpoint_sha256']:raise RuntimeError('Checkpoint changed.')
    environment='\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n'
    if hashlib.sha256(environment.encode()).hexdigest()!=freeze['environment_sha256']:raise RuntimeError('Resource environment differs.')
    if freeze['plan']['predict']!=retrieval.PREDICT or freeze['plan']['K']!=256:raise RuntimeError('Resource recipe mismatch.')
    files.update(freeze['source_hashes']);files[str(weight.relative_to(ROOT))]=sha(weight)
    return files,freeze['environment_sha256']


def check_frozen():
    frozen=read(OUT/'freeze.json')
    for name,digest in frozen['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen evidence changed: '+name)
    return frozen


def score(p,y):
    import numpy as np
    from sklearn.metrics import log_loss
    return dict(accuracy=float(np.mean((p>=.5)==y)),log_loss=float(log_loss(y,p,labels=[0,1])))


def paired(p,ref,y,mask):
    import numpy as np
    a=(p>=.5)==y;b=(ref>=.5)==y
    return dict(rows=int(mask.sum()),repaired=int((a&~b&mask).sum()),broken=int((~a&b&mask).sum()),
                net_correct=int(((a.astype(int)-b.astype(int))[mask]).sum()),candidate_correct=int((a&mask).sum()))


def confusion(p,y,mask):
    return dict(tn=int((mask&(y==0)&(p<.5)).sum()),fp=int((mask&(y==0)&(p>=.5)).sum()),
                fn=int((mask&(y==1)&(p<.5)).sum()),tp=int((mask&(y==1)&(p>=.5)).sum()))


def worker():
    retrieval.require_false();resource_gate();frozen=check_frozen()
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import tabdpt_adapter as old
    import error_overlap_diagnostic as overlap
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required, no fallback.')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
    inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated()
    dev,y=inputs['dev'],inputs['y'];a,b=splits[0]
    if len(a)!=4636 or len(b)!=2318 or not np.all(fold_ids[b]==0):raise RuntimeError('Original fold0 required.')
    x=inputs['features'];train=x.iloc[dev[a]];query=x.iloc[dev[b]]
    encoder=old.CompactEncoder().fit(train)
    # Only train labels enter the portable model. Global IDs preserve support identity.
    model=retrieval.RetrievalPredictor(encoder,encoder.transform(train),y[a],dev[a],device='mps')
    np.savez_compressed(OUT/'rows.npz',train_ids=dev[a],query_ids=dev[b])
    write(OUT/'input_evidence.json',evidence)
    write(OUT/'progress.json',dict(stage='predict4636pool2318query',training_rows=len(a),query_rows=len(b)))
    prediction=model.predict_proba(query);first=model.last_record
    joblib.dump(model,OUT/'model.joblib');del model;old.upstream.release(None,'mps')
    write(OUT/'progress.json',dict(stage='release_reload2318query',first_prediction=first))
    model=joblib.load(OUT/'model.joblib')
    if model.device!='cpu':raise RuntimeError('Portable default must be CPU.')
    np.testing.assert_array_equal(model.labels,y[a]);np.testing.assert_array_equal(model.ids,dev[a])
    np.testing.assert_array_equal(model.support,model.encoder.transform(train))
    model.device='mps';reload=model.predict_proba(query);second=model.last_record
    difference=retrieval.comparison(prediction,reload,2e-6)
    del model;old.upstream.release(None,'mps')
    # First access to outcome-defined historical case masks, after both complete inferences.
    rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
    for col,expected in [('row_id',dev),('fold',fold_ids),('y',y),('PassengerId',inputs['passengers'])]:
        np.testing.assert_array_equal(rows[col].to_numpy(),expected)
    masks={'all':np.ones(len(b),bool),'hard546':rows.all_recipes_still_wrong.to_numpy(bool)[b],
           'hard914':(rows.family_wrong_count.to_numpy()==5)[b],
           'common_correct5263':(rows.family_wrong_count.to_numpy()==0)[b]}
    if masks['hard546'].sum()!=178 or masks['hard914'].sum()!=292:raise RuntimeError('Historical fold0 masks changed.')
    masks['Earth_sleep']=(query.HomePlanet.eq('Earth')&query.CryoSleep.eq('True')).to_numpy()
    masks['Earth_awake']=(query.HomePlanet.eq('Earth')&query.CryoSleep.eq('False')).to_numpy()
    probabilities={'retrieval256_oneview':prediction[:,1],'rawMitra':bases['rawMitra'][b], 'TabDPT_full8':bases['TabDPT'][b]}
    p=prediction[:,1];target=y[b]
    subsets={name:dict(rows=int(mask.sum()),paired={k:paired(p,ref,target,mask) for k,ref in probabilities.items() if k!='retrieval256_oneview'},
        confusion={k:confusion(ref,target,mask) for k,ref in probabilities.items()}) for name,mask in masks.items()}
    np.savez_compressed(OUT/'predictions.npz',dev_ids=dev[b],fold_id=fold_ids[b],passenger_ids=np.asarray(inputs['passengers'][b],dtype=str),
        prediction=prediction,reload=reload,**probabilities,**masks)
    for name,digest in evidence.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Reference evidence changed.')
    check_frozen();retrieval.require_false()
    write(OUT/'worker_result.json',dict(status='complete',fold=0,training_rows=4636,query_rows=2318,
        metrics={k:score(v,target) for k,v in probabilities.items()},subsets=subsets,
        first_inference=first,reload_inference=second,reload_max_abs_difference=difference,reload_threshold_equal=True,
        model_sha256=sha(OUT/'model.joblib'),predictions_sha256=sha(OUT/'predictions.npz'),rows_sha256=sha(OUT/'rows.npz'),
        input_evidence_sha256=sha(OUT/'input_evidence.json'),no_full_export=True,no_upload=True,
        disclosure='One adaptive original fold0 diagnostic. rawMitra and TabDPT-full8 are descriptive references, not matched one-view causal controls. No query target enters encoder/context/retrieval/model.'))


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('Fresh output required; no resume/overwrite.')
    plan=read(PLAN)
    if plan['budget_seconds']!=BUDGET or plan['predict']!=retrieval.PREDICT:raise RuntimeError('Frozen diagnostic changed.')
    # Resource and source qualifications precede both labels and creation of this run.
    resource_files,envhash=resource_gate()
    names=[*SOURCES,str(PLAN.relative_to(ROOT))]
    for name in names:committed(ROOT/name)
    evidence={**resource_files,**{n:sha(ROOT/n) for n in names},**plan['pinned_evidence_sha256']}
    for name,digest in evidence.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Pinned input changed: '+name)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    start=time.monotonic();deadline=start+BUDGET;child=None
    try:
        frozen=dict(protocol=plan,evidence_sha256=evidence,environment_sha256=envhash,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(OUT/'freeze.json',frozen);write(REPORT/'freeze.json',frozen)
        with (OUT/'worker.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait(timeout=max(.01,deadline-time.monotonic()))
        if code:raise RuntimeError(f'Worker failed with exit{code}; see preserved native log.')
        result=read(OUT/'worker_result.json')
        if result.get('status')!='complete' or time.monotonic()>deadline:raise RuntimeError('Incomplete/late fold.')
        check_frozen()
        for name,key in [('model.joblib','model_sha256'),('predictions.npz','predictions_sha256'),('rows.npz','rows_sha256'),('input_evidence.json','input_evidence_sha256')]:
            if sha(OUT/name)!=result[key]:raise RuntimeError('Output changed: '+name)
        result.update(status='complete_single_fold_adaptive_diagnostic',elapsed_seconds=time.monotonic()-start,freeze_sha256=sha(OUT/'freeze.json'))
        write(OUT/'result.json',result);write(REPORT/'result.json',result)
    except BaseException as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        failure=dict(status='failed_incomplete_not_candidate',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-start)
        if (OUT/'worker_failure.json').exists():failure['worker_failure']=read(OUT/'worker_failure.json')
        for directory in [OUT,REPORT]:
            (directory/'result.json').unlink(missing_ok=True);write(directory/'failure.json',failure)
        if (OUT/'predictions.npz').exists():(OUT/'predictions.npz').rename(OUT/'predictions.partial.npz')
        raise


if __name__=='__main__':
    retrieval.runtime_entry()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:
        try:worker()
        except BaseException as error:write(OUT/'worker_failure.json',dict(error_type=type(error).__name__,error=str(error)));raise
    else:run()
