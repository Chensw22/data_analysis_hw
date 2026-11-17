"""One 30-second synthetic runtime check with duplicate-OpenMP tolerance disabled."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import iltm_r128bn_full8_probe as resource

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/probes/iltm_false_runtime'
REPORT=ROOT/'reports/iltm_false_runtime'
FLAG='KMP_DUPLICATE_LIB_OK'


def worker():
    record={'status':'incomplete','no_weights_or_real_data':True,'stages':[]}
    def stage(name):
        if os.environ.get(FLAG)!='FALSE':raise RuntimeError('Explicit FALSE must persist: '+name)
        record['stages'].append({'name':name,'duplicate_runtime_tolerance':'FALSE'})
        resource.write(OUT/'partial.json',record)
    try:
        stage('before_numeric_imports')
        import threadpoolctl
        stage('after_threadpoolctl')
        import numpy as np
        import pandas as pd
        import torch
        import sklearn
        stage('after_numpy_pandas_torch_sklearn')
        sys.path.insert(0,str(resource.VENDOR))
        import iltm
        from iltm.realmlp_td_s_preprocessing import get_realmlp_td_s_pipeline_separated
        stage('after_complete_native_iltm_import')
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        with threadpoolctl.threadpool_limits(limits=4):
            torch.manual_seed(20261117)
            a=torch.randn(512,512,requires_grad=True);b=torch.randn(512,512)
            z=(a@b).square().mean();z.backward()
            if not torch.isfinite(a.grad).all():raise RuntimeError('Nonfinite torch gradient.')
            stage('after_torch_matmul_backward')
            rng=np.random.default_rng(20261117);x=rng.normal(size=(128,64))
            u,s,v=np.linalg.svd(x,full_matrices=False)
            error=float(np.max(np.abs((u*s)@v-x)))
            if error>1e-10:raise RuntimeError('SVD reconstruction differs.')
            record['svd_max_reconstruction_error']=error
            stage('after_numpy_reduced_svd')
            frame=pd.DataFrame(rng.normal(size=(128,13)))
            for col in resource.CATEGORICAL:frame[col]=rng.choice(['a','b','c'],size=len(frame))
            pre=get_realmlp_td_s_pipeline_separated(resource.CATEGORICAL)
            parts=pre.fit_transform(frame)
            if any(len(t)!=128 or not np.isfinite(t).all() for t in parts):raise RuntimeError('Preprocessing invalid.')
            again=pre.transform(frame)
            if any(not np.array_equal(a,b) for a,b in zip(parts,again)):raise RuntimeError('Train transform mismatch.')
            stage('after_native_preprocessing')
            if not torch.isfinite(a.detach()@b).all():raise RuntimeError('Second torch operation invalid.')
            stage('after_torch_again')
            record['openmp_runtimes']=[{k:v for k,v in item.items() if k in ['filepath','num_threads','prefix','internal_api']}
                                      for item in threadpoolctl.threadpool_info() if item.get('user_api')=='openmp']
        stage('finished');record['status']='complete'
    except BaseException as error:
        record.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error))
    resource.write(OUT/'worker_result.json',record)


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('No retry/overwrite.')
    # Snapshot before resource qualification imports numpy or any data helper.
    env=resource.child_environment();env[FLAG]='FALSE'
    resource.require_committed(['iltm_false_runtime_probe.py','reports/iltm_r128bn_import_boundary_diagnostic.json'])
    frozen=resource.contract()
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    resource.write(OUT/'freeze.json',{'resource_identity':frozen,'source_sha256':resource.sha(Path(__file__)),
        'budget_seconds':30,'explicit_duplicate_runtime_tolerance':'FALSE','no_candidate_fit':True})
    start=time.monotonic()
    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=env,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed_out=False
    try:stdout,stderr=process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate()
    (OUT/'stdout.log').write_text(stdout);(OUT/'stderr.log').write_text(stderr)
    record=resource.read(OUT/'worker_result.json') if (OUT/'worker_result.json').exists() else {}
    result={'status':'complete' if process.returncode==0 and record.get('status')=='complete' and not timed_out else 'failed_inconclusive',
            'exit_code':process.returncode,'timed_out':timed_out,'seconds':time.monotonic()-start,'worker':record,
            'no_candidate_scores':True,'no_complete_training_qualification':True,'no_automatic_cv':True,
            'failure_policy':'Close this fixed FALSE runtime route on failure; do not enable TRUE or search overrides.'}
    if (OUT/'partial.json').exists():result['last_progress']=resource.read(OUT/'partial.json')
    for path in [OUT/'result.json',REPORT/'result.json']:resource.write(path,result)
    resource.write(REPORT/'freeze.json',resource.read(OUT/'freeze.json'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    worker() if sys.argv[1:]==['--worker'] else run()
