"""Synthetic CPU/MPS comparison for pinned LimiX-2, default32 pipelines only."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,subprocess,signal,resource,gc
from pathlib import Path
from limix2_setup import ROOT,WORK,REPORT,sha,write
from limix2_cpu_probe import verify
OUT=REPORT/'device_tiny'


def worker():
    started=time.monotonic();result={'status':'started','real_data_read':False,'MPS_fraction':.65,'tolerance':2e-5}
    try:
        weight,manifest=verify()
        import torch
        import numpy as np
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback.')
        torch.mps.set_per_process_memory_fraction(.65)
        if torch.serialization.get_unsafe_globals_in_checkpoint(str(weight)):raise RuntimeError('Unsafe checkpoint globals.')
        ckpt=torch.load(str(weight),map_location='cpu',weights_only=True)
        import limix2_cpu_compat as compat
        result['norm_equivalence']=compat.check_native_equivalence();compat.install()
        sys.path.insert(0,str(WORK/'source'))
        from inference.predictor import LimiXPredictor
        config=json.loads((WORK/'source/config/cls_default_noretrieval_v2.json').read_text())
        rng=np.random.default_rng(20261118);x=rng.normal(size=(36,13)).astype(np.float32);y=np.tile(np.array([0,1]),16);x[1,4]=np.nan;x[7,10]=np.nan
        predictions={};times={}
        for device in ['cpu','mps']:
            print('initializing',device,flush=True)
            model=LimiXPredictor(device=torch.device(device),model_path=str(weight),ckpt=ckpt,inference_config=config,
                mix_precision=False,seed=20261118,test_batch_mode='fixed',test_batch_size=4,inference_with_DDP=False,use_data_cache=False)
            before=time.monotonic();p=np.asarray(model.predict(x[:32],y,x[32:],task_type='Classification'))
            if device=='mps':torch.mps.synchronize()
            times[device]=time.monotonic()-before
            if p.shape!=(4,2) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):raise RuntimeError('Invalid probabilities.')
            predictions[device]=p.copy()
            result[device+'_parameter_devices']=sorted(set(str(t.device) for t in model.model.parameters()))
            if device=='mps':result['mps_current_bytes']=torch.mps.current_allocated_memory();result['mps_driver_bytes']=torch.mps.driver_allocated_memory()
            print('completed',device,times[device],flush=True)
            del model;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        diff=float(np.abs(predictions['cpu']-predictions['mps']).max());same=bool(np.array_equal(predictions['cpu'].argmax(1),predictions['mps'].argmax(1)))
        np.savez(OUT/'probabilities.npz',**predictions)
        result.update(max_absolute_difference=diff,decisions_equal=same,prediction_seconds=times,pipelines=len(config['pipelines']),support_rows=32,query_rows=4,features=13,weights_only=True,no_accuracy_computed=True)
        if diff>2e-5 or not same:raise RuntimeError('Fixed CPU/MPS tolerance or decisions check failed.')
        if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Runtime flag changed.')
        result['status']='cpu_mps_tiny_complete'
    except BaseException as e:result.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        result.update(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(OUT/'result.json',result)


def main():
    if sys.argv[1:]==['--worker']:return worker()
    if sys.argv[1:]:raise ValueError('No CLI options.')
    if OUT.exists():raise FileExistsError('Fresh independent result directory required.')
    OUT.mkdir(parents=True)
    freeze={'sources':{n:sha(ROOT/n) for n in ['limix2_device_probe.py','limix2_cpu_probe.py','limix2_cpu_compat.py','limix2_setup.py']},'source_manifest':sha(REPORT/'download.json'),'environment':sha(REPORT/'environment.lock.txt'),'budget_seconds':300,'full_context_claim':False}
    write(OUT/'freeze.json',freeze)
    with (OUT/'native.log').open('w') as f:
        child=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();write(OUT/'result.json',{'status':'failed_timeout','budget_seconds':300});raise
    for n,d in freeze['sources'].items():
        if sha(ROOT/n)!=d:raise RuntimeError('Source modified during probe.')
    if code:raise RuntimeError('Device probe failed; preserve native.log/result.json.')
    print((OUT/'result.json').read_text())

if __name__=='__main__':main()
