"""Pinned weights-only inspection and CPU tiny smoke; never accesses project data/MPS."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse,hashlib,json,sys,time,subprocess,signal,resource
from pathlib import Path
from limix2_setup import ROOT,WORK,REPORT,WEIGHT_SHA,WEIGHT_SIZE,sha,write


def verify():
    d=json.loads((REPORT/'download.json').read_text());weight=ROOT/d['weight_path']
    if weight.stat().st_size!=WEIGHT_SIZE or sha(weight)!=WEIGHT_SHA:raise RuntimeError('Checkpoint identity differs.')
    for path,digest in d['source_files'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Source changed: '+path)
    return weight,d


def worker(mode):
    started=time.monotonic();record={'status':'started','mode':mode,'device':'cpu','real_data_read':False}
    try:
        weight,manifest=verify()
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(str(weight))
        record['unsafe_checkpoint_globals']=unsafe
        if unsafe:raise RuntimeError('Additional checkpoint types require specific review; no unrestricted pickle permitted.')
        checkpoint=torch.load(str(weight),map_location='cpu',weights_only=True)
        record['weights_only']=True;record['checkpoint_keys']=list(checkpoint)
        config=checkpoint['config'];state=checkpoint['state_dict']
        write(WORK/'checkpoint_config.json',config)
        record['config_sha256']=sha(WORK/'checkpoint_config.json')
        record['config_summary']={k:config.get(k) for k in ['nlayers','nhead','embed_dim','features_per_group','rmsnorm_impl','feature_attention_pipeline','separate_x_y_ffn','num_cls_tokens','tf_norm_type','dsti_attention_backend']}
        record['state_keys']=len(state)
        record['state_elements']=sum(t.numel() for t in state.values() if isinstance(t,torch.Tensor))
        record['tensor_bytes']=sum(t.numel()*t.element_size() for t in state.values() if isinstance(t,torch.Tensor))
        if mode=='inspect':record['status']='safe_checkpoint_inspected'
        else:
            import numpy as np
            import limix2_cpu_compat as compat
            record['norm_equivalence']=compat.check_native_equivalence();record['local_import_adapter']=compat.install()
            sys.path.insert(0,str(WORK/'source'))
            from inference.predictor import LimiXPredictor
            inference=json.loads((WORK/'source/config/cls_default_noretrieval_v2.json').read_text())
            record['pipelines']=len(inference['pipelines'])
            model=LimiXPredictor(device=torch.device('cpu'),model_path=str(weight),ckpt=checkpoint,
                inference_config=inference,mix_precision=False,seed=20261118,
                test_batch_mode='fixed',test_batch_size=4,inference_with_DDP=False,use_data_cache=False)
            # Native loader uses strict state dict loading. Check every actual tensor.
            actual=model.model.state_dict()
            if set(actual)!=set(state):raise RuntimeError('Native state keys differ.')
            if not all(torch.equal(actual[k],state[k]) for k in actual):raise RuntimeError('Loaded weights differ.')
            record['strict_loaded_state_equal']=True
            rng=np.random.default_rng(20261118);x=rng.normal(size=(36,13)).astype(np.float32);y=np.tile(np.array([0,1]),16)
            x[1,4]=np.nan;x[7,10]=np.nan
            before=time.monotonic();p=np.asarray(model.predict(x[:32],y,x[32:],task_type='Classification'))
            record['prediction_seconds']=time.monotonic()-before
            if p.shape!=(4,2) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6):raise RuntimeError('Invalid tiny probabilities.')
            if any(t.device.type!='cpu' for t in model.model.parameters()):raise RuntimeError('NonCPU model tensor.')
            np.savez(REPORT/'tiny_probabilities.npz',probabilities=p)
            record.update(status='cpu_tiny_complete',support_rows=32,query_rows=4,features=13,probability_sha256=sha(REPORT/'tiny_probabilities.npz'),model_parameters=sum(t.numel() for t in model.model.parameters()),no_accuracy_computed=True)
        record['runtime_KMP_DUPLICATE_LIB_OK']=os.environ.get('KMP_DUPLICATE_LIB_OK')
        if record['runtime_KMP_DUPLICATE_LIB_OK']!='FALSE':raise RuntimeError('Runtime flag mutated.')
    except BaseException as e:
        record.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        record.update(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(REPORT/(mode+'_result.json'),record)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inspect','smoke']);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:return worker(args.mode)
    dest=REPORT/(args.mode+'_result.json')
    if dest.exists():raise FileExistsError('No overwrite of prior result.')
    env={**os.environ,'KMP_DUPLICATE_LIB_OK':'FALSE','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    with (REPORT/(args.mode+'.log')).open('w') as out:
        child=subprocess.Popen([sys.executable,__file__,args.mode,'--worker'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait();write(dest,{'status':'failed_timeout','seconds':300,'mode':args.mode});raise
    if code:raise RuntimeError('CPU worker failed, see '+str(REPORT/(args.mode+'.log')))
    r=json.loads(dest.read_text());print(json.dumps({k:v for k,v in r.items() if k!='config'},indent=2))

if __name__=='__main__':main()
