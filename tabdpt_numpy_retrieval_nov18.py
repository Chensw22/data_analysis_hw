"""Synthetic-only TabDPT K256/one-view NumPy retrieval feasibility; no scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'reports/error_followup_20261118/retrieval_protocol.json'
OUT=ROOT/'artifacts/error_followup_20261118/numpy_retrieval_probe'
REPORT=ROOT/'reports/error_followup_20261118/numpy_retrieval_probe'
SEED=20261112
K=256
PREDICT=dict(n_ensembles=1,temperature=1.,context_size=256,batch_size=8,permute_classes=True,seed=SEED)


def runtime_entry():
    if any(n in sys.modules for n in ['numpy','torch','sklearn','faiss']):
        raise RuntimeError('Explicit FALSE entry must precede numeric imports.')
    os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='4'


def require_false():
    if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('KMP duplicate-runtime override forbidden.')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def exact_l2(support,queries,k=K):
    import numpy as np
    s=np.asarray(support,dtype=np.float64);q=np.asarray(queries,dtype=np.float64)
    if s.ndim!=2 or q.ndim!=2 or s.shape[1]!=q.shape[1] or len(s)<k or k<1:
        raise ValueError('Finite aligned numeric matrices with enough support required.')
    if not np.isfinite(s).all() or not np.isfinite(q).all():raise ValueError('Imputation/scaling must precede retrieval.')
    result=np.empty((len(q),k),dtype=np.int64)
    for start in range(0,len(q),8):
        diff=q[start:start+8,None,:]-s[None,:,:]
        distance=np.sum(diff*diff,axis=2,dtype=np.float64)
        if not np.isfinite(distance).all():raise ValueError('Nonfinite squared distances.')
        result[start:start+8]=np.argsort(distance,axis=1,kind='stable')[:,:k]
    return result


def check():
    import numpy as np
    s=np.array([[0,0],[0,0],[1,0],[-1,0],[3,0]],dtype=np.float32)
    q=np.array([[0,0],[2,0],[1,0]],dtype=np.float32)
    found=exact_l2(s,q,3)
    np.testing.assert_array_equal(found[0],[0,1,2])
    expected=np.array([sorted(range(len(s)),key=lambda i:(float(np.sum((s[i].astype(float)-x)**2)),i))[:3] for x in q])
    np.testing.assert_array_equal(found,expected)
    order=np.array([2,0,1]);np.testing.assert_array_equal(exact_l2(s,q[order],3),found[order])
    labels=np.array([0,1,0,1,0]);labels[:]=1-labels
    np.testing.assert_array_equal(exact_l2(s,q,3),found) # Function has no label argument/state.
    perm=np.array([4,2,0,3,1]);mapped=perm[exact_l2(s[perm],q,3)]
    for j,x in enumerate(q):
        np.testing.assert_array_equal(np.sum((s[mapped[j]]-x)**2,axis=1),np.sum((s[found[j]]-x)**2,axis=1))
    require_false();assert 'torch' not in sys.modules and 'faiss' not in sys.modules
    print('NumPy stable tie/duplicates/query permutation/support-distance permutation/label independence checks passed.')


def new_engine(device):
    import torch
    import tabdpt_adapter as old
    old.verify_sources();old.upstream.verify_checkpoint()
    class NumpyRetrieval(old.upstream.TabDPTClassifier):
        def _get_faiss_knn_indices(self,X_test,context_size,seed=None):
            require_false()
            if context_size!=K or self.context_reduction!='retrieval':raise RuntimeError('Wrong fixed retrieval settings.')
            result=exact_l2(self.X_train,X_test,K)
            self.numpy_neighbors=result.copy()
            if result.shape!=(len(X_test),K) or (result<0).any() or (result>=self.n_instances).any():raise RuntimeError('Invalid neighbor IDs.')
            if self.faiss_knn is not None:raise RuntimeError('Native FAISS index unexpectedly constructed.')
            return result
    engine=NumpyRetrieval(device=device,use_flash=False,compile=False,model_weight_path=str(old.upstream.CHECKPOINT),
        context_reduction='retrieval',normalizer='standard',missing_indicators=False,clip_sigma=8.,feature_reduction='pca',verbose=False)
    if sum(p.numel() for p in engine.model.parameters())!=old.upstream.SETTINGS['parameters']:
        raise RuntimeError('Pinned architecture changed.')
    if any(p.dtype!=torch.float32 for p in engine.model.parameters()):raise RuntimeError('Expected FP32 weights.')
    return engine


class RetrievalPredictor:
    def __init__(self,encoder,support,labels,ids,device='cpu'):
        import numpy as np
        import tabdpt_adapter as old
        self.encoder=encoder;self.support=np.asarray(support,dtype=np.float32).copy()
        self.labels=old.binary_labels(labels,len(support));self.ids=np.asarray(ids,dtype=np.int64).copy()
        if self.ids.shape!=self.labels.shape or len(np.unique(self.ids))!=len(self.ids):raise ValueError('Unique support IDs required.')
        self.device=device;self.provenance=old.verify_sources();self.last_record=None
        self.fingerprints={n:old.array_sha(getattr(self,n)) for n in ['support','labels','ids']}

    def predict_proba(self,x):
        import numpy as np
        import torch
        import tabdpt_adapter as old
        from tabdpt.utils import generate_random_permutation
        require_false()
        if self.provenance!=old.verify_sources():raise RuntimeError('Source provenance changed.')
        for name,digest in self.fingerprints.items():
            if old.array_sha(getattr(self,name))!=digest:raise RuntimeError('Portable support changed.')
        query=self.encoder.transform(x);old.check_encoded(query,len(x))
        engine=handle=None;calls=[];cursor=0;tick=time.monotonic()
        try:
            torch.manual_seed(SEED);np.random.seed(SEED)
            if self.device=='mps':torch.mps.manual_seed(SEED)
            engine=new_engine(self.device);engine.fit(self.support.copy(),self.labels.copy())
            np.testing.assert_array_equal(engine.y_train,self.labels)
            if engine.n_instances!=len(self.ids):raise RuntimeError('Support pool changed.')
            permutation=generate_random_permutation(engine.n_features,SEED).numpy()
            setup_seconds=time.monotonic()-tick
            def observe(_module,args,kwargs):
                nonlocal cursor
                actual_x=kwargs['x_src'].detach().cpu().numpy();actual_y=kwargs['y_src'].detach().cpu().numpy()
                batch=len(actual_x);ix=engine.numpy_neighbors[cursor:cursor+batch]
                if not 1<=batch<=8 or actual_x.shape[1]!=257 or actual_y.shape!=(batch,256):raise RuntimeError('Wrong actual forward shape.')
                # Check the exact returned IDs against an independent stable lexicographic distance calculation.
                for row,queryrow in zip(ix,engine.X_test[cursor:cursor+batch]):
                    dist=np.sum((engine.X_train.astype(np.float64)-queryrow.astype(np.float64))**2,axis=1)
                    want=np.lexsort((np.arange(len(dist)),dist))[:K]
                    np.testing.assert_array_equal(row,want)
                expected_support=engine.X_train.astype(np.float32)[ix][:,:,permutation]
                expected_query=engine.X_test.astype(np.float32)[cursor:cursor+batch][:,None,permutation]
                expected=np.concatenate([expected_support,expected_query],axis=1)
                expected=np.pad(expected,((0,0),(0,0),(0,engine.max_features-engine.n_features)))
                np.testing.assert_array_equal(actual_x,expected)
                np.testing.assert_array_equal(actual_y,self.labels[ix].astype(np.float32))
                calls.append(dict(batch=batch,tokens=actual_x.shape[1],labels=actual_y.shape[1],neighbor_ids_sha256=old.array_sha(self.ids[ix])))
                cursor+=batch
            handle=engine.model.register_forward_pre_hook(observe,with_kwargs=True)
            start=time.monotonic()
            with torch.no_grad():p=np.asarray(engine.ensemble_predict_proba(query,**PREDICT))
            if self.device=='mps':torch.mps.synchronize()
            seconds=time.monotonic()-start
            if p.shape!=(len(x),2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=0,atol=2e-6):raise RuntimeError('Incomplete probabilities.')
            if cursor!=len(x) or engine.faiss_knn is not None:raise RuntimeError('Forward coverage/native retrieval invariant failed.')
            self.last_record=dict(support_pool=len(self.ids),query_rows=len(x),setup_seconds=setup_seconds,prediction_seconds=seconds,
                calls=calls,neighbor_sha256=old.array_sha(engine.numpy_neighbors),resources=old.upstream.snapshot(self.device),native_faiss_search=False)
            require_false();return p
        finally:
            if handle is not None:handle.remove()
            old.upstream.release(engine,self.device);engine=None;old.upstream.release(None,self.device)

    def __getstate__(self):
        state=dict(self.__dict__);state['device']='cpu';return state


def predictor(frame,y,device):
    import numpy as np
    import tabdpt_adapter as old
    encoder=old.CompactEncoder().fit(frame)
    return RetrievalPredictor(encoder,encoder.transform(frame),y,np.arange(len(frame),dtype=np.int64),device)


def synthetic(n,query):
    import numpy as np
    import pandas as pd
    import tabdpt_adapter as old
    rng=np.random.default_rng(SEED+n)
    x=pd.DataFrame(rng.normal(size=(n+query,13)),columns=old.COMPACT)
    for col in old.CATEGORICAL:x[col]=rng.choice(['a','b','c'],len(x)).astype(object)
    for col in old.COMPACT:x.loc[rng.random(len(x))<.02,col]=np.nan
    return x.iloc[:n].copy(),np.arange(n)%2,x.iloc[n:].copy()


def comparison(a,b,tolerance):
    import numpy as np
    diff=float(np.max(np.abs(a-b)))
    if a.shape!=b.shape or not np.isfinite(b).all() or diff>tolerance or not np.array_equal(a[:,1]>=.5,b[:,1]>=.5):
        raise RuntimeError(f'Probability/decision disagreement maxdiff={diff}')
    return diff


def source_hashes():
    paths=[Path(__file__),PLAN,ROOT/'tabdpt_adapter.py',ROOT/'tabdpt_resource_probe.py',ROOT/'requirements-tabdpt.lock.txt']
    paths.extend(p for p in (ROOT/'third_party/tabdpt').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def frozen_check():
    record=read(OUT/'freeze.json')
    for n,d in record['source_hashes'].items():
        if sha(ROOT/n)!=d:raise RuntimeError('Frozen source changed: '+n)
    if sha(ROOT/record['checkpoint_path'])!=record['checkpoint_sha256']:raise RuntimeError('Checkpoint changed.')
    return record


def worker():
    require_false();frozen=frozen_check()
    import joblib
    import numpy as np
    import torch
    import tabdpt_adapter as old
    if hashlib.sha256(old.upstream.environment().encode()).hexdigest()!=frozen['environment_sha256']:raise RuntimeError('Environment changed.')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
    stages={}
    def progress(stage):write(OUT/'progress.json',dict(stage=stage,completed_stages=stages))
    progress('cpu_mps_same_checkpoint_257_support_2_query')
    x,y,q=synthetic(257,2);model=predictor(x,y,'cpu');cpu=model.predict_proba(q)
    model.device='mps';mps=model.predict_proba(q);stages['cpu_mps']=comparison(cpu,mps,2e-5)
    del model;old.upstream.release(None,'mps')
    progress('full_pool8693_query32')
    x,y,q=synthetic(8693,32);model=predictor(x,y,'mps');p=model.predict_proba(q);timing=[model.last_record]
    joblib.dump(model,OUT/'model.joblib');del model;old.upstream.release(None,'mps')
    progress('full_reload')
    model=joblib.load(OUT/'model.joblib')
    if model.device!='cpu':raise RuntimeError('Artifact is not portable CPU state.')
    model.device='mps';reload=model.predict_proba(q);timing.append(model.last_record);stages['reload']=comparison(p,reload,2e-6)
    progress('reverse_query')
    reverse=model.predict_proba(q.iloc[::-1]);timing.append(model.last_record);stages['reverse']=comparison(p,reverse[::-1],2e-6)
    progress('tail_subset19')
    subset=model.predict_proba(q.iloc[:19]);stages['subset19']=comparison(p[:19],subset,2e-6)
    tail_record=model.last_record
    del model;old.upstream.release(None,'mps')
    np.savez_compressed(OUT/'probabilities.npz',cpu=cpu,mps=mps,full=p,reload=reload,reverse=reverse,subset19=subset)
    # Includes observed audited preprocessing/retrieval/forward cost; never extrapolate only model kernels.
    prediction=max(r['prediction_seconds'] for r in timing);setup=max(r['setup_seconds'] for r in timing)
    forecast=1.25*(2*setup+2*(2318/32)*prediction)
    require_false();frozen_check()
    write(OUT/'worker_result.json',dict(status='complete',stages=stages,timing=timing,tail_record=tail_record,
        projected_fold2318_two_predictions_seconds=forecast,cost_passed=forecast<=1200,
        probabilities_sha256=sha(OUT/'probabilities.npz'),model_sha256=sha(OUT/'model.joblib'),
        real_data_read=False,accuracy_computed=False,cv_authorized=False))


def probe():
    if OUT.exists() or REPORT.exists():raise FileExistsError('Independent fresh output only; no overwrite/resume.')
    plan=read(PLAN)
    if plan['K']!=K or plan['predict']!=PREDICT:raise RuntimeError('Fixed probe recipe changed.')
    sources=source_hashes()
    for n,d in sources.items():
        if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()!=d:raise RuntimeError('Commit probe sources first: '+n)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True);start=time.monotonic();deadline=start+600;child=None
    try:
        # Metadata only in parent; all native/model imports isolated under external deadline.
        envcode="import importlib.metadata,sys;sys.stdout.write(chr(10).join(sorted(f\"{d.metadata['Name']}=={d.version}\" for d in importlib.metadata.distributions()))+chr(10))"
        env=subprocess.check_output([sys.executable,'-c',envcode],timeout=deadline-time.monotonic())
        (OUT/'environment.lock.txt').write_bytes(env)
        weight='artifacts/pretrained/tabdpt-v1.3/tabdpt1_3.safetensors'
        record=dict(source_hashes=sources,checkpoint_path=weight,checkpoint_sha256=sha(ROOT/weight),environment_sha256=hashlib.sha256(env).hexdigest(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),plan=plan)
        write(OUT/'freeze.json',record)
        with (OUT/'worker.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=child.wait(timeout=max(.01,deadline-time.monotonic()))
        if code:raise RuntimeError(f'Isolated worker exit{code}; see preserved worker.log/failure.')
        result=read(OUT/'worker_result.json');frozen_check()
        if result['status']!='complete' or time.monotonic()>deadline:raise RuntimeError('Incomplete/late resource probe.')
        result.update(status='resource_complete',elapsed_seconds=time.monotonic()-start,
            qualified_for_root_consideration=result['cost_passed'],no_automatic_diagnostic=True,freeze_sha256=sha(OUT/'freeze.json'))
        write(OUT/'result.json',result);write(REPORT/'result.json',result);write(REPORT/'freeze.json',record)
    except BaseException as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        failure=dict(status='failed_unqualified',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-start)
        if (OUT/'worker_failure.json').exists():failure['worker_failure']=read(OUT/'worker_failure.json')
        write(OUT/'failure.json',failure);write(REPORT/'failure.json',failure);raise


if __name__=='__main__':
    runtime_entry()
    # Stable import name for portable CPU-only pickle state in this fresh worker.
    sys.modules.setdefault('tabdpt_numpy_retrieval_nov18',sys.modules[__name__])
    RetrievalPredictor.__module__='tabdpt_numpy_retrieval_nov18'
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--probe',action='store_true');mode.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.check:check()
    elif args.probe:probe()
    else:
        try:worker()
        except BaseException as error:
            write(OUT/'worker_failure.json',dict(error_type=type(error).__name__,error=str(error)));raise
