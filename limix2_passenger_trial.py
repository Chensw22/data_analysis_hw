"""One predeclared CPU LimiX-2 passenger recipe; no holdout labels or upload."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import json
import signal
import subprocess
import sys
import time
import resource
from pathlib import Path
from limix2_setup import ROOT, WORK, sha, write

REPORT=ROOT/'reports/limix2_passenger_trial'
OUT=ROOT/'artifacts/limix2_passenger_trial'
SETTINGS=dict(support_rows=512,pipelines=[0],seed_base=20261118,query_batch=256,
              cpu_threads=4,device='cpu',mix_precision=False,fold_deadline_seconds=900,
              threshold=.5,combinations=['half_rawMitra','half_v96'])
SOURCES=['limix2_passenger_trial.py','limix2_setup.py','limix2_cpu_probe.py',
         'limix2_cpu_compat.py','error_overlap_diagnostic.py','exaone_validation_data.py',
         'exception_error_detector_nov18.py','reports/limix2_setup/download.json',
         'reports/limix2_setup/environment.lock.txt']


def check_freeze():
    plan=json.loads((REPORT/'protocol.json').read_text())
    if plan['settings']!=SETTINGS:raise RuntimeError('Recipe differs.')
    for name,digest in plan['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed: '+name)
    return plan


def valid(p,n):
    import numpy as np
    p=np.asarray(p)
    if p.shape!=(n,2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-6):
        raise RuntimeError('Invalid native binary probability matrix.')
    return p[:,1].copy()


def worker(fold):
    start=time.monotonic();meta={'status':'running','fold':fold}
    try:
        check_freeze()
        import numpy as np
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        from limix2_cpu_probe import verify
        weight,_=verify()
        packet_path=OUT/f'fold{fold}_packet.json'
        packet=json.loads(packet_path.read_text())
        expected={'support','query','support_y','support_ids','query_ids','categorical_indices','columns'}
        if set(packet)!=expected:raise RuntimeError('Unexpected worker packet, query targets forbidden.')
        if set(packet['support_ids']) & set(packet['query_ids']):raise RuntimeError('Support/query overlap.')
        x=np.asarray([[np.nan if v is None else v for v in row] for row in packet['support']],dtype=object)
        q=np.asarray([[np.nan if v is None else v for v in row] for row in packet['query']],dtype=object)
        y=np.asarray(packet['support_y'],dtype=np.int64)
        if x.shape!=(512,13) or q.shape[1]!=13 or set(y)!={0,1}:raise RuntimeError('Context schema differs.')
        import limix2_cpu_compat as compat
        compat.install()
        sys.path.insert(0,str(WORK/'source'))
        from inference.predictor import LimiXPredictor
        if torch.serialization.get_unsafe_globals_in_checkpoint(weight):raise RuntimeError('Unsafe checkpoint globals.')
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        inference=json.loads((WORK/'source/config/cls_default_noretrieval_v2.json').read_text())
        inference['pipelines']=[inference['pipelines'][0]]
        model=LimiXPredictor(device=torch.device('cpu'),model_path=str(weight),ckpt=checkpoint,
            inference_config=inference,mix_precision=False,seed=20261118+fold,
            test_batch_mode='fixed',test_batch_size=256,inference_with_DDP=False,
            use_data_cache=False,categorical_features_indices=packet['categorical_indices'])
        before=time.monotonic()
        p=valid(model.predict(x,y,q,task_type='Classification'),len(q))
        elapsed=time.monotonic()-before
        if any(t.device.type!='cpu' for t in model.model.parameters()):raise RuntimeError('NonCPU inference.')
        if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Runtime flag changed.')
        np.savez(OUT/f'fold{fold}_prediction.npz',prob=p,
                 query_ids=np.asarray(packet['query_ids'],dtype=np.int64),
                 support_ids=np.asarray(packet['support_ids'],dtype=np.int64))
        check_freeze()
        meta.update(status='complete',support_rows=len(x),query_rows=len(q),pipelines=1,
                    prediction_seconds=elapsed,packet_sha256=sha(packet_path),
                    prediction_sha256=sha(OUT/f'fold{fold}_prediction.npz'))
    except BaseException as exc:
        meta.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        meta.update(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(REPORT/f'fold{fold}_worker.json',meta)


def statistics(p,y,raw,v96,masks,frame):
    import numpy as np
    from sklearn.metrics import log_loss
    results={}
    for name,prob in [('LimiX2',p),('rawMitra',raw),('v96',v96),
                      ('half_rawMitra',.5*p+.5*raw),('half_v96',.5*p+.5*v96)]:
        ok=(prob>=.5)==y
        row=dict(rows=len(y),accuracy=float(ok.mean()),log_loss=float(log_loss(y,prob,labels=[0,1])))
        for refname,ref in [('rawMitra',raw),('v96',v96)]:
            prior=(ref>=.5)==y
            row['vs_'+refname]=dict(repairs=int((ok&~prior).sum()),breaks=int((~ok&prior).sum()),net=int(ok.sum()-prior.sum()))
        row['subsets']={key:dict(rows=int(mask.sum()),correct=int(ok[mask].sum()),
            repairs_vs_raw=int((ok&((raw>=.5)!=y)&mask).sum()),
            breaks_vs_raw=int((~ok&((raw>=.5)==y)&mask).sum())) for key,mask in masks.items()}
        row['earth']={}
        for state in ['True','False']:
            m=(frame.HomePlanet.astype(str).to_numpy()=='Earth')&(frame.CryoSleep.astype(str).to_numpy()==state)
            pred=prob[m]>=.5;truth=y[m].astype(bool)
            row['earth']['sleep' if state=='True' else 'awake']=dict(rows=int(m.sum()),
                tn=int((~pred&~truth).sum()),fp=int((pred&~truth).sum()),fn=int((~pred&truth).sum()),tp=int((pred&truth).sum()))
        results[name]=row
    return results


def run():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import error_overlap_diagnostic as overlap
    import exaone_validation_data as data
    if OUT.exists() or (REPORT/'protocol.json').exists():raise FileExistsError('No overwrite or partial resume.')
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();fold_results=[]
    try:
        inputs,splits,folds,bases,evidence=overlap.load_authenticated()
        dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra']
        vpath=ROOT/'artifacts/v96_oof_completion/oof.npz'
        vrpath=ROOT/'reports/v96_oof_completion/result.json';vr=json.loads(vrpath.read_text())
        if vr['status']!='complete_v96_fixed15_original3fold_oof' or sha(vpath)!=vr['oof_sha256']:raise RuntimeError('Invalid v96 cache.')
        with np.load(vpath,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['dev_ids'],dev);np.testing.assert_array_equal(z['fold_id'],folds)
            v96=z['prob'].copy()
        evidence.update({str(p.relative_to(ROOT)):sha(p) for p in [vpath,vrpath]})
        frame=inputs['features'].iloc[dev].copy()
        if list(frame.columns)!=data.COMPACT:raise RuntimeError('Compact13 differs.')
        configs=json.loads((WORK/'source/config/cls_default_noretrieval_v2.json').read_text())
        plan=dict(status='fixed_before_any_LimiX_passenger_prediction',settings=SETTINGS,
            columns=data.COMPACT,categorical_columns=data.CATEGORICAL,official_pipeline=configs['pipelines'][0],
            support_selection='Sorted512 stratified random outertrain indices; seed20261118+fold; all queries retained.',
            missing='Numerical NaN and category __MISSING__ converted to np.nan before native preprocessing.',
            worker_boundary='Worker packet contains support labels only. No query labels, hard masks, Name or heldout labels.',
            source_hashes={name:sha(ROOT/name) for name in SOURCES},evidence_sha256=evidence,
            limitations=['Adaptive reused development folds, not independent new validation.',
              '512 supports and one official pipeline differ from default32 and full-context cached baselines.',
              'Half means fixed in advance; no weight, threshold, context or pipeline selection from results.',
              'CPU recipe; prior MPS alignment failure remains separate. No upload or fullfit.'],attribution='Built with StableAI LimiX')
        write(REPORT/'protocol.json',plan)
        full=np.full(len(dev),np.nan)
        for fold,(a,b) in enumerate(splits):
            support,_=train_test_split(a,train_size=512,stratify=y[a],random_state=20261118+fold)
            support=np.sort(support)
            def values(f):
                return [[None if pd.isna(v) or (isinstance(v,str) and v=='__MISSING__') else v for v in row] for row in f.to_numpy(dtype=object).tolist()]
            packet=dict(support=values(frame.iloc[support]),query=values(frame.iloc[b]),support_y=y[support].astype(int).tolist(),
                support_ids=dev[support].tolist(),query_ids=dev[b].tolist(),columns=data.COMPACT,
                categorical_indices=[data.COMPACT.index(c) for c in data.CATEGORICAL])
            write(OUT/f'fold{fold}_packet.json',packet)
            print('Starting fixed CPU fold',fold,'support512/query',len(b),flush=True)
            with (OUT/f'fold{fold}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,__file__,'--worker',str(fold)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError(f'Fold{fold} timed out after900s.')
            if code:raise RuntimeError(f'Fold{fold} worker failed; see independent log.')
            wm=json.loads((REPORT/f'fold{fold}_worker.json').read_text());predpath=OUT/f'fold{fold}_prediction.npz'
            if wm['status']!='complete' or sha(predpath)!=wm['prediction_sha256']:raise RuntimeError('Incomplete fold.')
            with np.load(predpath,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['query_ids'],dev[b]);np.testing.assert_array_equal(z['support_ids'],dev[support]);p=z['prob'].copy()
            full[b]=p
            # Historical target-selected masks are loaded only after native predictions exist; never enter worker.
            from exception_error_detector_nov18 import reporting_masks
            masks=reporting_masks(dev,folds,y,inputs,bases,evidence)
            stats=statistics(p,y[b],raw[b],v96[b],{k:m[b] for k,m in masks.items()},frame.iloc[b])
            fr=dict(fold=fold,status='complete',worker=wm,metrics=stats)
            fold_results.append(fr);write(REPORT/f'fold{fold}_result.json',fr)
            print(json.dumps({'fold':fold,'seconds':wm['seconds'],'metrics':{k:dict(accuracy=v['accuracy'],vs_raw=v['vs_rawMitra'],hard546=v['subsets']['all_recipes_common_wrong546']['correct']) for k,v in stats.items()}}),flush=True)
        check_freeze()
        np.savez(OUT/'oof.npz',dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str),
                 probability=full,raw=raw,v96=v96,half_raw=.5*(full+raw),half_v96=.5*(full+v96))
        result=dict(status='complete_fixed3fold_cpu_trial',metrics=statistics(full,y,raw,v96,masks,frame),
            folds=fold_results,seconds=time.monotonic()-started,oof_path=str((OUT/'oof.npz').relative_to(ROOT)),
            oof_sha256=sha(OUT/'oof.npz'),protocol_sha256=sha(REPORT/'protocol.json'),evidence_sha256=evidence,
            limitations=plan['limitations'],no_submission=True)
        write(REPORT/'result.json',result)
        print(json.dumps({'status':result['status'],'metrics':{k:dict(accuracy=v['accuracy'],vs_raw=v['vs_rawMitra'],hard546=v['subsets']['all_recipes_common_wrong546']['correct']) for k,v in result['metrics'].items()}}),flush=True)
    except BaseException as exc:
        write(REPORT/'failure.json',dict(status='failed_no_complete_candidate',error_type=type(exc).__name__,error=str(exc),completed_folds=[f['fold'] for f in fold_results],seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    worker(args.worker) if args.worker is not None else run()
