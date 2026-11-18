"""Fixed CPU LimiX-2: planet-routed 2048-row contexts and official pipelines0..3."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import gc
import json
import signal
import subprocess
import sys
import time
import resource
from limix2_setup import ROOT,WORK,sha,write
from limix2_passenger_trial import valid,statistics

REPORT=ROOT/'reports/limix2_planet_trial'
OUT=ROOT/'artifacts/limix2_planet_trial'
SETTINGS=dict(max_support=2048,target_same_planet=1536,pipelines=[0,1,2,3],
    seed_base=20261118,query_batch=256,cpu_threads=4,device='cpu',mix_precision=False,
    fold_deadline_seconds=3600,threshold=.5,combinations=['half_rawMitra','half_v96'])
OWN=['limix2_planet_trial.py','limix2_passenger_trial.py','limix2_setup.py','limix2_cpu_probe.py',
     'limix2_cpu_compat.py','error_overlap_diagnostic.py','exaone_validation_data.py',
     'exception_error_detector_nov18.py','reports/limix2_setup/download.json',
     'reports/limix2_setup/environment.lock.txt']


def check():
    plan=json.loads((REPORT/'protocol.json').read_text())
    if plan['settings']!=SETTINGS:raise RuntimeError('Recipe mismatch.')
    for name,digest in plan['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Source changed: '+name)
    return plan


def choose_support(a,planets,route,seed):
    import numpy as np
    rng=np.random.default_rng(seed)
    n=min(2048,len(a))
    if route=='__MISSING__':return np.sort(rng.permutation(a)[:n])
    same=rng.permutation(a[planets[a]==route]);other=rng.permutation(a[planets[a]!=route])
    k=min(1536,len(same),n)
    j=min(n-k,len(other))
    k=n-j
    out=np.sort(np.concatenate([same[:k],other[:j]]))
    if len(out)!=n or len(np.unique(out))!=n:raise RuntimeError('Support selection incomplete.')
    return out


def worker(fold):
    started=time.monotonic();record=dict(status='running',fold=fold,routes=[])
    try:
        check()
        import numpy as np
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        from limix2_cpu_probe import verify
        weight,_=verify()
        packetpath=OUT/f'fold{fold}_packet.json';packet=json.loads(packetpath.read_text())
        if set(packet)!={'query_ids','columns','categorical_indices','routes'}:raise RuntimeError('Unexpected packet fields.')
        all_ids=np.asarray(packet['query_ids'],dtype=np.int64);pred=np.full(len(all_ids),np.nan)
        if len(set(all_ids.tolist()))!=len(all_ids):raise RuntimeError('Duplicate query rows.')
        import limix2_cpu_compat as compat
        compat.install();sys.path.insert(0,str(WORK/'source'))
        from inference.predictor import LimiXPredictor
        if torch.serialization.get_unsafe_globals_in_checkpoint(weight):raise RuntimeError('Unsafe checkpoint.')
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        inference=json.loads((WORK/'source/config/cls_default_noretrieval_v2.json').read_text())
        inference['pipelines']=inference['pipelines'][:4]
        for route,part in packet['routes'].items():
            if set(part)!={'support','query','support_y','support_ids','query_ids','positions','support_planet_counts'}:
                raise RuntimeError('Query labels or unexpected fields in route packet.')
            if set(part['support_ids'])&set(all_ids.tolist()):raise RuntimeError('Outer query entered support.')
            x=np.asarray([[np.nan if v is None else v for v in row] for row in part['support']],dtype=object)
            q=np.asarray([[np.nan if v is None else v for v in row] for row in part['query']],dtype=object)
            y=np.asarray(part['support_y'],dtype=np.int64);positions=np.asarray(part['positions'],dtype=np.int64)
            if x.shape!=(2048,13) or q.shape!=(len(positions),13) or set(y)!={0,1}:raise RuntimeError('Context schema mismatch.')
            np.testing.assert_array_equal(all_ids[positions],part['query_ids'])
            if np.isfinite(pred[positions]).any():raise RuntimeError('Repeated query route.')
            before=time.monotonic()
            model=LimiXPredictor(device=torch.device('cpu'),model_path=str(weight),ckpt=checkpoint,
                inference_config=inference,mix_precision=False,seed=20261118+fold,
                test_batch_mode='fixed',test_batch_size=256,inference_with_DDP=False,
                use_data_cache=False,categorical_features_indices=packet['categorical_indices'])
            p=valid(model.predict(x,y,q,task_type='Classification'),len(q));pred[positions]=p
            if any(t.device.type!='cpu' for t in model.model.parameters()):raise RuntimeError('NonCPU model.')
            record['routes'].append(dict(route=route,support_rows=len(x),query_rows=len(q),
                support_planet_counts=part['support_planet_counts'],seconds=time.monotonic()-before))
            np.savez(OUT/f'fold{fold}_{route}_prediction.npz',prob=p,query_ids=all_ids[positions],support_ids=np.asarray(part['support_ids'],dtype=np.int64))
            del model;gc.collect()
            write(REPORT/f'fold{fold}_progress.json',record)
            print(json.dumps(record['routes'][-1]),flush=True)
        if not np.isfinite(pred).all():raise RuntimeError('Incomplete query partition.')
        if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Runtime flag changed.')
        np.savez(OUT/f'fold{fold}_prediction.npz',prob=pred,query_ids=all_ids)
        check()
        record.update(status='complete',packet_sha256=sha(packetpath),prediction_sha256=sha(OUT/f'fold{fold}_prediction.npz'))
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record.update(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(REPORT/f'fold{fold}_worker.json',record)


def run():
    import numpy as np
    import pandas as pd
    import error_overlap_diagnostic as overlap
    import exaone_validation_data as data
    if OUT.exists() or (REPORT/'protocol.json').exists():raise FileExistsError('No overwrite/resume.')
    REPORT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True)
    started=time.monotonic();fold_results=[]
    try:
        inputs,splits,folds,bases,evidence=overlap.load_authenticated()
        dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra']
        vrpath=ROOT/'reports/v96_oof_completion/result.json';vp=ROOT/'artifacts/v96_oof_completion/oof.npz'
        vr=json.loads(vrpath.read_text())
        if vr['status']!='complete_v96_fixed15_original3fold_oof' or sha(vp)!=vr['oof_sha256']:raise RuntimeError('v96 provenance invalid.')
        with np.load(vp,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['dev_ids'],dev);np.testing.assert_array_equal(z['fold_id'],folds);v96=z['prob'].copy()
        evidence.update({str(p.relative_to(ROOT)):sha(p) for p in [vrpath,vp]})
        frame=inputs['features'].iloc[dev].copy()
        if list(frame.columns)!=data.COMPACT:raise RuntimeError('Original13 required.')
        planets=frame.HomePlanet.astype(str).to_numpy()
        config=json.loads((WORK/'source/config/cls_default_noretrieval_v2.json').read_text())
        plan=dict(status='fixed_before_new_prediction',settings=SETTINGS,columns=data.COMPACT,
            categorical_columns=data.CATEGORICAL,official_pipelines=config['pipelines'][:4],
            support_rule='For each known query HomePlanet, draw1536 samePlanet+512 others without replacement using outertrain attributes only. If samePlanet short, use all and fill from others. If others short, fill from same. Missing queryPlanet uses uniform2048 outertrain. Sort selected original positions. No labels used in support selection.',
            seed_rule='Each route reinitializes NumPy default_rng(20261118+fold); native seed same20261118+fold.',
            boundaries='No query target, Name, oldholdout targets or historical hardmask enters native worker. Allqueryrows retained. Outertrain-only native preprocessing.',
            source_hashes={name:sha(ROOT/name) for name in OWN},evidence_sha256=evidence,
            limitations=['Adaptive already-used development folds; not independent confirmation.',
             'Bundled increase of context, planet routing and4pipelines; cannot attribute change to one factor.',
             'No tuning by route/fold/result; unknown planets fallback predeclared.',
             'Full raw/v96 baselines have different architectures/support; combinations fixed .5 each.',
             'No automatic export/upload; no oldfailed evidence replaced.'],attribution='Built with StableAI LimiX')
        write(REPORT/'protocol.json',plan);full=np.full(len(dev),np.nan)
        def values(f):
            return [[None if pd.isna(v) or (isinstance(v,str) and v=='__MISSING__') else v for v in row] for row in f.to_numpy(dtype=object).tolist()]
        for fold,(a,b) in enumerate(splits):
            packet=dict(query_ids=dev[b].tolist(),columns=data.COMPACT,
                categorical_indices=[data.COMPACT.index(c) for c in data.CATEGORICAL],routes={})
            for route in sorted(set(planets[b])):
                pos=np.flatnonzero(planets[b]==route);qidx=b[pos]
                support=choose_support(a,planets,route,20261118+fold)
                if set(support)&set(b):raise RuntimeError('Support includes outer validation.')
                names,counts=np.unique(planets[support],return_counts=True)
                packet['routes'][route]=dict(support=values(frame.iloc[support]),query=values(frame.iloc[qidx]),
                    support_y=y[support].astype(int).tolist(),support_ids=dev[support].tolist(),query_ids=dev[qidx].tolist(),
                    positions=pos.tolist(),support_planet_counts=dict(zip(names.tolist(),counts.astype(int).tolist())))
            write(OUT/f'fold{fold}_packet.json',packet)
            print('Starting CPU planet2048x4 fold',fold,'query',len(b),flush=True)
            with (OUT/f'fold{fold}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,__file__,'--worker',str(fold)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError(f'Fold{fold} exceeded3600seconds.')
            if code:raise RuntimeError(f'Fold{fold} failed; see preserved worker log.')
            wm=json.loads((REPORT/f'fold{fold}_worker.json').read_text());pp=OUT/f'fold{fold}_prediction.npz'
            if wm['status']!='complete' or sha(pp)!=wm['prediction_sha256']:raise RuntimeError('Incomplete fold.')
            with np.load(pp,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['query_ids'],dev[b]);p=z['prob'].copy()
            full[b]=p
            from exception_error_detector_nov18 import reporting_masks
            masks=reporting_masks(dev,folds,y,inputs,bases,evidence)
            stats=statistics(p,y[b],raw[b],v96[b],{k:m[b] for k,m in masks.items()},frame.iloc[b])
            fr=dict(fold=fold,status='complete',worker=wm,metrics=stats)
            fold_results.append(fr);write(REPORT/f'fold{fold}_result.json',fr)
            print(json.dumps({'fold':fold,'seconds':wm['seconds'],'metrics':{k:dict(accuracy=v['accuracy'],vs_raw=v['vs_rawMitra'],hard546=v['subsets']['all_recipes_common_wrong546']['correct']) for k,v in stats.items()}}),flush=True)
        check()
        np.savez(OUT/'oof.npz',dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str),
            probability=full,raw=raw,v96=v96,half_raw=.5*(full+raw),half_v96=.5*(full+v96))
        result=dict(status='complete_fixed3fold_cpu_planet_trial',metrics=statistics(full,y,raw,v96,masks,frame),
            folds=fold_results,seconds=time.monotonic()-started,oof_path=str((OUT/'oof.npz').relative_to(ROOT)),
            oof_sha256=sha(OUT/'oof.npz'),protocol_sha256=sha(REPORT/'protocol.json'),evidence_sha256=evidence,
            limitations=plan['limitations'],no_submission=True)
        write(REPORT/'result.json',result)
        print(json.dumps({'status':result['status'],'metrics':{k:dict(accuracy=v['accuracy'],vs_raw=v['vs_rawMitra'],hard546=v['subsets']['all_recipes_common_wrong546']['correct']) for k,v in result['metrics'].items()}}),flush=True)
    except BaseException as exc:
        write(REPORT/'failure.json',dict(status='failed_no_complete_candidate',error_type=type(exc).__name__,error=str(exc),completed_folds=[f['fold'] for f in fold_results],seconds=time.monotonic()-started));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    worker(args.worker) if args.worker is not None else run()
