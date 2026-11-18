"""Zero-shot binary label permutation symmetry on original three outer folds."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):raise RuntimeError('Fresh FALSE process required')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import json
from pathlib import Path
import hashlib
import time
import signal
import subprocess
import numpy as np
import pandas as pd
import torch
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import error_rescue_feature_probe as helper
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/mitra_label_flip_probe'
OUT=ROOT/'artifacts/mitra_label_flip_probe'
BUDGET=1200

def predict(features,ids,labels,qids,seed,key):
    if len(ids)!=4636 or len(qids)!=2318 or np.intersect1d(ids,qids).size:raise ValueError('Original disjoint fold required')
    model=None;t=time.monotonic()
    try:
        model=base.fit_mitra(features.loc[ids],labels,ids,seed,'mps',recipe='zero-shot',query_chunk=2318,log_path=OUT/(key+'_fit.json'))
        if len(model.support)!=4636 or not np.array_equal(model.labels,labels) or model.preprocessor.columns!=base.COMPACT:raise RuntimeError('Support/label/feature recipe mismatch')
        p=np.asarray(model.predict_proba(features.loc[qids]))[:,1]
        if p.shape!=(2318,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('Invalid fullquery')
        meta=dict(seed=seed,training_rows=4636,query_rows=2318,selected_steps=0,checkpoint_sha256=base.WEIGHT_SHA,
            training_ids_sha256=base.ids_hash(ids),query_ids_sha256=base.ids_hash(qids),labels_sha256=base.ids_hash(labels),
            feature_count=model.preprocessor.n_features,seconds=time.monotonic()-t)
        return p,meta
    finally:
        if model is not None:model.release()
        model=None;base.release_memory()

def run():
    if OUT.exists() or REPORT.exists():raise RuntimeError('Fixed experiment no overwrite')
    previous,lock=counts.verify_reference()
    inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];raw=bases['rawMitra'];features=inputs['features']
    names=set(helper.OWN)|{Path(__file__).name,'error_overlap_diagnostic.py','relational_error_corrector.py'}
    names.update(n for n in previous['source_hashes'] if n.startswith('third_party/mitra/'))
    sources={n:r.sha(ROOT/n) for n in names};sources.update(evidence)
    historical=[]
    for fold,(a,b) in enumerate(splits):
        sha=base.ids_hash(dev[a]);path=ROOT/'artifacts/run_038'/f'v71_Mitra2_zeroshot_seed{20261112+fold}_{sha[:12]}_training.json';meta=json.loads(path.read_text())
        if meta['seed']!=20261112+fold or meta['training_ids_sha256']!=sha or meta['training_rows']!=4636 or meta['selected_steps']!=0 or meta['checkpoint_sha256']!=base.WEIGHT_SHA:raise RuntimeError('Old raw metadata differs')
        sources[str(path.relative_to(ROOT))]=r.sha(path);historical.append(str(path.relative_to(ROOT)))
    catalog_path=ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv';sources[str(catalog_path.relative_to(ROOT))]=r.sha(catalog_path)
    REPORT.mkdir(parents=True);OUT.mkdir(parents=True)
    plan=dict(status='frozen_before_any_inference',recipe='OriginalrawMitra compact13 all4636outertrain support; flip onlylabels y→1-y, correctedprobability1−p_flip. Sameoriginal checkpoint/prep/seed perarm.',
        folds=3,split_seed=20261112,fit_seed='20261112+fold, matches originalrun038 metadata',query_chunk=2318,device='mps',memory_fraction=.65,cpu_threads=4,
        candidates=['raw_cached','flipped_corrected','fixed_half_mean'],threshold=.5,mean='0.5raw+0.5flipped_corrected; no tuning',
        original_check='Recompute fold0 original labels first; require np.array_equal freshp/cachedraw before anyflipped inference.',
        boundary='Onlyoutertrain labels passed to model, querylabels/hardmask notpassed; noadaptation, no supportselection, no seedsearch.',
        budget_seconds=BUDGET,historical_metadata=historical,
        limitations=['Adaptive reused6954dev, not independent validation.','Unequal probabilities imply checkpoint class-label asymmetry, not necessarily a beneficial average.','No v96 reverse-label adaptation authorized in this experiment.'])
    r.write(REPORT/'protocol.json',plan);sources[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json')
    frozen=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),protocol=plan)
    r.write(OUT/'freeze.json',frozen);r.write(REPORT/'freeze.json',frozen)
    start=time.monotonic();base.DEADLINE=start+BUDGET
    def timeout(*_):raise TimeoutError('Fixedlabelswap1200s')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    try:
        if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        a,b=splits[0];fresh,checkmeta=predict(features,dev[a],y[a].copy(),dev[b],20261112,'fold0_original')
        np.testing.assert_array_equal(fresh,raw[b]);checkmeta['cached_probability_max_diff']=0.;r.write(REPORT/'raw_fold0_check.json',checkmeta)
        corrected=np.full(len(y),np.nan);records=[]
        for fold,(a,b) in enumerate(splits):
            helper.verify_sources(frozen);flipped,meta=predict(features,dev[a],(1-y[a]).copy(),dev[b],20261112+fold,f'fold{fold}_flipped')
            corrected[b]=1.-flipped
            record=dict(fold=fold,model=meta,mean_abs_raw_difference=float(np.mean(np.abs(corrected[b]-raw[b]))),max_abs_raw_difference=float(np.max(np.abs(corrected[b]-raw[b]))))
            records.append(record)
            np.savez_compressed(OUT/f'fold{fold}.npz',prob=corrected[b],native_flipped=flipped,train_ids=dev[a],query_ids=dev[b],passenger_ids=np.asarray(inputs['passengers'][b],dtype=str))
        predictions=dict(raw_cached=raw,flipped_corrected=corrected,fixed_half_mean=.5*raw+.5*corrected)
        rows=pd.read_csv(catalog_path,dtype={'PassengerId':str})
        for k,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[k],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        masks=dict(all=np.ones(len(y),bool),hard546=hard,hard914=rows.family_wrong_count.to_numpy()==5,commoncorrect5263=rows.family_wrong_count.to_numpy()==0)
        def stat(m):return {k:dict(correct=int((((p>=.5)==y)&m).sum()),vs_raw=r.pair(p,raw,y,m)) for k,p in predictions.items()}
        for i,record in enumerate(records):record['metrics']={k:r.score(p[folds==i],y[folds==i]) for k,p in predictions.items()};record['paired']=stat(folds==i);record['hard546']=stat((folds==i)&hard);r.write(REPORT/f'fold{i}.json',record)
        cache=OUT/'oof.npz';np.savez_compressed(cache,**predictions,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        helper.verify_sources(frozen);base.check_deadline()
        result=dict(status='complete',metrics={k:r.score(p,y) for k,p in predictions.items()},subsets={k:stat(m) for k,m in masks.items()},folds=records,
            raw_fold0_exact_cache_match=True,seconds=time.monotonic()-start,oof_sha256=r.sha(cache),source_hashes=sources,protocol_sha256=r.sha(REPORT/'protocol.json'),no_upload=True,no_training_updates=True)
        r.write(REPORT/'result.json',result);r.write(OUT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:r.write(OUT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:base.release_memory();signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
