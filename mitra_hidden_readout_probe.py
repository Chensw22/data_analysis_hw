"""Frozen query-label-token representations, strictly cross-fitted supervised readout."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):raise RuntimeError('Fresh FALSE process required')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import json
from pathlib import Path
import hashlib
import signal
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import error_rescue_feature_probe as helper
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/mitra_hidden_readout_probe'
OUT=ROOT/'artifacts/mitra_hidden_readout_probe'
BUDGET=1800


@torch.no_grad()
def extract(features, train_ids, labels, query_ids, seed, key):
    """Hook records only query label token; replays the unchanged native head."""
    if np.intersect1d(train_ids,query_ids).size:raise ValueError('Representation cannot see its own label')
    predictor=model=handle=None
    started=time.monotonic();capture={}
    try:
        predictor=base.fit_mitra(features.loc[train_ids],labels,train_ids,seed,'mps',recipe='zero-shot',query_chunk=2318,log_path=OUT/(key+'_fit.json'))
        model=predictor.engine();model.eval()
        def capture_head(module, args, output):
            # Inside the original autocast context, same full tensor shape and weights.
            replay=F.linear(args[0],module.weight,module.bias)
            if not torch.equal(output,replay):raise RuntimeError('Native head replay differs')
            capture['embedding']=args[0][0,:,0,:].detach().float().cpu().numpy().copy()
            capture['logits']=output[0,:,0,:].detach().float().cpu().numpy().copy()
        handle=model.final_layer.register_forward_hook(capture_head)
        support=torch.as_tensor(predictor.support,device='mps')
        target=torch.as_tensor(predictor.labels,device='mps')
        query=torch.as_tensor(predictor.preprocessor.transform(features.loc[query_ids]),device='mps')
        logits=base.model_forward(model,support,target,query)
        np.testing.assert_array_equal(logits.cpu().numpy(),capture['logits'])
        p=logits[:,:2].softmax(-1).cpu().numpy()[:,1].copy()
        h=capture['embedding']
        if h.shape!=(len(query_ids),512) or not np.isfinite(h).all():raise RuntimeError('Wrong finite representation shape')
        meta=dict(seed=seed,train_rows=len(train_ids),query_rows=len(query_ids),train_ids_sha256=base.ids_hash(train_ids),
            query_ids_sha256=base.ids_hash(query_ids),head_replay_all10_exact=True,hidden_dimension=512,
            checkpoint_sha256=base.WEIGHT_SHA,selected_steps=0,seconds=time.monotonic()-started)
        np.savez_compressed(OUT/(key+'.npz'),embedding=h,prob=p,logits=capture['logits'],train_ids=train_ids,query_ids=query_ids)
        meta['packet_sha256']=r.sha(OUT/(key+'.npz'))
        return h,p,meta
    finally:
        if handle is not None:handle.remove()
        capture.clear()
        model=None
        if predictor is not None:predictor.release()
        predictor=None;base.release_memory()


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('No overwrite/resume')
    previous,lock=counts.verify_reference()
    inputs,splits,folds,bases,evidence=overlap.load_authenticated()
    dev,y=inputs['dev'],inputs['y'];features=inputs['features'];raw=bases['rawMitra']
    oldfreeze=r.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
    r.nested.verify_hashes(oldfreeze['sources']);evidence.update(oldfreeze['sources'])
    inner_cached=[r.load_inner(i,a,b,dev,y,evidence) for i,(a,b) in enumerate(splits)]
    catalog=ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv';evidence[str(catalog.relative_to(ROOT))]=r.sha(catalog)
    names=set(helper.OWN)|{Path(__file__).name,'error_overlap_diagnostic.py','relational_error_corrector.py'}
    names.update(n for n in previous['source_hashes'] if n.startswith('third_party/mitra/'))
    sources={n:r.sha(ROOT/n) for n in names};sources.update(evidence)
    REPORT.mkdir(parents=True);OUT.mkdir(parents=True)
    plan=dict(status='frozen_before_any_new_representation_or_readout',representation='Official Tab2D post-final_layer_norm, pre-final_layer, query label token index0; 512dimensions. No backbone updates.',
        head_verification='Every forward: F.linear full native input inside same autocast reproduces all10 native logits exactly; returned query logits equal head token0.',
        data='Original6954dev only targets, original3 SKF seed20261112; no old1739 targets.',
        nested='For outer i, inner3 SKF seed20261117+i, model seeds20261117+100*i+j; matches authenticated cached raw inner predictions. Each row is absent from its own context and preprocessing fit.',
        outer='Full4636 outertrain support,2318queries, modelseed20261112+outer, compact13. Require fresh probability exactly matches authenticated rawOOF.',
        readout='StandardScaler fit on 4636 nestedOOF512vectors only; fixed raw logit offset. mean binary NLL +0.1*sum(w**2), free intercept. L-BFGS-B500/ftol1e-12/gtol1e-8, zero init; no hyperparameter search.',
        control='Same nested raw offsets, intercept only, identical objective and optimizer.',
        prediction='sigmoid(logit(raw_outer)+intercept+standardized_outer_embedding@w), threshold0.5',
        precision='Original MPS bf16 autocast, FP32 weights/input; 0.65memory, CPU4',budget_seconds=BUDGET,
        execution='Fold0 first, then same frozen recipe folds1/2; no selection based on fold0.',
        limitations=['Adaptive reuseddev, not independent validation.','Training representations use about3090 support; validation4636; context distribution shift remains.','Representation is supervised by other support labels, never own/query labels.','546mask is posthoc reporting only; not used to train or select rows.'])
    r.write(REPORT/'protocol.json',plan);sources[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json')
    frozen=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),protocol=plan)
    r.write(OUT/'freeze.json',frozen);r.write(REPORT/'freeze.json',frozen)
    start=time.monotonic();base.DEADLINE=start+BUDGET
    def timeout(*_):raise TimeoutError('Hidden readout1800s')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    predictions=dict(raw=raw.copy(),constant=np.full(len(y),np.nan),hidden=np.full(len(y),np.nan));records=[]
    try:
        if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        for outer,(a,b) in enumerate(splits):
            helper.verify_sources(frozen)
            inner=StratifiedKFold(3,shuffle=True,random_state=20261117+outer)
            h=np.full((len(a),512),np.nan);p=np.full(len(a),np.nan);coverage=np.zeros(len(a),int);jobs=[]
            for j,(ia,ib) in enumerate(inner.split(a,y[a])):
                if np.intersect1d(dev[b],np.r_[dev[a[ia]],dev[a[ib]]]).size:raise RuntimeError('Outer labels entered inner support/query')
                hh,pp,meta=extract(features,dev[a[ia]],y[a[ia]].copy(),dev[a[ib]],20261117+100*outer+j,f'outer{outer}_inner{j}')
                np.testing.assert_array_equal(pp,inner_cached[outer][ib]);h[ib]=hh;p[ib]=pp;coverage[ib]+=1;jobs.append(meta)
                print(f'outer{outer} inner{j} representation verified',flush=True)
            if not np.all(coverage==1):raise RuntimeError('OOF coverage not exactly once')
            hq,pq,meta=extract(features,dev[a],y[a].copy(),dev[b],20261112+outer,f'outer{outer}_query')
            np.testing.assert_array_equal(pq,raw[b])
            mean=h.mean(0);scale=h.std(0);scale[scale==0]=1.;z=(h-mean)/scale;zq=(hq-mean)/scale
            theta,fit=r.fit_corrector(p,y[a],z)
            constant,controlfit=r.fit_corrector(p,y[a],np.empty((len(a),0)))
            predictions['hidden'][b]=expit(r.logit(pq)+theta[0]+zq@theta[1:])
            predictions['constant'][b]=expit(r.logit(pq)+constant[0])
            package=dict(mean=mean,scale=scale,theta=theta,constant=constant,penalty=.1,checkpoint_sha256=base.WEIGHT_SHA,representation=plan['representation'])
            path=OUT/f'fold{outer}_readout.joblib';joblib.dump(package,path);loaded=joblib.load(path)
            np.testing.assert_array_equal(predictions['hidden'][b],expit(r.logit(pq)+loaded['theta'][0]+((hq-loaded['mean'])/loaded['scale'])@loaded['theta'][1:]))
            np.testing.assert_array_equal(predictions['constant'][b],expit(r.logit(pq)+loaded['constant'][0]))
            record=dict(fold=outer,inner_jobs=jobs,outer_job=meta,fit=fit,constant_fit=controlfit,readout_sha256=r.sha(path),readout_reload_max_diff=0.,raw_cached_exact=True,
                metrics={k:r.score(v[b],y[b]) for k,v in predictions.items()},paired={k:r.pair(predictions['hidden'][b],predictions[k][b],y[b]) for k in ['raw','constant']})
            records.append(record);r.write(REPORT/f'fold{outer}.json',record)
            print(json.dumps(dict(fold=outer,metrics=record['metrics'],paired=record['paired'])),flush=True)
        rows=pd.read_csv(catalog,dtype={'PassengerId':str})
        for key,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[key],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        masks=dict(all=np.ones(len(y),bool),hard546=hard,hard914=rows.family_wrong_count.to_numpy()==5,commoncorrect5263=rows.family_wrong_count.to_numpy()==0)
        def summary(mask):return {k:dict(correct=int((((v>=.5)==y)&mask).sum()),vs_raw=r.pair(v,raw,y,mask),vs_constant=r.pair(v,predictions['constant'],y,mask)) for k,v in predictions.items()}
        for i,record in enumerate(records):record['hard546']=summary(hard&(folds==i));r.write(REPORT/f'fold{i}.json',record)
        cache=OUT/'oof.npz';np.savez_compressed(cache,**predictions,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        helper.verify_sources(frozen);base.check_deadline()
        result=dict(status='complete',metrics={k:r.score(v,y) for k,v in predictions.items()},subsets={k:summary(m) for k,m in masks.items()},folds=records,seconds=time.monotonic()-start,
            source_hashes=sources,oof_sha256=r.sha(cache),no_upload=True,backbone_updates=0,all12_head_replays_exact=True,all12_cached_raw_probabilities_exact=True)
        r.write(REPORT/'result.json',result);r.write(OUT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:
        r.write(OUT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:base.release_memory();signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
