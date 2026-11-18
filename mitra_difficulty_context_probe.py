"""Fixed fold0 hard-support sampling vs layer-matched uniform half context."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):raise RuntimeError('Fresh FALSE process required')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
import torch
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import error_rescue_feature_probe as helper
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/mitra_difficulty_context'
OUT=ROOT/'artifacts/mitra_difficulty_context'
PLAN=REPORT/'protocol.json'
SEED=20261112
BUDGET=1200

def sample_support(frame,y,inner_p,weighted):
    """Only outertrain attrs/y/strict inner probabilities; no query inputs or y."""
    y=np.asarray(y);p=np.asarray(inner_p)
    if len(y)!=len(frame) or p.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid support input')
    keys=np.array([json.dumps([int(t),str(h),str(c)],separators=(',',':')) for t,h,c in zip(y,frame.HomePlanet,frame.CryoSleep)])
    levels=sorted(set(keys));members={k:np.flatnonzero(keys==k) for k in levels};quota={k:len(v)//2 for k,v in members.items()}
    remaining=len(y)//2-sum(quota.values())
    for k in [k for k in levels if len(members[k])%2][:remaining]:quota[k]+=1
    difficulty=1+2*np.abs(y-p);rng=np.random.default_rng(SEED);selected=[];summary=[]
    for k in levels:
        idx=members[k];w=difficulty[idx] if weighted else np.ones(len(idx));n=quota[k]
        chosen=rng.choice(idx,size=n,replace=False,p=w/w.sum());selected.extend(chosen.tolist())
        summary.append(dict(stratum=json.loads(k),available=len(idx),selected=n,weight_mean=float(difficulty[idx].mean()),selected_weight_mean=float(difficulty[chosen].mean()) if n else None))
    selected=np.sort(np.asarray(selected,dtype=int))
    if len(selected)!=len(y)//2 or len(np.unique(selected))!=len(selected):raise RuntimeError('Half sample failed')
    return selected,summary

def checks():
    x=pd.DataFrame(dict(HomePlanet=['Earth']*6+['Mars']*5,CryoSleep=['True']*11));y=np.array([0,0,0,1,1,1,0,0,1,1,1]);p=np.full(11,.5)
    u,s=sample_support(x,y,p,False);w,t=sample_support(x,y,p,True)
    np.testing.assert_array_equal(u,w);assert len(u)==5
    u2,_=sample_support(x,y,p,False);np.testing.assert_array_equal(u,u2)
    assert [(r['stratum'],r['selected']) for r in s]==[(r['stratum'],r['selected']) for r in t]

def inputs_and_samples():
    inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];a,b=splits[0]
    frozen=r.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json');r.nested.verify_hashes(frozen['sources']);evidence.update(frozen['sources'])
    inner=r.load_inner(0,a,b,dev,y,evidence);frame=inputs['features'].loc[dev[a]]
    selected={};summary={}
    for name,w in [('uniform',False),('difficulty',True)]:selected[name],summary[name]=sample_support(frame,y[a],inner,w)
    assert len(a)==4636 and len(b)==2318 and all(len(v)==2318 for v in selected.values())
    assert [x['selected'] for x in summary['uniform']]==[x['selected'] for x in summary['difficulty']]
    for idx in selected.values():assert not np.intersect1d(dev[a[idx]],dev[b]).size
    return inputs,a,b,folds,bases,evidence,inner,selected,summary

def run(check=False):
    checks();plan=r.nested.read(PLAN)
    if plan['seed']!=SEED or plan['support_rows']!=2318 or plan['budget_seconds']!=BUDGET:raise RuntimeError('Protocol mismatch')
    previous,lock=counts.verify_reference();inputs,a,b,folds,bases,evidence,inner,selected,strata=inputs_and_samples()
    check_result=dict(status='CPU_membership_and_inner_provenance_verified_no_model_execution',training_pool=4636,query_rows=2318,support_rows=2318,
        shared_selected=int(np.intersect1d(selected['uniform'],selected['difficulty']).size),strata=strata,
        sample_ids_sha256={k:base.ids_hash(inputs['dev'][a[v]]) for k,v in selected.items()})
    if check:r.write(REPORT/'cpu_check.json',check_result);print(json.dumps({k:v for k,v in check_result.items() if k!='strata'}));return
    if OUT.exists() or (REPORT/'result.json').exists():raise RuntimeError('No overwrite/resume')
    names=set(helper.OWN)|{Path(__file__).name,str(PLAN.relative_to(ROOT)),'relational_error_corrector.py','error_overlap_diagnostic.py','planet_mixture_experiment.py'}
    names.update(n for n in previous['source_hashes'] if n.startswith('third_party/mitra/'))
    commit=helper.committed(names);sources={n:r.sha(ROOT/n) for n in names};sources.update(evidence)
    catalog_path=ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv'
    if r.sha(catalog_path)!=plan['catalog_rows_sha256']:raise RuntimeError('Historicalhardmask changed')
    sources[str(catalog_path.relative_to(ROOT))]=r.sha(catalog_path)
    frozen=dict(source_commit=commit,sources=sources,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),protocol=plan)
    OUT.mkdir(parents=True);r.write(OUT/'freeze.json',frozen);r.write(REPORT/'freeze.json',frozen)
    start=time.monotonic();base.DEADLINE=start+BUDGET
    def timeout(*_):raise TimeoutError('Fixed1200s no fallback')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    try:
        if not torch.backends.mps.is_available():raise RuntimeError('OriginalMPS required')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65)
        dev,y=inputs['dev'],inputs['y'];features=inputs['features'];raw=bases['rawMitra'][b];pred={'raw_full':raw.copy()};logs={};ids={}
        for name in ['uniform','difficulty']:
            helper.verify_sources(frozen);idx=selected[name];sid=dev[a[idx]];qid=dev[b];predictor=None;t=time.monotonic()
            try:
                predictor=base.fit_mitra(features.loc[sid],y[a[idx]],sid,SEED,'mps',recipe='zero-shot',query_chunk=2318,log_path=OUT/f'{name}_fit.json')
                if len(predictor.support)!=2318 or predictor.preprocessor.columns!=base.COMPACT or not np.array_equal(predictor.labels,y[a[idx]]):raise RuntimeError('Exact complete selectedsupport mismatch')
                p=np.asarray(predictor.predict_proba(features.loc[qid]))[:,1]
                if p.shape!=(2318,) or not np.isfinite(p).all():raise RuntimeError('Incompletequery')
                pred[name]=p;pred[name+'_half_raw']=.5*p+.5*raw;ids[name+'_support_ids']=sid
                logs[name]=dict(support_rows=2318,query_rows=2318,selected_steps=0,seed=SEED,support_ids_sha256=base.ids_hash(sid),query_ids_sha256=base.ids_hash(qid),
                    actual_features=predictor.preprocessor.n_features,checkpoint_sha256=base.WEIGHT_SHA,seconds=time.monotonic()-t)
                np.savez_compressed(OUT/f'{name}.npz',prob=p,support_ids=sid,query_ids=qid,passenger_ids=np.asarray(inputs['passengers'][b],dtype=str))
            finally:
                if predictor is not None:predictor.release()
                predictor=None;base.release_memory()
        rows=pd.read_csv(catalog_path,dtype={'PassengerId':str})
        for k,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[k],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool)[b];assert hard.sum()==178
        masks=dict(all=np.ones(len(b),bool),hard178=hard,hard292=rows.family_wrong_count.to_numpy()[b]==5,
            Earth_sleep=(features.loc[dev[b]].HomePlanet.eq('Earth')&features.loc[dev[b]].CryoSleep.eq('True')).to_numpy(),Earth_awake=(features.loc[dev[b]].HomePlanet.eq('Earth')&features.loc[dev[b]].CryoSleep.eq('False')).to_numpy())
        def stat(mask):return {k:dict(correct=int((((p>=.5)==y[b])&mask).sum()),vs_raw=r.pair(p,raw,y[b],mask),vs_uniform=r.pair(p,pred['uniform'],y[b],mask),vs_uniform_half=r.pair(p,pred['uniform_half_raw'],y[b],mask)) for k,p in pred.items()}
        result=dict(status='complete_adaptive_single_fold_support_sampling',metrics={k:r.score(p,y[b]) for k,p in pred.items()},subsets={k:stat(m) for k,m in masks.items()},sampling=check_result,models=logs,
            seconds=time.monotonic()-start,source_commit=commit,protocol_sha256=r.sha(PLAN),no_update=True,no_upload=True)
        np.savez_compressed(OUT/'predictions.npz',**pred,**ids,query_ids=dev[b],passenger_ids=np.asarray(inputs['passengers'][b],dtype=str))
        result['predictions_sha256']=r.sha(OUT/'predictions.npz');helper.verify_sources(frozen);base.check_deadline()
        r.write(OUT/'result.json',result);r.write(REPORT/'result.json',result);print(json.dumps(result['metrics']))
    except BaseException as exc:r.write(OUT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:base.release_memory();signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');run(parser.parse_args().check)
