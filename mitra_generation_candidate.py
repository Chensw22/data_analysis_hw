"""Two pinned Mitra generations: unchanged v2, v1, and fixed equal-probability mean.

No fine-tuning, view search, context truncation, or learned ensemble weight.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time

import joblib
import numpy as np
import pandas as pd

import iterate
BASE_FIT, BASE_EVALUATE = iterate.fit, iterate.evaluate
import mitra_candidate as mitra

ROOT=Path(__file__).resolve().parent
CONTROL='v71_Mitra2_zeroshot'
V1='v78_Mitra1_zeroshot'
NAME='v79_Mitra12_mean'
VERSIONS=[V1,NAME]
RECIPES={CONTROL:['v2'],V1:['v1'],NAME:['v1','v2']}
CHECKPOINTS={
    'v1':dict(directory='artifacts/pretrained/mitra-v1',revision='c425e9fa0910a6be1c494321792e7ba2a1367b1a',
        sha256='e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19',bytes=302717904),
    'v2':dict(directory='artifacts/pretrained/mitra-v2',revision='edada0d20759c58ada8c8605c25f22f6e98ea5f0',
        sha256=mitra.WEIGHT_SHA,bytes=mitra.WEIGHT_BYTES)}
SETTINGS=dict(columns=list(mitra.COMPACT),device='mps',query_chunk=2318,
    threshold=.5,mps_memory_fraction=.65,weights_dtype='float32',mps_autocast_dtype='bfloat16',
    full_support=True,optimizer_steps=0,n_preprocessing_views=1,
    probability_mean_dtype='float64',mean_weights=[.5,.5],
    probe_budget_seconds=1200,screen_budget_seconds=3600,group_budget_seconds=3600,
    stability_budget_seconds=3600,export_budget_seconds=3600)
OWN_SOURCES=['mitra_generation_candidate.py','test_mitra_generation_candidate.py',
    'reports/top5_mitra_v1_review.json','top5_generation_evaluate.py','test_top5_generation_evaluate.py',
    'reports/top5_generation_validation_review.json','reports/top5_generation_authorization.json',
    'reports/mitra_v1_checkpoint.json','third_party/mitra_v1/MODEL_CARD.md',
    'third_party/mitra_v1/LICENSE','third_party/mitra_v1/NOTICE',
    'top5_group_evaluate.py','test_top5_group_evaluate.py',
    'audit_top5_references.py','test_audit_top5_references.py',
    'top5_validate.py','test_top5_validate.py',
    'reports/top5_group_validation_review.json','reports/top5_group_authorization.json',
    'reports/top5_validation_review.json','reports/top5_authorization.json']
SHAPES={'small':(128,32),'outer':(4636,2318),'final':(8693,2318)}
RUN_OUT=None
PROBABILITY_CACHE={}


def write_json(path,value):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def verify_checkpoint(generation):
    if generation not in CHECKPOINTS:raise ValueError(generation)
    spec=CHECKPOINTS[generation];directory=ROOT/spec['directory']
    if generation=='v2':return mitra.verify_checkpoint(directory)
    weights=directory/'model.safetensors'
    if not weights.is_file() or weights.stat().st_size!=spec['bytes'] or mitra.sha256(weights)!=spec['sha256']:
        raise RuntimeError('Pinned Mitra-v1 checkpoint missing/wrong SHA or size; no download/fallback.')
    config=json.loads((directory/'config.json').read_text())
    if config!=mitra.MODEL_CONFIG:raise RuntimeError('Mitra-v1 architecture differs.')
    return config


def fit_component(generation,x,y,ids,seed,device):
    verify_checkpoint(generation)
    if generation=='v2':
        return mitra.fit_mitra(x,y,ids,seed,device,'zero-shot',
            checkpoint_dir=ROOT/CHECKPOINTS['v2']['directory'],query_chunk=2318)
    mitra.check_deadline();mitra.seed_everything(seed,device)
    prep=mitra.MitraPreprocessor().fit(x,y,seed)
    config=verify_checkpoint('v1')
    state=mitra.load_file(str(ROOT/CHECKPOINTS['v1']['directory']/'model.safetensors'),device='cpu')
    try:
        return mitra.MitraPredictor(prep,config,state,prep.transform(x),y,device=device,query_chunk=2318,
            metadata=dict(generation='v1',seed=int(seed),optimizer_steps=0,
                full_support_rows=len(x),training_ids_sha256=mitra.ids_hash(ids),
                checkpoint_sha256=CHECKPOINTS['v1']['sha256']))
    finally:
        del state;gc.collect()


def check_probabilities(p,rows):
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(rows,2) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise FloatingPointError('Incomplete or invalid component probabilities.')
    np.testing.assert_allclose(p.sum(axis=1),1,rtol=0,atol=1e-6)
    return p


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()


class GenerationPredictor:
    """Portable raw CPU context plus SHA-pinned recipes, never resident engines.

    Each predict rebuilds every required generation sequentially. Cache reuse
    is an explicit fit-only optimization, disabled by default and on reload.
    """
    def __init__(self,x,y,ids,seed,version,device='mps'):
        if version not in RECIPES:raise ValueError(version)
        self.training=x[mitra.COMPACT].copy(deep=True)
        self.labels=mitra.binary_labels(y,len(x))
        self.train_ids=np.asarray(ids,dtype=np.int64).copy()
        if len(self.train_ids)!=len(x) or len(np.unique(self.train_ids))!=len(x):
            raise ValueError('Unique aligned training IDs required.')
        self.seed=int(seed);self.version=version;self.device=device
        self.generations=list(RECIPES[version]);self.checkpoints=copy.deepcopy(CHECKPOINTS)
        self.query_chunk=2318;self.metadata={}

    def validate(self):
        if self.generations!=RECIPES.get(self.version) or self.checkpoints!=CHECKPOINTS or self.query_chunk!=2318:
            raise RuntimeError('Serialized generation/weight recipe changed.')
        if list(self.training)!=list(mitra.COMPACT) or len(self.labels)!=len(self.train_ids) or len(self.training)!=len(self.labels):
            raise RuntimeError('Serialized context/schema changed.')

    def cache_key(self,generation,query,query_ids):
        return (generation,self.checkpoints[generation]['sha256'],self.seed,self.device,
            mitra.ids_hash(self.train_ids),mitra.ids_hash(query_ids),
            hashlib.sha256(self.labels.tobytes()).hexdigest(),frame_hash(self.training),frame_hash(query))

    def predict_proba(self,x,prediction_ids=None,*,use_cache=False,log_path=None,prediction_path=None):
        self.validate()
        query=x[mitra.COMPACT]
        if prediction_ids is not None:
            prediction_ids=np.asarray(prediction_ids,dtype=np.int64)
            if (len(prediction_ids)!=len(query) or len(np.unique(prediction_ids))!=len(query)
                    or np.intersect1d(prediction_ids,self.train_ids).size):
                raise ValueError('Aligned unique query IDs disjoint from training are required.')
        elif use_cache:
            raise ValueError('Fit-only cache requires explicit query IDs.')
        self.metadata=dict(status='incomplete',version=self.version,seed=self.seed,
            generations=self.generations,weights=[1/len(self.generations)]*len(self.generations),
            checkpoints=self.checkpoints,full_support_rows=len(self.training),query_rows=len(query),
            training_ids_sha256=mitra.ids_hash(self.train_ids),
            prediction_ids_sha256=mitra.ids_hash(prediction_ids) if prediction_ids is not None else None,
            input_columns=list(mitra.COMPACT),query_chunk=2318,components=[])
        started=time.monotonic();predictions=[]
        try:
            for generation in self.generations:
                mitra.check_deadline();verify_checkpoint(generation)
                key=self.cache_key(generation,query,prediction_ids) if use_cache else None
                tick=time.monotonic();component=None
                try:
                    if key is not None and key in PROBABILITY_CACHE:
                        p,detail=PROBABILITY_CACHE[key]
                        p=p.copy();detail=copy.deepcopy(detail);detail['cache_hit']=True
                    else:
                        component=fit_component(generation,self.training,self.labels,self.train_ids,self.seed,self.device)
                        if len(component.support)!=len(self.training):raise RuntimeError('Truncated generation support.')
                        p=check_probabilities(component.predict_proba(query),len(query))
                        detail=dict(generation=generation,cache_hit=False,
                            checkpoint_sha256=self.checkpoints[generation]['sha256'],
                            full_support_rows=len(component.support),
                            preprocessing_mirror=np.asarray(component.preprocessor.upstream.mirror).reshape(-1).tolist(),
                            fit_metadata=copy.deepcopy(component.metadata))
                        if key is not None:PROBABILITY_CACHE[key]=(p.copy(),copy.deepcopy(detail))
                    p=check_probabilities(p,len(query))
                    predictions.append(p.copy())
                finally:
                    if component is not None:component.release()
                    del component;gc.collect();mitra.release_memory()
                detail.update(seconds=time.monotonic()-tick,resources_after_release=mitra.snapshot(self.device),
                    probabilities_sha256=hashlib.sha256(p.tobytes()).hexdigest())
                self.metadata['components'].append(detail);write_json(log_path,self.metadata)
            if len(predictions)!=len(self.generations):raise RuntimeError('No partial generation ensemble.')
            stacked=np.stack(predictions);mean=stacked.mean(axis=0,dtype=np.float64)
            mitra.check_deadline()
            self.metadata.update(status='complete',completed_components=len(predictions),seconds=time.monotonic()-started,
                probability_sha256=hashlib.sha256(mean.tobytes()).hexdigest())
            if prediction_path is not None:
                np.savez(prediction_path,component_probabilities=stacked,prob=mean[:,1],generations=self.generations,
                    train_ids=self.train_ids,prediction_ids=prediction_ids if prediction_ids is not None else [])
            write_json(log_path,self.metadata);return mean
        except Exception as error:
            self.metadata.update(status='failed_incomplete',completed_components=len(predictions),error=repr(error))
            write_json(log_path,self.metadata);raise

    def __getstate__(self):
        self.validate();return dict(self.__dict__)

    def __setstate__(self,state):
        self.__dict__.update(state);self.validate()


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name not in RECIPES:raise ValueError(name)
    if RUN_OUT is None:raise RuntimeError('Configure caller-owned fit log destination first.')
    a,b=np.asarray(train_idx,dtype=np.int64),np.asarray(predict_idx,dtype=np.int64)
    x=matrices[False][0]
    artifact=GenerationPredictor(x.iloc[a],y,a,seed,name)
    key=mitra.ids_hash(a)[:12];path=RUN_OUT/f'{name}_seed{seed}_{key}.json'
    if path.exists():raise RuntimeError('Existing fit log; no implicit overwrite.')
    p=artifact.predict_proba(x.iloc[b],b,use_cache=True,log_path=path,
        prediction_path=path.with_name(path.stem+'_views.npz'))
    return p[:,1],artifact


def source_hashes():
    reference=json.loads((ROOT/'artifacts/run_038/protocol.json').read_text())
    for path,digest in reference['sources'].items():
        if mitra.sha256(ROOT/path)!=digest:raise RuntimeError('Frozen v2 dependency changed: '+path)
    lock=mitra.environment_lock()
    if hashlib.sha256(lock.encode()).hexdigest()!=reference['mitra_environment_sha256']:
        raise RuntimeError('Use original frozen Mitra environment.')
    provenance=json.loads((ROOT/'reports/mitra_v1_checkpoint.json').read_text())
    expected=CHECKPOINTS['v1']
    if (provenance['repository']!='autogluon/mitra-classifier' or provenance['revision']!=expected['revision']
            or provenance['files']['model.safetensors']!={'sha256':expected['sha256'],'bytes':expected['bytes']}):
        raise RuntimeError('Mitra-v1 provenance differs from the fixed checkpoint.')
    for filename in ['config.json','README.md']:
        if mitra.sha256(ROOT/expected['directory']/filename)!=provenance['files'][filename]['sha256']:
            raise RuntimeError('Mitra-v1 config/model-card identity differs.')
    for generation in CHECKPOINTS:verify_checkpoint(generation)
    return {path:mitra.sha256(ROOT/path) for path in sorted(set(reference['sources'])|set(OWN_SOURCES))}


def probe_path(shape):return ROOT/'artifacts/probes'/f'mitra_generations_{shape}'


def require_resources():
    sources=source_hashes();env=hashlib.sha256(mitra.environment_lock().encode()).hexdigest();records={}
    for shape,(n,q) in SHAPES.items():
        path=probe_path(shape)/'result.json';value=json.loads(path.read_text())
        if (value.get('status')!='complete' or value.get('support_rows')!=n or value.get('query_rows')!=q
            or value.get('settings')!=SETTINGS or value.get('checkpoints')!=CHECKPOINTS
            or value.get('source_hashes')!=sources or value.get('environment_sha256')!=env
            or value.get('full_support') is not True or value.get('subset_passed') is not True
            or value.get('reload_passed') is not True or value.get('completed_generations')!=['v1','v2']):
            raise RuntimeError('Complete unchanged generation resource probes required: '+shape)
        records[shape]=dict(path=str(path.relative_to(ROOT)),sha256=mitra.sha256(path))
    return records


def record_protocol(out):
    path=out/'protocol.json';record=json.loads(path.read_text())
    if 'generation_policy' in record:return
    evidence=require_resources();sources=source_hashes()
    for name,digest in sources.items():
        record['sources'][name]=digest;target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    lock=mitra.environment_lock();(out/'generation_environment.lock.txt').write_text(lock)
    record.update(generation_policy=dict(settings=SETTINGS,recipes=RECIPES,checkpoints=CHECKPOINTS,
        resources=evidence,environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        eligibility='Both new recipes remain in the separately frozen group queue; no winner-only promotion.'))
    write_json(path,record)


def evaluate(name,matrices,train,dev,seed,out):
    record_protocol(out);return BASE_EVALUATE(name,matrices,train,dev,seed,out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:return
    if iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE or BASE_FIT.__module__!='iterate':
        raise RuntimeError('Fresh process with original iterate baseline required.')
    iterate.CONFIGS={iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT)}
    for version,generations in RECIPES.items():
        iterate.CONFIGS[version]=dict(context=False,model='mitra_generation',generations=generations,
            weights=[1/len(generations)]*len(generations),settings=SETTINGS,checkpoints=CHECKPOINTS)
    iterate.BLEND=[];iterate.fit,iterate.evaluate=fit,evaluate


def configure_runtime(out,stage='screen'):
    global RUN_OUT
    if stage not in ['screen','group','stability','export','probe']:raise ValueError(stage)
    configure();RUN_OUT=Path(out)
    if not mitra.torch.backends.mps.is_available():raise RuntimeError('MPS required; no fallback.')
    mitra.torch.set_num_threads(4);mitra.torch.mps.set_per_process_memory_fraction(.65)
    mitra.DEADLINE=time.monotonic()+SETTINGS[f'{stage}_budget_seconds']


def assert_frozen(out):
    out=Path(out);iterate.assert_frozen(out)
    policy=json.loads((out/'protocol.json').read_text())['generation_policy']
    if (policy['settings']!=SETTINGS or policy['recipes']!=RECIPES or policy['checkpoints']!=CHECKPOINTS
        or policy['environment_sha256']!=hashlib.sha256(mitra.environment_lock().encode()).hexdigest()
        or mitra.sha256(out/'generation_environment.lock.txt')!=policy['environment_sha256']
        or policy['resources']!=require_resources()):raise RuntimeError('Generation freeze differs.')


def resource_probe(shape):
    out=probe_path(shape);out.mkdir(parents=True,exist_ok=False)
    n,q=SHAPES[shape];result=dict(status='incomplete',support_rows=n,query_rows=q,
        settings=SETTINGS,checkpoints=CHECKPOINTS,full_support=True,synthetic_labels=True,no_accuracy_computed=True)
    started=time.monotonic();artifact=restored=None
    try:
        result['source_hashes']=source_hashes();lock=mitra.environment_lock()
        (out/'environment.lock.txt').write_text(lock)
        result['environment_sha256']=hashlib.sha256(lock.encode()).hexdigest()
        for name in result['source_hashes']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        x=mitra.synthetic_frame(n,42);query=mitra.synthetic_frame(q,43)
        ids=np.arange(n);queryids=np.arange(n,n+q);y=ids%2
        artifact=GenerationPredictor(x,y,ids,42,NAME)
        result['resources_before']=mitra.snapshot('mps')
        tick=time.monotonic();p=artifact.predict_proba(query,queryids,log_path=out/'predict.json',prediction_path=out/'components.npz')
        result['full_forward_seconds']=time.monotonic()-tick
        detail=copy.deepcopy(artifact.metadata)
        count=min(128,q);small=artifact.predict_proba(query.iloc[:count],queryids[:count],
            prediction_path=out/'subset_components.npz')
        np.testing.assert_allclose(small,p[:count],atol=2e-6,rtol=0)
        with np.load(out/'components.npz') as saved,np.load(out/'subset_components.npz') as subset:
            np.testing.assert_allclose(subset['component_probabilities'],saved['component_probabilities'][:,:count],atol=2e-6,rtol=0)
        result['subset_passed']=True
        joblib.dump(artifact,out/'model.joblib');artifact=None;gc.collect();mitra.release_memory()
        restored=joblib.load(out/'model.joblib');restored.device='mps'
        again=restored.predict_proba(query,queryids,log_path=out/'reload.json',
            prediction_path=out/'reload_components.npz')
        np.testing.assert_allclose(again,p,atol=2e-6,rtol=0)
        np.testing.assert_array_equal(again[:,1]>=.5,p[:,1]>=.5)
        with np.load(out/'components.npz') as saved,np.load(out/'reload_components.npz') as reloaded:
            first_components=saved['component_probabilities'];last_components=reloaded['component_probabilities']
            np.testing.assert_allclose(last_components,first_components,atol=2e-6,rtol=0)
            np.testing.assert_array_equal(last_components[:,:,1]>=.5,first_components[:,:,1]>=.5)
        mitra.check_deadline()
        result.update(status='complete',reload_passed=True,completed_generations=['v1','v2'],
            reload_max_difference=float(np.max(np.abs(p-again))),full_prediction_metadata=detail,
            resources_after=mitra.snapshot('mps'),seconds=time.monotonic()-started)
        write_json(out/'result.json',result);return result
    except Exception as error:
        result.update(status='failed_inconclusive',error=repr(error));write_json(out/'result.json',result);raise
    finally:
        artifact=restored=None;gc.collect();mitra.release_memory()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['probe','screen']);parser.add_argument('--run',default='run_044')
    parser.add_argument('--shape',choices=list(SHAPES),default='small');args=parser.parse_args()
    out=ROOT/'artifacts'/args.run;existed=out.exists()
    configure_runtime(probe_path(args.shape) if args.stage=='probe' else out,args.stage)
    def timeout(*_):raise TimeoutError('Generation complete-stage budget exceeded.')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,SETTINGS[f'{args.stage}_budget_seconds'])
    try:
        if args.stage=='probe':resource_probe(args.shape)
        else:
            require_resources();iterate.screen(out);assert_frozen(out)
            record=json.loads((out/'selection.json').read_text())
            record.update(stage='both_new_recipes_pending_fixed_queue_validation',automatic_eligibility=False,
                retained_candidates=VERSIONS)
            write_json(out/'selection.json',record)
    except Exception as error:
        if args.stage=='screen' and not existed and out.exists():
            (out/'selection.json').unlink(missing_ok=True)
            if (out/'screen_results.csv').exists():(out/'screen_results.csv').replace(out/'screen_results.partial.csv')
            write_json(out/'failure.json',dict(status='failed_inconclusive',error=repr(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);mitra.DEADLINE=None


if __name__=='__main__':
    sys.modules['mitra_generation_candidate']=sys.modules[__name__]
    GenerationPredictor.__module__='mitra_generation_candidate';main()
