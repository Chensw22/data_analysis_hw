"""Fixed Mitra20-column unlabeled-roster bundle, with untouched13-column control.

No target enters roster construction. Group validation is transductive: withheld
attributes are available but withheld targets never enter fitting or preprocessing.
"""
import argparse
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

ROOT = Path(__file__).resolve().parent
NAME, CONTROL = 'v77_Mitra_context20', 'v71_Mitra2_zeroshot'
EXTRA = ['GroupSize', 'SurnameCount', 'CabinOccupants', 'PeerCryoSleepRate',
         'PeerKnownCryoCount', 'PeerMeanAge', 'GroupChildren']
COLUMNS = list(mitra.COMPACT)+EXTRA
ATTRIBUTES = ['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','Age','VIP',
              'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Name']
SETTINGS = dict(columns=COLUMNS, categorical_columns=list(mitra.CATEGORICAL),
    query_chunk=2318, device='mps', mps_memory_fraction=.65,
    weights_dtype='float32', mps_autocast_dtype='bfloat16',
    full_support=True, n_estimators=1, optimizer_steps=0, threshold=.5,
    screen_budget_seconds=1800, stability_budget_seconds=3600,
    probe_budget_seconds=1200, export_budget_seconds=3600,
    checkpoint_sha256=mitra.WEIGHT_SHA,
    roster_policy='All12970 unlabelled attribute rows; fixed precomputed context, no target propagation')
OWN_SOURCES = ['mitra_context_candidate.py','test_mitra_context_candidate.py',
    'reports/top5_mitra_context_review.json','top5_validate.py','test_top5_validate.py',
    'audit_top5_references.py','test_audit_top5_references.py',
    'reports/top5_validation_review.json','reports/top5_authorization.json']
PROBES = {key:ROOT/'artifacts/probes'/f'mitra_context20_{key}' for key in ['outer','final']}
RUN_OUT = None
ROSTER_CACHE = None


def write_json(path, value):
    if path is None:
        return
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def frame_hash(frame):
    h=hashlib.sha256()
    h.update(json.dumps(list(frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
    return h.hexdigest()


def build_roster(roster):
    """Pure attribute transform: discard target before either feature builder."""
    raw=roster[ATTRIBUTES].reset_index(drop=True).copy()
    if raw.PassengerId.isna().any() or not raw.PassengerId.is_unique:
        raise ValueError('Roster PassengerIds must be nonmissing and unique.')
    plain=iterate.build_features(raw,False)[0]
    context=iterate.build_features(raw,True)[0][COLUMNS].copy()
    return dict(plain=plain, context=context, passenger_ids=raw.PassengerId.astype(str).to_numpy(),
                attribute_sha256=frame_hash(raw), plain_sha256=frame_hash(plain),
                context_sha256=frame_hash(context))


def load_roster():
    global ROSTER_CACHE
    if ROSTER_CACHE is None:
        # Explicit usecols: pandas never parses Transported, including old holdout labels.
        train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=ATTRIBUTES)
        test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=ATTRIBUTES)
        if len(train)!=8693 or len(test)!=4277:
            raise RuntimeError('Full roster shape changed.')
        ROSTER_CACHE=build_roster(pd.concat([train,test],ignore_index=True))
    return ROSTER_CACHE


def aligned_context(plain):
    roster=load_roster()
    # The validator supplies the same full False matrix. Verify every row/value,
    # including RangeIndex, rather than relying on an unchecked global ID map.
    pd.testing.assert_frame_equal(plain,roster['plain'],check_exact=True)
    return roster


class ContextPreprocessor(mitra.MitraPreprocessor):
    """Original numerical/category encoding, explicitly fit on20 input columns."""
    def fit(self,x,y,seed=42):
        y=mitra.binary_labels(y,len(x))
        self.columns=list(COLUMNS)
        self.categories=list(mitra.CATEGORICAL)
        self.vocab={}
        for col in self.categories:
            missing=self.category_missing(x[col])
            self.vocab[col]={value:i+2 for i,value in enumerate(sorted(set(x.loc[~missing,col].astype(str))))}
        array=self.encode(x)
        self.all_missing=np.isnan(array).all(axis=0)
        array[:,self.all_missing]=0.
        self.upstream=mitra.Preprocessor(dim_embedding=None,n_classes=2,dim_output=10,
            use_quantile_transformer=False,use_feature_count_scaling=False,
            use_random_transforms=False,shuffle_classes=False,shuffle_features=False,
            random_mirror_regression=False,random_mirror_x=mitra.SETTINGS['random_mirror_x'],
            task=mitra.Task.CLASSIFICATION)
        previous=np.random.get_state()
        try:
            np.random.seed(seed)
            self.upstream.fit(array.copy(),y.copy())
        finally:
            np.random.set_state(previous)
        self.n_features=self.transform(x).shape[1]
        if self.n_features<1:
            raise ValueError('No nonconstant context features.')
        return self


class ContextPredictor(mitra.MitraPredictor):
    """Original Mitra inference, with a persisted fixed-manifest lookup.

    predict_proba accepts the original engineered False frame with its stable
    roster index and checks its values. For ID-based queries use
    predict_proba_by_passenger_ids. New/unseen passengers are not supported.
    Query subsets never recompute peer statistics.
    """
    def __init__(self,*args,roster,**kwargs):
        super().__init__(*args,**kwargs)
        self.roster_plain=roster['plain'].copy(deep=True)
        self.roster_context=roster['context'].copy(deep=True)
        self.passenger_ids=roster['passenger_ids'].copy()
        self.roster_attribute_sha256=roster['attribute_sha256']

    def predict_proba(self,x):
        if not x.index.is_unique or not x.index.isin(self.roster_plain.index).all():
            raise ValueError('Query index must identify unique known manifest rows.')
        pd.testing.assert_frame_equal(x,self.roster_plain.loc[x.index],check_exact=True)
        return super().predict_proba(self.roster_context.loc[x.index])

    def predict_proba_by_passenger_ids(self,ids):
        lookup=pd.Index(self.passenger_ids)
        rows=lookup.get_indexer(np.asarray(ids).astype(str))
        if (rows<0).any() or len(np.unique(rows))!=len(rows):
            raise ValueError('Unknown or repeated manifest PassengerId.')
        return self.predict_proba(self.roster_plain.iloc[rows])


def fit_context(x,y,ids,seed,device,roster):
    ids=np.asarray(ids,dtype=np.int64)
    y=mitra.binary_labels(y,len(x))
    if len(ids)!=len(x) or len(np.unique(ids))!=len(ids):
        raise ValueError('Unique aligned support IDs required.')
    mitra.check_deadline()
    start=time.monotonic()
    mitra.seed_everything(seed,device)
    prep=ContextPreprocessor().fit(x,y,seed)
    config=mitra.verify_checkpoint(mitra.CHECKPOINT_DIR)
    state=mitra.load_file(str(mitra.CHECKPOINT_DIR/'model.safetensors'),device='cpu')
    try:
        support=prep.transform(x)
        metadata=dict(status='fit_complete',candidate=NAME,seed=int(seed),optimizer_steps=0,
            input_columns=COLUMNS,actual_retained_features=prep.n_features,
            requested_support_rows=len(ids),resolved_support_rows=len(support),
            training_ids_sha256=mitra.ids_hash(ids),
            roster_attribute_sha256=roster['attribute_sha256'],
            roster_context_sha256=roster['context_sha256'],
            checkpoint_sha256=mitra.WEIGHT_SHA,query_chunk=2318,
            mirror_mask=np.asarray(prep.upstream.mirror).reshape(-1).tolist(),
            preprocessing_seconds=time.monotonic()-start)
        return ContextPredictor(prep,config,state,support,y,device=device,
            metadata=metadata,query_chunk=2318,roster=roster)
    finally:
        del state
        gc.collect()


def valid_probabilities(p,n):
    p=np.asarray(p)
    if p.shape!=(n,2) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise FloatingPointError('Incomplete/invalid probability matrix.')
    np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-6,rtol=0)
    return p


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name not in [NAME,CONTROL]:
        raise ValueError(name)
    if RUN_OUT is None:
        raise RuntimeError('Configure caller-owned log destination first.')
    a,b=np.asarray(train_idx,dtype=np.int64),np.asarray(predict_idx,dtype=np.int64)
    if len(a)!=len(y) or len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b) or np.intersect1d(a,b).size:
        raise ValueError('Aligned unique disjoint train/query IDs required.')
    x=matrices[False][0]
    roster=aligned_context(x)
    if (a<0).any() or (b<0).any() or (a>=len(x)).any() or (b>=len(x)).any():
        raise ValueError('Invalid row positions.')
    log=RUN_OUT/f'{name}_seed{seed}_{mitra.ids_hash(a)[:12]}.json'
    if log.exists():
        raise RuntimeError('Existing fit log; no implicit overwrite.')
    predictor=None
    started=time.monotonic()
    try:
        if name==CONTROL:
            predictor=mitra.fit_mitra(x.iloc[a],y,a,seed,'mps','zero-shot',
                checkpoint_dir=mitra.CHECKPOINT_DIR,query_chunk=2318)
        else:
            predictor=fit_context(roster['context'].iloc[a],y,a,seed,'mps',roster)
        p=valid_probabilities(predictor.predict_proba(x.iloc[b]),len(b))
        mitra.check_deadline()
        predictor.metadata.update(status='complete',candidate=name,
            prediction_ids_sha256=mitra.ids_hash(b),prediction_rows=len(b),
            training_ids_sha256=mitra.ids_hash(a),full_support_rows=len(a),
            probabilities_sha256=hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
            roster_attribute_sha256=roster['attribute_sha256'],
            seconds=time.monotonic()-started,resources=mitra.snapshot('mps'))
        write_json(log,predictor.metadata)
        return p[:,1],predictor
    except Exception as error:
        write_json(log,dict(status='failed_inconclusive',error=repr(error),candidate=name,
            training_ids_sha256=mitra.ids_hash(a),prediction_ids_sha256=mitra.ids_hash(b)))
        raise
    finally:
        if predictor is not None:
            predictor.release()
        mitra.release_memory()


def verify_reference():
    protocol=json.loads((ROOT/'artifacts/run_038/protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        if mitra.sha256(ROOT/name)!=digest:
            raise RuntimeError(f'Original Mitra source changed: {name}')
    lock=mitra.environment_lock()
    if hashlib.sha256(lock.encode()).hexdigest()!=protocol['mitra_environment_sha256']:
        raise RuntimeError('Frozen Mitra environment differs.')
    mitra.verify_checkpoint(mitra.CHECKPOINT_DIR)
    import top5_validate
    top5_validate.check_policy()
    return protocol,lock


def source_hashes():
    reference,_=verify_reference()
    return {name:mitra.sha256(ROOT/name) for name in sorted(set(reference['sources'])|set(OWN_SOURCES))}


def require_resources():
    sources=source_hashes()
    lock_hash=hashlib.sha256(mitra.environment_lock().encode()).hexdigest()
    roster=load_roster()
    evidence={}
    for shape,rows in [('outer',4636),('final',8693)]:
        path=PROBES[shape]/'result.json'
        result=json.loads(path.read_text())
        if (result.get('status')!='complete' or result.get('support_rows')!=rows
            or result.get('query_rows')!=2318 or result.get('actual_retained_features')!=20
            or result.get('full_support') is not True or result.get('source_hashes')!=sources
            or result.get('settings')!=SETTINGS or result.get('environment_sha256')!=lock_hash
            or result.get('roster_attribute_sha256')!=roster['attribute_sha256']
            or result.get('reload_passed') is not True or result.get('subset_passed') is not True):
            raise RuntimeError(f'Complete matching20-column resource evidence required: {shape}')
        evidence[shape]=dict(path=str(path.relative_to(ROOT)),sha256=mitra.sha256(path))
    return evidence


def record_protocol(out):
    path=out/'protocol.json'
    protocol=json.loads(path.read_text())
    if 'context20_policy' in protocol:
        return
    evidence=require_resources()
    sources=source_hashes()
    for name,digest in sources.items():
        protocol['sources'][name]=digest
        target=out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    lock=mitra.environment_lock()
    (out/'context20_environment.lock.txt').write_text(lock)
    protocol.update(context20_policy=dict(settings=SETTINGS,resources=evidence,
        environment_sha256=hashlib.sha256(lock.encode()).hexdigest(),
        roster_attribute_sha256=load_roster()['attribute_sha256'],
        persistence='Original Mitra CPU state plus20-column preprocessing and fixed unlabeled manifest lookup',
        eligibility='Separate top5 stability A/S validation; historical83 gate unchanged'))
    write_json(path,protocol)


def evaluate(name,matrices,train,dev,seed,out):
    record_protocol(out)
    return BASE_EVALUATE(name,matrices,train,dev,seed,out)


def configure():
    if iterate.fit is fit and iterate.evaluate is evaluate:
        return
    if iterate.fit is not BASE_FIT or iterate.evaluate is not BASE_EVALUATE or BASE_FIT.__module__!='iterate':
        raise RuntimeError('Use a fresh process without other wrappers.')
    iterate.CONFIGS={iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
        CONTROL:dict(context=False,model='mitra_zeroshot',columns=list(mitra.COMPACT),
            query_chunk=2318,checkpoint_sha256=mitra.WEIGHT_SHA),
        NAME:dict(context=True,model='mitra_context20',**SETTINGS)}
    iterate.BLEND=[]
    iterate.fit,iterate.evaluate=fit,evaluate


def configure_runtime(out,stage='screen'):
    global RUN_OUT
    if stage not in ['screen','stability','probe','export']:
        raise ValueError(stage)
    configure()
    RUN_OUT=Path(out)
    if not mitra.torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS backend unavailable; no fallback.')
    mitra.torch.set_num_threads(4)
    mitra.torch.mps.set_per_process_memory_fraction(.65)
    mitra.DEADLINE=time.monotonic()+SETTINGS[f'{stage}_budget_seconds']


def assert_frozen(out):
    out=Path(out)
    iterate.assert_frozen(out)
    protocol=json.loads((out/'protocol.json').read_text())['context20_policy']
    lock_hash=hashlib.sha256(mitra.environment_lock().encode()).hexdigest()
    if (protocol['settings']!=SETTINGS or protocol['environment_sha256']!=lock_hash
            or mitra.sha256(out/'context20_environment.lock.txt')!=lock_hash
            or protocol['roster_attribute_sha256']!=load_roster()['attribute_sha256']):
        raise RuntimeError('Frozen context input/environment/configuration differs.')
    if protocol['resources']!=require_resources():
        raise RuntimeError('Frozen resource evidence changed.')


def resource_probe(shape):
    output=PROBES[shape]
    output.mkdir(parents=True,exist_ok=False)
    model=restored=None
    result=dict(status='incomplete',shape=shape,settings=SETTINGS,
        synthetic_labels=True,no_accuracy_computed=True,query_rows=2318,full_support=True)
    started=time.monotonic()
    try:
        result['source_hashes']=source_hashes()
        lock=mitra.environment_lock()
        (output/'environment.lock.txt').write_text(lock)
        result['environment_sha256']=hashlib.sha256(lock.encode()).hexdigest()
        for name in result['source_hashes']:
            target=output/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        roster=load_roster()
        rows=4636 if shape=='outer' else 8693
        a=np.arange(rows)
        b=np.arange(rows,rows+2318) if shape=='outer' else np.arange(8693,8693+2318)
        y=np.arange(rows,dtype=np.int64)%2
        result.update(support_rows=rows,roster_attribute_sha256=roster['attribute_sha256'],
            training_ids_sha256=mitra.ids_hash(a),query_ids_sha256=mitra.ids_hash(b),
            resources_before=mitra.snapshot('mps'))
        write_json(output/'result.json',result)
        model=fit_context(roster['context'].iloc[a],y,a,42,'mps',roster)
        result['actual_retained_features']=model.preprocessor.n_features
        timings=[]
        for repeat in range(2):
            tick=time.monotonic()
            p=valid_probabilities(model.predict_proba(roster['plain'].iloc[b]),len(b))
            timings.append(time.monotonic()-tick)
            if repeat:
                np.testing.assert_allclose(p,first,atol=2e-6,rtol=0)
            first=p.copy()
            print(shape,'repeat',repeat,'seconds',timings[-1],flush=True)
        small=valid_probabilities(model.predict_proba(roster['plain'].iloc[b[:128]]),128)
        np.testing.assert_allclose(small,first[:128],atol=2e-6,rtol=0)
        result.update(inference_seconds=timings,subset_passed=True,
            subset_max_difference=float(np.max(np.abs(small-first[:128]))),
            resources_after=mitra.snapshot('mps'))
        joblib.dump(model,output/'model.joblib')
        model.release()
        model=None
        gc.collect()
        mitra.release_memory()
        restored=joblib.load(output/'model.joblib')
        restored.device='mps'
        reloaded=valid_probabilities(restored.predict_proba_by_passenger_ids(roster['passenger_ids'][b]),len(b))
        np.testing.assert_allclose(reloaded,first,atol=2e-6,rtol=0)
        np.testing.assert_array_equal(reloaded[:,1]>=.5,first[:,1]>=.5)
        if len(restored.support)!=rows or restored.preprocessor.n_features!=20:
            raise RuntimeError('Resource probe did not use complete20-column support.')
        mitra.check_deadline()
        result.update(status='complete',reload_passed=True,
            reload_max_difference=float(np.max(np.abs(reloaded-first))),seconds=time.monotonic()-started)
        write_json(output/'result.json',result)
        return result
    except Exception as error:
        result.update(status='failed_inconclusive',error=repr(error),seconds=time.monotonic()-started)
        write_json(output/'result.json',result)
        raise
    finally:
        for model_to_release in [model,restored]:
            if model_to_release is not None:
                model_to_release.release()
        mitra.release_memory()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['probe','screen','stability'])
    parser.add_argument('--run',default='run_043')
    parser.add_argument('--shape',choices=['outer','final'],default='outer')
    args=parser.parse_args()
    out=ROOT/'artifacts'/args.run
    screen_preexisting=out.exists()
    configure_runtime(PROBES[args.shape] if args.stage=='probe' else out,stage=args.stage)
    def timeout(*_):
        raise TimeoutError('Context20 complete-stage budget exceeded; no partial result.')
    previous=signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,SETTINGS[f'{args.stage}_budget_seconds'])
    try:
        if args.stage=='probe':
            resource_probe(args.shape)
        elif args.stage=='stability':
            import top5_validate
            top5_validate.validate('mitra_context_candidate',args.run)
        else:
            require_resources()  # Before iterate.screen reads its labels or fits baseline.
            iterate.screen(out)
            assert_frozen(out)
            selection=json.loads((out/'selection.json').read_text())
            selection.update(stage='screen_only_pending_separate_A_S_review',automatic_eligibility=False)
            write_json(out/'selection.json',selection)
    except Exception as error:
        if args.stage=='screen' and not screen_preexisting and out.exists():
            (out/'selection.json').unlink(missing_ok=True)
            table=out/'screen_results.csv'
            if table.exists():
                table.replace(out/'screen_results.partial.csv')
            write_json(out/'failure.json',dict(status='failed_inconclusive',error=repr(error),
                partial_evidence_not_eligible=True))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        mitra.DEADLINE=None


if __name__=='__main__':
    sys.modules['mitra_context_candidate']=sys.modules[__name__]
    ContextPreprocessor.__module__='mitra_context_candidate'
    ContextPredictor.__module__='mitra_context_candidate'
    main()
