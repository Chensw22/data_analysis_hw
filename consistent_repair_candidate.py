"""One frozen rawMitra compact13 attribute-repair package, never an upload.

Group evidence uses original unlabeled attributes only. No observed value,
additional feature, target, threshold, or base-model setting is changed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import StratifiedKFold

import audit_top5_references as audit
import three_family_mean_screen as history

ROOT=Path(__file__).resolve().parent
DIRECTORY='consistent_repair_20261117'
PLAN=f'reports/{DIRECTORY}/protocol.json'
OUTPUT=ROOT/'artifacts'/DIRECTORY
REPORT=ROOT/'reports'/DIRECTORY
NAME='rawMitra13_consistent_attribute_repair'
REFERENCE='v71_Mitra2_zeroshot'
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP',*SPEND,'Deck','CabinNumber','Side']
CATEGORICAL=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
ALLOWED=['HomePlanet','Side',*SPEND]
OWN=['consistent_repair_candidate.py','test_consistent_repair_candidate.py','audit_top5_references.py',
     'three_family_mean_screen.py','iterate.py','experiment.py','data_quality_diagnostic.py',PLAN,
     'reports/data_error_diagnosis_20261117/data_quality.json','reports/official_data_verification.json']
BUDGET,SEED=1800,20261112
RECIPE={'development_rows': 6954, 'folds': 3, 'split_seed': 20261112, 'fit_seed': '20261112+fold', 'threshold': 0.5, 'model': 'unchanged original rawMitra compact13, checkpoint/preprocessing/mirror/fullcontext/querybatch/backend unchanged', 'rules': ['Observed CryoSleep=True: fill missing RoomService/FoodCourt/ShoppingMall/Spa/VRDeck with0', 'Missing HomePlanet: fill if nonmissing original same-GroupId roster values are unanimous', 'Missing Side: fill if nonmissing original same-GroupId Cabin-derived Side values are unanimous'], 'unlabeled_roster': 'full train+test attributes; Transported excluded before aggregation', 'rule_evidence': 'Only original observed attribute values; no cascading newly imputed evidence', 'input_format': 'Original build_features(False)30-column matrix; change only the seven allowed compact13 columns. Model selects original compact13; extra columns remain unchanged. Side filled in feature matrix only, not by inventing Cabin/Deck/number.', 'forbidden': ['Observed-value replacement', 'Label changes/row removal', 'Filling Destination/Deck/CryoSleep/Age/CabinNumber', 'Adding flags/counts/IDs', 'Choosing rules or affected people based on OOF errors', 'Probability routing restricted to repaired rows'], 'budget_seconds': 1800, 'threads': 4, 'device': 'mps', 'backend_fallback': False}
CONTINUATION={'minimum_accuracy_gain_vs_raw': 0.001, 'log_loss_not_higher_than_raw': True, 'minimum_folds_accuracy_not_lower': 2, 'if_pass': 'Separate review for fixed group validation only; no automatic upload or adaptation.', 'if_fail': 'Archive exact package; do not split/select rules, add new rules, change backend/threshold, or route by repaired status.'}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def frame_hash(frame):
    value=hashlib.sha256(json.dumps([(str(c),str(d)) for c,d in zip(frame.columns,frame.dtypes)]).encode())
    value.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes());return value.hexdigest()
def equivalent(a,b):return a.eq(b)|(a.isna()&b.isna())
def deadline_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Fixed repair1800s deadline exceeded.')
def verify_hashes(values):
    for name,digest in values.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(ROOT/path)!=digest:raise RuntimeError('Frozen input changed: '+name)


def repair_features(roster,framework):
    """Patch seven allowed feature columns; never synthesize a Cabin string."""
    if 'Transported' in roster:raise ValueError('Attribute repair must never receive target column.')
    roster=roster.reset_index(drop=True)
    original_hash=frame_hash(roster)
    baseline,cats=framework.build_features(roster,False)
    if len(baseline.columns)!=30 or len(set(baseline.columns))!=30 or not set(COMPACT).issubset(baseline):
        raise RuntimeError('Original30-column features with raw compact13 required.')
    cabin=roster.Cabin.str.split('/',expand=True).reindex(columns=[0,1,2])
    original=roster[[c for c in COMPACT if c not in ['Deck','CabinNumber','Side']]].copy()
    original['Deck']=cabin[0];original['CabinNumber']=pd.to_numeric(cabin[1],errors='coerce').astype(float);original['Side']=cabin[2]
    original=original[COMPACT]
    expected=original.copy()
    for col in CATEGORICAL:expected[col]=expected[col].fillna('__MISSING__').astype(str)
    if not equivalent(baseline[COMPACT],expected).all().all():raise RuntimeError('Baseline raw13 values differ from original attributes.')
    repaired=baseline.copy(deep=True);masks={}
    asleep=roster.CryoSleep.eq(True).fillna(False)
    for col in SPEND:
        missing=asleep&original[col].isna();repaired.loc[missing,col]=0.
    masks['CryoTrue_missing_spend']=asleep&original[SPEND].isna().any(axis=1)
    groups=roster.PassengerId.astype('string').str.split('_').str[0]
    for col in ['HomePlanet','Side']:
        observed=original[col];grouped=observed.groupby(groups)
        peer=groups.map(grouped.first().where(grouped.nunique().eq(1)))
        missing=observed.isna()&peer.notna();masks[col]=missing
        repaired.loc[missing,col]=peer.loc[missing].astype(str)
    changed=~equivalent(baseline,repaired)
    compact_changed=changed[COMPACT]
    if (changed[[c for c in baseline if c not in ALLOWED]].any().any()
            or (compact_changed&original.notna()).any().any()
            or not baseline.columns.equals(repaired.columns) or frame_hash(roster)!=original_hash):
        raise RuntimeError('Repair changed observed values, forbidden columns, schema or original roster.')
    masks['any_repair']=changed.any(axis=1)
    if not equivalent(baseline.loc[~masks['any_repair'],COMPACT],repaired.loc[~masks['any_repair'],COMPACT]).all().all():
        raise RuntimeError('Unrepaired raw feature values changed.')
    proof=dict(original_roster_sha256=original_hash,baseline_feature_sha256=frame_hash(baseline),
        repaired_feature_sha256=frame_hash(repaired),model_columns=COMPACT,all_feature_columns=list(repaired.columns),
        changed_columns=[c for c in repaired if changed[c].any()],changed_cells=int(changed.sum().sum()),
        changed_rows=int(masks['any_repair'].sum()),per_rule_rows={k:int(v.sum()) for k,v in masks.items()},
        observed_cells_unchanged=True,extra17_columns_unchanged=True,unrepaired_raw13_equal=True,
        no_cabin_synthesized=True,no_cascading_evidence=True,aggregation_labels_used=False,
        preprocessing_disclosure='Training-fold preprocessing may change transformed values of otherwise unchanged rows; equality is at raw13 input only.')
    return (repaired,cats),{k:v.to_numpy(bool) for k,v in masks.items()},proof


def contract():
    history.require_committed(OWN)
    plan=read(ROOT/PLAN)
    if plan.get('status')!='prospective_before_new_candidate_fit' or plan.get('candidate')!=NAME or plan.get('recipe')!=RECIPE or plan.get('continuation')!=CONTINUATION:
        raise RuntimeError('Fixed repair protocol differs.')
    old=read(ROOT/'reports/run_038/protocol.json')
    sources={**old['sources'],**{name:sha(ROOT/name) for name in OWN}}
    verify_hashes(sources);history.require_committed(sources)
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-mitra').resolve():raise RuntimeError('Original isolated Mitra environment required.')
    import importlib.metadata
    lock=''.join(f'{n}=={v}\n' for n,v in sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))
    environment_sha=hashlib.sha256(lock.encode()).hexdigest()
    if environment_sha!=old['mitra_environment_sha256']:raise RuntimeError('Original raw environment changed.')
    evidence={'reports/run_038/requirements-mitra.lock.txt':environment_sha,
        'artifacts/pretrained/mitra-v2/model.safetensors':old['mitra_checkpoint_sha256'],
        'artifacts/pretrained/mitra-v2/config.json':sha(ROOT/'artifacts/pretrained/mitra-v2/config.json')}
    for name in read(ROOT/'reports/mitra_protocol.json')['runtime_decision']['evidence_paths']:
        evidence[name]=sha(ROOT/name)
    official=read(ROOT/'reports/official_data_verification.json')
    for name in ['train.csv','test.csv']:evidence['data/raw/'+name]=official['files'][name]['official_sha256']
    verify_hashes(evidence)
    return dict(sources=sources,evidence=evidence,environment_sha256=environment_sha,protocol_sha256=sha(ROOT/PLAN),source_commit=history.require_committed(OWN))

def assert_frozen(frozen):
    verify_hashes(frozen['sources']);verify_hashes(frozen['evidence'])
    if sha(ROOT/PLAN)!=frozen['protocol_sha256']:raise RuntimeError('Repair protocol changed.')


def load_inputs(framework,frozen):
    assert_frozen(frozen)
    baseline,dev,y,_,_=audit.load_audit_data(framework)
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c!='Transported')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Transported')
    roster=pd.concat([train,test],ignore_index=True)
    matrices,masks,proof=repair_features(roster,framework)
    if frame_hash(baseline[False][0])!=proof['baseline_feature_sha256']:raise RuntimeError('Original feature values differ between readers.')
    splits=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,y))
    with np.load(ROOT/'artifacts/run_038/v6_no_id_features_seed20261112.npz',allow_pickle=False) as cache:control=cache['prob'].copy()
    reference,_,evidence=history.read_component('run_038',REFERENCE,'mitra',dev,y,splits,control)
    frozen['evidence'].update(evidence)
    frozen['evidence']['artifacts/run_002/split_indices.npz']=sha(ROOT/'artifacts/run_002/split_indices.npz')
    assert_frozen(frozen)
    return dict(matrices={False:matrices},dev=dev,y=y,splits=splits,reference=reference,
                masks={name:value[dev] for name,value in masks.items()},proof=proof)


def metrics(p,y,splits):
    p=np.asarray(p,dtype=float)
    if p.shape!=y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('Complete aligned probabilities required.')
    return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
        fold_accuracy=[float(accuracy_score(y[b],p[b]>=.5)) for _,b in splits],
        fold_log_loss=[float(log_loss(y[b],p[b],labels=[0,1])) for _,b in splits])
def paired(p,raw,y):
    now=(p>=.5)==y;old=(raw>=.5)==y
    return dict(rows=len(y),repaired=int((now&~old).sum()),broken=int((~now&old).sum()),net_correct=int(now.sum()-old.sum()))
def subset_report(p,raw,y,masks):
    result={}
    for name,mask in {**masks,'unrepaired':~masks['any_repair']}.items():
        n=int(mask.sum());row=paired(p[mask],raw[mask],y[mask])
        row.update(candidate_accuracy=float(np.mean((p[mask]>=.5)==y[mask])) if n else None,
                   raw_accuracy=float(np.mean((raw[mask]>=.5)==y[mask])) if n else None,
                   descriptive_only=True)
        result[name]=row
    return result

def continuation(candidate,raw):
    checks=dict(accuracy_gain_at_least_001=candidate['accuracy']-raw['accuracy']>=.001-1e-12,
        log_loss_not_higher=candidate['log_loss']<=raw['log_loss']+1e-12,
        at_least_two_nonlower_folds=sum(a+1e-12>=b for a,b in zip(candidate['fold_accuracy'],raw['fold_accuracy']))>=2)
    return dict(passed=all(checks.values()),checks=checks,automatic_export=False,automatic_upload=False,
                next_if_pass='separate fixed group validation review only',stable_superiority_claim=False)


def worker(deadline):
    frozen=read(OUTPUT/'freeze.json');model=wrapper=None;completed=0;result=dict(status='failed_inconclusive',release_complete=False)
    try:
        assert_frozen(frozen);current=contract()
        if any(current[key]!=frozen[key] for key in ['sources','environment_sha256','protocol_sha256']):raise RuntimeError('Worker source/environment differs.')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None,'0'):raise RuntimeError('Automatic backend fallback forbidden.')
        framework,wrapper,fit,_,version=audit.initialize_reference('mitra',OUTPUT)
        wrapper.DEADLINE=deadline
        inputs=load_inputs(framework,frozen);dev,y=inputs['dev'],inputs['y'];splits=inputs['splits'];raw=inputs['reference']
        if len(dev)!=6954 or wrapper.COMPACT!=COMPACT:raise RuntimeError('Original development rows/raw13 schema required.')
        for folder in [OUTPUT,REPORT]:write(folder/'freeze.json',frozen);write(folder/'repair_proof.json',inputs['proof'])
        probability=np.full(len(dev),np.nan);folds=[]
        for fold,(a,b) in enumerate(splits):
            deadline_check(deadline);assert_frozen(frozen)
            write(OUTPUT/'progress.json',dict(status='incomplete',completed_folds=completed,current_fold=fold))
            try:
                prediction,model=fit(version,inputs['matrices'],dev[a],y[a],dev[b],SEED+fold)
                history.check_fold_log(model.metadata,'mitra',dev[a],dev[b],y[a],SEED+fold)
                if model.preprocessor.columns!=COMPACT or model.metadata.get('recipe')!='zero-shot':
                    raise RuntimeError('Original raw13 zero-shot preprocessing/model required.')
                metadata=dict(model.metadata)
                write(OUTPUT/f'fold{fold}_metadata.json',metadata)
                prediction=np.asarray(prediction,dtype=float)
                if prediction.shape!=(len(b),) or not np.isfinite(prediction).all() or np.any((prediction<0)|(prediction>1)):
                    raise RuntimeError('Incomplete fold prediction.')
                probability[b]=prediction
            finally:
                active_error=sys.exc_info()[1]
                try:
                    if model is not None:model.release()
                    model=None;wrapper.release_memory();gc.collect()
                except BaseException as cleanup:
                    result['cleanup_error']=str(cleanup)
                    if active_error is None:raise
            row=dict(status='complete',fold=fold,seed=SEED+fold,train_ids_sha256=audit.row_hash(dev[a]),
                query_ids_sha256=audit.row_hash(dev[b]),metadata_sha256=sha(OUTPUT/f'fold{fold}_metadata.json'),
                candidate_accuracy=float(accuracy_score(y[b],prediction>=.5)),raw_accuracy=float(accuracy_score(y[b],raw[b]>=.5)),
                candidate_log_loss=float(log_loss(y[b],prediction,labels=[0,1])),raw_log_loss=float(log_loss(y[b],raw[b],labels=[0,1])),
                paired=paired(prediction,raw[b],y[b]))
            completed+=1;folds.append(row)
            np.savez(OUTPUT/'partial_oof.npz',probability=probability,dev_ids=dev,reference=raw,**inputs['masks'])
            for folder in [OUTPUT,REPORT]:write(folder/f'fold{fold}.json',row)
            shutil.copy2(OUTPUT/f'fold{fold}_metadata.json',REPORT/f'fold{fold}_metadata.json')
            print(f'Consistent repair fold{fold+1}/3: accuracy={row["candidate_accuracy"]:.8f}, raw={row["raw_accuracy"]:.8f}',flush=True)
        candidate,reference=metrics(probability,y,splits),metrics(raw,y,splits)
        summary=dict(status='complete',candidate=NAME,completed_folds=3,metrics=candidate,reference=reference,
            paired=paired(probability,raw,y),subsets=subset_report(probability,raw,y,inputs['masks']),
            continuation=continuation(candidate,reference),old_holdout_labels_read=False,
            validation_kind='adaptive_development_fixed_three_fold',automatic_export=False,automatic_upload=False,
            budget_seconds=BUDGET,repair_proof_sha256=sha(OUTPUT/'repair_proof.json'))
        fold_ids=np.empty(len(dev),dtype=np.int64)
        for k,(_,b) in enumerate(splits):fold_ids[b]=k
        np.savez(OUTPUT/'oof.npz',probability=probability,reference=raw,dev_ids=dev,fold_id=fold_ids,**inputs['masks'])
        assert_frozen(frozen);deadline_check(deadline)
        write(OUTPUT/'assessment.json',summary)
        result=dict(status='worker_complete',release_complete=True,completed_folds=3,
            oof_sha256=sha(OUTPUT/'oof.npz'),assessment_sha256=sha(OUTPUT/'assessment.json'),freeze_sha256=sha(OUTPUT/'freeze.json'))
    except BaseException as error:
        result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),completed_folds=completed)
    finally:
        try:
            if model is not None:model.release()
            model=None
            if wrapper is not None:wrapper.release_memory()
            gc.collect();deadline_check(deadline);result['release_complete']=True
        except BaseException as error:
            result.update(status='failed_inconclusive',release_complete=False,cleanup_error=str(error))
            result.setdefault('error_type',type(error).__name__);result.setdefault('error',str(error))
        write(OUTPUT/'worker_result.json',result)
    return result


def supervise(process,deadline):
    while process.poll() is None:
        if time.monotonic()>=deadline:
            process.kill();process.wait(timeout=5);raise TimeoutError('Parent killed repair worker at1800s deadline.')
        try:process.wait(timeout=min(1,max(.001,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:pass
    if process.returncode!=0:raise RuntimeError('Repair worker abnormal exit: '+str(process.returncode))
    deadline_check(deadline)


def run():
    if OUTPUT.exists() or any((REPORT/name).exists() for name in ['freeze.json','summary.json','failure.json']):raise FileExistsError('No overwrite/resume.')
    started=time.monotonic();deadline=started+BUDGET;frozen=contract();deadline_check(deadline)
    OUTPUT.mkdir(parents=True);write(OUTPUT/'freeze.json',frozen);write(REPORT/'freeze.json',frozen)
    process=None
    try:
        shutil.copy2(ROOT/PLAN,OUTPUT/'protocol.json')
        with (OUTPUT/'native.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--deadline',str(deadline)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            supervise(process,deadline)
        child=read(OUTPUT/'worker_result.json')
        if child.get('status')!='worker_complete' or child.get('release_complete') is not True or child.get('completed_folds')!=3:
            raise RuntimeError('Incomplete repair worker: '+str(child.get('error_type'))+': '+str(child.get('error')))
        for name,key in [('oof.npz','oof_sha256'),('assessment.json','assessment_sha256'),('freeze.json','freeze_sha256')]:
            if sha(OUTPUT/name)!=child[key]:raise RuntimeError('Complete output changed: '+name)
        assert_frozen(read(OUTPUT/'freeze.json'));summary=read(OUTPUT/'assessment.json')
        if summary.get('status')!='complete' or summary.get('completed_folds')!=3:raise RuntimeError('Complete3fold assessment required.')
        summary.update(seconds=time.monotonic()-started,result_files_sha256={'oof.npz':child['oof_sha256'],'worker_result.json':sha(OUTPUT/'worker_result.json')})
        deadline_check(deadline)
        for folder in [OUTPUT,REPORT]:write(folder/'summary.json',summary);write(folder/'progress.json',summary)
        shutil.copy2(OUTPUT/'worker_result.json',REPORT/'worker_result.json')
        return summary
    except BaseException as error:
        if process is not None and process.poll() is None:process.kill();process.wait(timeout=5)
        failure=dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),budget_seconds=BUDGET,
            seconds=time.monotonic()-started,automatic_export=False,automatic_upload=False)
        if (OUTPUT/'worker_result.json').exists():failure['worker_failure']=read(OUTPUT/'worker_result.json')
        if (OUTPUT/'oof.npz').exists():(OUTPUT/'oof.npz').replace(OUTPUT/'oof.partial.npz')
        if (OUTPUT/'assessment.json').exists():(OUTPUT/'assessment.json').replace(OUTPUT/'assessment.partial.json')
        for folder in [OUTPUT,REPORT]:
            (folder/'summary.json').unlink(missing_ok=True);write(folder/'failure.json',failure);write(folder/'progress.json',failure)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:
        if args.deadline is None:parser.error('Worker requires shared deadline.')
        worker(args.deadline)
    else:print(json.dumps(run(),indent=2))
