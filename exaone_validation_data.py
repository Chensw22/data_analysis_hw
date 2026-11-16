"""Row-local compact13 attributes and authenticated historical CV, no model imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import StratifiedKFold

import audit_top5_references as audit
import three_family_mean_screen as common
import prior_adapt_mean_screen as adapted

ROOT=Path(__file__).resolve().parent
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CATEGORICAL=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
PROOF='reports/exaone_validation_features.json'


def compact_features(roster):
    x=roster[[name for name in COMPACT if name not in ['Deck','CabinNumber','Side']]].copy()
    cabin=roster.Cabin.str.split('/',expand=True).reindex(columns=[0,1,2])
    x['Deck']=cabin[0];x['CabinNumber']=pd.to_numeric(cabin[1],errors='coerce').astype(float);x['Side']=cabin[2]
    for name in CATEGORICAL:x[name]=x[name].fillna('__MISSING__').astype(str)
    return x[COMPACT]


def feature_hash(frame):
    values={name:[None if pd.isna(v) else str(v) if name in CATEGORICAL else float(v)
                  for v in frame[name].tolist()] for name in COMPACT}
    return hashlib.sha256(json.dumps(values,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def remember(path,evidence):
    evidence[str(path.relative_to(ROOT))]=audit.file_hash(path)


def report_pair(directory,name,evidence):
    local=ROOT/'artifacts'/directory/name;public=ROOT/'reports'/directory/name
    common.require_committed([str(public.relative_to(ROOT))])
    if audit.file_hash(local)!=audit.file_hash(public):raise RuntimeError('Historical artifact/report mismatch: '+str(public))
    remember(local,evidence);remember(public,evidence)
    return local


def load_data():
    proof=json.loads((ROOT/PROOF).read_text());evidence={PROOF:audit.file_hash(ROOT/PROOF)}
    common.require_committed([PROOF])
    if (proof.get('status')!='exact_compact13_match' or proof.get('rows')!=12970
            or proof.get('columns')!=COMPACT or proof.get('holdout_targets_read') is not False):
        raise RuntimeError('Committed exact compact13 feature proof required.')
    for relative,digest in proof['source_data_sha256'].items():
        if audit.file_hash(ROOT/relative)!=digest:raise RuntimeError('Feature proof input changed: '+relative)
        evidence[relative]=digest
    path=ROOT/'artifacts/run_002/split_indices.npz'
    with np.load(path,allow_pickle=False) as cache:dev=np.sort(cache['development']).astype(np.int64)
    if len(dev)!=6954 or audit.row_hash(dev)!=audit.DEV_HASH:raise RuntimeError('Original development membership required.')
    remember(path,evidence)
    train,test,y=audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev)
    features=compact_features(pd.concat([train,test],ignore_index=True))
    if feature_hash(features)!=proof['compact_values_sha256']:raise RuntimeError('Compact attributes differ from baseline proof.')
    passengers=train.PassengerId.iloc[dev].astype(str).to_numpy();groups=np.array([p.split('_')[0] for p in passengers])
    return dict(dev=dev,y=y,features=features,passengers=passengers,groups=groups,evidence=evidence)


def metrics(p,y,splits):
    p=np.asarray(p)
    if p.shape!=y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('Incomplete reference OOF.')
    return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
        fold_accuracy=[float(accuracy_score(y[b],p[b]>=.5)) for _,b in splits],
        fold_log_loss=[float(log_loss(y[b],p[b],labels=[0,1])) for _,b in splits])


def random_references(inputs):
    dev,y=inputs['dev'],inputs['y'];evidence={}
    splits=list(StratifiedKFold(3,shuffle=True,random_state=20261112).split(dev,y))
    path=ROOT/'artifacts/run_038/v6_no_id_features_seed20261112.npz'
    with np.load(path,allow_pickle=False) as cache:baseline=cache['prob']
    raw,_,files=common.read_component('run_038','v71_Mitra2_zeroshot','mitra',dev,y,splits,baseline);evidence.update(files)
    full,_,files=adapted.read_adapt(dev,y,splits,baseline,raw);evidence.update(files)
    # These helpers authenticate historic locks against historic hashes, never the EXAONE environment.
    for run,name in [('run_038','requirements-mitra.lock.txt'),('run_019','requirements.lock.txt')]:
        report_pair(run,name,evidence)
    folder=ROOT/'artifacts/run_019';_,files=common.verify_protocol(folder);evidence.update(files)
    for name in ['protocol.json','selection.json','screen_results.csv']:report_pair('run_019',name,evidence)
    path=folder/'v44_seed_ensemble_seed20261112.npz'
    with np.load(path,allow_pickle=False) as cache:
        v44=cache['prob'].copy();common.cached_fold_scores(v44,cache['fold_accuracy'],y,splits)
    remember(path,evidence)
    summary=json.loads((ROOT/'reports/top5_reference_audit/v44/summary.json').read_text())
    source=summary['original_screen']
    if source['oof_sha256']!=audit.file_hash(path):raise RuntimeError('Original v44 OOF digest changed.')
    remember(ROOT/'reports/top5_reference_audit/v44/summary.json',evidence)
    actual=metrics(v44,y,splits)
    if abs(actual['accuracy']-source['recorded_metrics']['accuracy'])>1e-12:raise RuntimeError('v44 original score differs.')
    # v81 logs provide original train and query ID hashes for every reconstructed random fold.
    for fold,(a,b) in enumerate(splits):
        name=f'v81_Mitra_full_episode50_seed{20261112+fold}_{audit.row_hash(dev[a])[:12]}.json'
        log=json.loads(report_pair('run_046',name,evidence).read_text())
        if log.get('training_ids_sha256')!=audit.row_hash(dev[a]) or log.get('prediction_ids_sha256')!=audit.row_hash(dev[b]):
            raise RuntimeError('Current random split differs from original v81 membership.')
    references={'v71_Mitra2_zeroshot':raw,'v44_seed_ensemble':v44,'v81_Mitra_full_episode50':full}
    return splits,references,evidence


def validate_saved_group(cache,row_arrays,inputs):
    dev=inputs['dev'];fold_ids=cache['fold_id']
    for key,expected in [('dev_ids',dev),('passenger_ids',inputs['passengers']),('group_ids',inputs['groups'])]:
        if not np.array_equal(cache[key],expected):raise RuntimeError('Saved group roster differs: '+key)
    if fold_ids.shape!=dev.shape or set(fold_ids.tolist())!={0,1,2}:raise RuntimeError('Three complete saved folds required.')
    splits=[]
    for fold,row in enumerate(row_arrays):
        a,b=np.flatnonzero(fold_ids!=fold),np.flatnonzero(fold_ids==fold)
        if (not np.array_equal(row['train_ids'],dev[a]) or not np.array_equal(row['validation_ids'],dev[b])
                or set(inputs['groups'][a])&set(inputs['groups'][b])):
            raise RuntimeError('Saved fold arrays mismatch or passenger groups overlap.')
        splits.append((a,b))
    if len(splits)!=3:raise RuntimeError('Missing saved fold.')
    return splits


def group_references(inputs):
    evidence={};references={};baseline_cache=None;splits=None
    for key,version in [('mitra','v71_Mitra2_zeroshot'),('v44','v44_seed_ensemble')]:
        directory=f'top5_reference_audit/{key}';folder=ROOT/'artifacts'/directory
        for name in ['protocol.json','summary.json']:
            report_pair(directory,name,evidence)
        metrics_path=ROOT/'reports'/directory/'metrics.csv'
        common.require_committed([str(metrics_path.relative_to(ROOT))]);remember(metrics_path,evidence)
        protocol=json.loads((folder/'protocol.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        if (protocol.get('version')!=version or protocol.get('development_ids_sha256')!=audit.DEV_HASH
                or protocol.get('group_ids_sha256')!=audit.string_hash(inputs['groups'])
                or protocol.get('passenger_ids_sha256')!=audit.string_hash(inputs['passengers'])
                or summary.get('status')!='complete'):
            raise RuntimeError('Historical group reference identity differs.')
        for mapping in [protocol['historical_sources'],protocol['audit_sources']]:
            for relative,digest in mapping.items():
                if audit.file_hash(ROOT/relative)!=digest:raise RuntimeError('Historic group source changed: '+relative)
                evidence[relative]=digest
        history=ROOT/'artifacts'/protocol['historical_run']/'protocol.json'
        if audit.file_hash(history)!=protocol['historical_protocol_sha256']:raise RuntimeError('Historical model protocol differs.')
        remember(history,evidence)
        path=folder/'group_20261112.npz'
        with np.load(path,allow_pickle=False) as saved:cache={k:saved[k].copy() for k in saved.files}
        remember(path,evidence);rows=[]
        for fold in range(3):
            path=folder/f'group_20261112_fold{fold}_rows.npz'
            with np.load(path,allow_pickle=False) as saved:rows.append({k:saved[k].copy() for k in saved.files})
            remember(path,evidence)
        these=validate_saved_group(cache,rows,inputs)
        if baseline_cache is not None and any(not np.array_equal(cache[k],baseline_cache[k]) for k in ['dev_ids','fold_id','passenger_ids','group_ids']):
            raise RuntimeError('Raw/v44 group reference memberships differ.')
        baseline_cache=cache;splits=these;actual=metrics(cache['prob'],inputs['y'],splits)
        selected=[r for r in summary['audits'] if r['kind']=='group' and r['seed']==20261112]
        if len(selected)!=1 or selected[0]['oof_sha256']!=audit.file_hash(folder/'group_20261112.npz'):
            raise RuntimeError('Group OOF differs from authenticated summary.')
        if abs(actual['accuracy']-selected[0]['accuracy'])>1e-12 or abs(actual['log_loss']-selected[0]['log_loss'])>1e-6:
            raise RuntimeError('Group reference metrics differ.')
        table=pd.read_csv(metrics_path);entry=table[(table.kind=='group')&(table.seed==20261112)]
        if len(entry)!=1 or abs(float(entry.iloc[0].accuracy)-actual['accuracy'])>1e-12:
            raise RuntimeError('Committed group metrics differ from saved OOF.')
        for fold,(a,b) in enumerate(splits):
            log=json.loads(report_pair(directory,f'group_20261112_fold{fold}.json',evidence).read_text())
            if (log.get('train_ids_sha256')!=audit.row_hash(inputs['dev'][a])
                    or log.get('validation_ids_sha256')!=audit.row_hash(inputs['dev'][b])
                    or log.get('group_overlap_count')!=0 or log.get('model_fit_seed')!=20261112+fold
                    or abs(log['accuracy']-actual['fold_accuracy'][fold])>1e-12):raise RuntimeError('Group fold evidence mismatch.')
            if key=='mitra':
                training=log['training_metadata']
                if training.get('status')!='complete' or training.get('full_prediction_support') is not True or training.get('training_ids_sha256')!=audit.row_hash(inputs['dev'][a]):
                    raise RuntimeError('Mitra group training log incomplete.')
            elif log.get('component_seeds')!=[20261112+fold,20261212+fold,20261312+fold] or len(log.get('component_catboost',[]))!=3:
                raise RuntimeError('v44 group ensemble training evidence incomplete.')
        references[version]=cache['prob']
    return splits,references,evidence
