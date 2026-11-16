"""One prospective cached-OOF comparison: three fixed families, equal probability mean."""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p/'audit_top5_references.py').is_file())
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
import audit_top5_references as audit

PLAN = 'reports/online_iteration_20261116/fixed_three_family_plan.json'
NAME = 'v84_Mitra_TabDPT_Kumo_equal3'
RUN = 'run_049'
SEED = 20261112
COMPONENTS = [('run_038','v71_Mitra2_zeroshot','mitra'),
              ('run_047','v82_TabDPT13_full8','tabdpt'),
              ('run_048','v83_Kumo_medium8','kumo')]
OWN = ['three_family_mean_screen.py','test_three_family_mean_screen.py',
       'audit_top5_references.py',PLAN,'reports/official_data_verification.json']
sha = audit.file_hash


def require_committed(paths):
    paths = sorted(set(paths))
    tracked = subprocess.check_output(['git','ls-files','-z','--',*paths],cwd=ROOT).decode()
    if set(filter(None,tracked.split('\0'))) != set(paths):
        raise RuntimeError('Plan, implementation and evidence must be committed before evaluation.')
    if subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT).strip():
        raise RuntimeError('Required sources/evidence differ from their committed versions.')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def fixed_mean(probabilities):
    values = np.asarray(probabilities,dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Exactly three complete aligned probability vectors are required.')
    return np.mean(values,axis=0)  # No weights, subsets, votes or threshold options.


def require_practical_band(components, best):
    if len(components)!=3 or any(best['accuracy']-c['accuracy']>.01+1e-12 for c in components):
        raise RuntimeError('A component is more than1pp below the current common local best; do not evaluate combination.')


def ids_hash(values, kind):
    values = np.ascontiguousarray(values,dtype=np.int64)
    h = hashlib.sha256()
    if kind == 'tabdpt':
        h.update(str(values.dtype).encode()); h.update(str(values.shape).encode())
    elif kind == 'kumo': h.update(str((values.dtype,values.shape)).encode())
    elif kind != 'mitra': raise ValueError(kind)
    h.update(values.tobytes())
    return h.hexdigest()


def check_complete(directory, version):
    required = ['protocol.json','selection.json','screen_results.csv',f'{version}_seed{SEED}.npz']
    if any(not (directory/p).is_file() for p in required) or (directory/'screen_failure.json').exists():
        raise RuntimeError(f'Incomplete component: {directory.name}/{version}')


def cached_fold_scores(p, stored_folds, y, splits):
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise RuntimeError('Partial, invalid or wrong-length component OOF.')
    scores = np.array([accuracy_score(y[b],p[b]>=.5) for _,b in splits])
    if stored_folds.shape != (3,) or not np.allclose(scores,stored_folds,rtol=0,atol=1e-12):
        raise RuntimeError('Stored fold metrics do not match aligned predictions.')
    return scores


def check_fold_log(record, kind, train_ids, query_ids, labels, fit_seed):
    if (record.get('status') != 'complete' or record.get('seed') != fit_seed
            or record.get('training_rows') != len(train_ids) or record.get('prediction_rows') != len(query_ids)
            or record.get('training_ids_sha256') != ids_hash(train_ids,kind)):
        raise RuntimeError('Component fold completion or training alignment mismatch.')
    if kind != 'mitra':
        if (record.get('prediction_ids_sha256') != ids_hash(query_ids,kind)
                or record.get('labels_sha256') != ids_hash(labels,kind)
                or record.get('full_training_context') is not True):
            raise RuntimeError('Component query/label/context alignment mismatch.')
    elif record.get('full_prediction_support') is not True:
        raise RuntimeError('Mitra must use full support; query order inferred from frozen CV code.')


def verify_protocol(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    if [protocol.get(k) for k in ('seed','folds','development_rows','threshold')] != [SEED,3,6954,.5]:
        raise RuntimeError('Not the original 6954-row, 3-fold, seed20261112, threshold0.5 screen.')
    sources = {}
    for relative,digest in protocol['sources'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('Invalid source path.')
        candidates = [directory/'source'/relative]
        if relative == 'iterate.py': candidates.append(directory/'iterate_source.py')
        if relative == 'experiment.py': candidates.append(directory/'base_source.py')
        candidates.append(ROOT/relative)
        source = next((p for p in candidates if p.is_file() and sha(p)==digest),None)
        if source is None: raise RuntimeError(f'Frozen component source mismatch: {relative}')
        sources[str(source.relative_to(ROOT))] = digest
    return protocol,sources


def read_component(run, version, kind, dev, y, splits, baseline):
    directory = ROOT/'artifacts'/run
    check_complete(directory,version)
    protocol,sources = verify_protocol(directory)
    if version not in protocol['configs']: raise RuntimeError('Component missing from its protocol.')
    evidence = {}
    for name in ('protocol.json','selection.json','screen_results.csv'):
        report = ROOT/'reports'/run/name
        require_committed([str(report.relative_to(ROOT))])
        if sha(report) != sha(directory/name): raise RuntimeError('Artifact/report evidence differs.')
        evidence[str(report.relative_to(ROOT))] = sha(report)
        evidence[str((directory/name).relative_to(ROOT))] = sha(directory/name)
    path = directory/f'{version}_seed{SEED}.npz'
    with np.load(path,allow_pickle=False) as cache:
        p = np.asarray(cache['prob'],dtype=float)
        stored_folds = cache['fold_accuracy']
    scores = cached_fold_scores(p,stored_folds,y,splits)
    rows = pd.read_csv(directory/'screen_results.csv')
    row = rows[rows.version==version]
    if (len(row)!=1 or abs(float(row.iloc[0].accuracy)-accuracy_score(y,p>=.5))>1e-12
            or abs(float(row.iloc[0].log_loss)-log_loss(y,p))>1e-7):
        raise RuntimeError('Completed screen CSV does not match component cache.')
    control = directory/f'v6_no_id_features_seed{SEED}.npz'
    with np.load(control,allow_pickle=False) as cache:
        if not np.array_equal(cache['prob'],baseline): raise RuntimeError('Original v6 control changed.')
    evidence.update({str(path.relative_to(ROOT)):sha(path),str(control.relative_to(ROOT)):sha(control)})
    for fold,(a,b) in enumerate(splits):
        ending = '_training.json' if kind=='mitra' else '.json'
        log = directory/f'{version}_seed{SEED+fold}_{ids_hash(dev[a],kind)[:12]}{ending}'
        record = json.loads(log.read_text())
        check_fold_log(record,kind,dev[a],dev[b],y[a],SEED+fold)
        if kind=='kumo' and record.get('estimators')!=8: raise RuntimeError('Kumo must use all eight views in every fold.')
        if kind=='tabdpt' and record.get('prediction_settings')!=protocol['configs'][version]['predict']:
            raise RuntimeError('TabDPT fold prediction recipe differs from its frozen protocol.')
        report = ROOT/'reports'/run/log.name
        require_committed([str(report.relative_to(ROOT))])
        if sha(log)!=sha(report): raise RuntimeError('Fold log differs from committed evidence.')
        evidence[str(log.relative_to(ROOT))] = sha(log)
        evidence[str(report.relative_to(ROOT))] = sha(report)
    if kind=='kumo':
        for phase in ('full_mps','full8_mps'):
            resource = f'reports/kumo_resources/{phase}/result.json'
            if resource not in protocol['sources']: raise RuntimeError('Missing Kumo resource gate provenance.')
            if sha(ROOT/resource)!=protocol['sources'][resource]: raise RuntimeError('Kumo resource evidence changed.')
            record = json.loads((ROOT/resource).read_text())
            if record.get('status')!='complete' or record.get('device')!='mps':
                raise RuntimeError('Kumo resource gate incomplete.')
    return p,dict(version=version,run=run,weight=1/3,accuracy=float(row.iloc[0].accuracy),
                  log_loss=float(row.iloc[0].log_loss),fold_accuracy=scores.tolist()),{**sources,**evidence}


def common_local_best():
    """Read completed historical screen scores, not just these three components."""
    rows,evidence = [],{}
    csvs = list((ROOT/'reports').glob('run_*/screen_results.csv'))
    csvs += list((ROOT/'reports').glob('run_*/development_results.csv'))
    for csv in sorted(csvs):
        report = csv.parent
        protocol_path,selection = report/'protocol.json',report/'selection.json'
        if not protocol_path.exists() or not selection.exists(): continue
        p = json.loads(protocol_path.read_text())
        if [p.get(k) for k in ('seed','folds','development_rows','threshold')] != [SEED,3,6954,.5]: continue
        artifact = ROOT/'artifacts'/report.name
        if not (artifact/'selection.json').exists() or (artifact/'screen_failure.json').exists(): continue
        for file in (csv,protocol_path,selection):
            require_committed([str(file.relative_to(ROOT))])
            if sha(file)!=sha(artifact/file.name): raise RuntimeError('Historical common-screen evidence differs.')
            evidence[str(file.relative_to(ROOT))] = sha(file)
        for row in pd.read_csv(csv).itertuples():
            if not np.isfinite(row.accuracy) or not 0<=row.accuracy<=1: raise RuntimeError('Invalid historical score.')
            rows.append(dict(run=report.name,version=row.version,accuracy=float(row.accuracy)))
    if not rows: raise RuntimeError('No complete common-protocol local reference.')
    return max(rows,key=lambda x:x['accuracy']),evidence


def screen():
    out,report = ROOT/'artifacts'/RUN,ROOT/'reports'/RUN
    if out.exists() or report.exists(): raise RuntimeError('Reserved run already exists; no overwrite/resume.')
    commit = require_committed(OWN)
    plan = json.loads((ROOT/PLAN).read_text())
    expected = [dict(version=v,run=r,weight=1/3) for r,v,_ in COMPONENTS]
    if plan.get('candidate')!=NAME or plan.get('reserved_run')!=RUN or plan.get('components')!=expected:
        raise RuntimeError('Prospective fixed three-family plan differs.')
    for relative,digest in plan['known_inputs'].items():
        if sha(ROOT/relative)!=digest: raise RuntimeError('Prospective known input changed.')
    # Reject unavailable Kumo before reading even development labels.
    for run,version,_ in COMPONENTS: check_complete(ROOT/'artifacts'/run,version)
    evidence = {p:sha(ROOT/p) for p in OWN}
    data = json.loads((ROOT/'reports/official_data_verification.json').read_text())
    for name in ('train.csv','test.csv'):
        path = ROOT/'data/raw'/name
        if sha(path)!=data['files'][name]['official_sha256']: raise RuntimeError('Official dataset changed.')
        evidence[str(path.relative_to(ROOT))] = sha(path)
    split_path = ROOT/'artifacts/run_002/split_indices.npz'
    with np.load(split_path,allow_pickle=False) as split: dev = np.sort(split['development']).astype(np.int64)
    if len(dev)!=6954 or audit.row_hash(dev)!=audit.DEV_HASH: raise RuntimeError('Development membership changed.')
    evidence[str(split_path.relative_to(ROOT))] = sha(split_path)
    _,_,y = audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev)
    splits = list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,y))
    with np.load(ROOT/f'artifacts/run_038/v6_no_id_features_seed{SEED}.npz',allow_pickle=False) as cache:
        baseline = cache['prob']
    probabilities,components = [],[]
    for run,version,kind in COMPONENTS:
        p,record,files = read_component(run,version,kind,dev,y,splits,baseline)
        probabilities.append(p);components.append(record);evidence.update(files)
    best,files = common_local_best();evidence.update(files)
    require_practical_band(components,best)
    protocol = dict(candidate=NAME,seed=SEED,folds=3,threshold=.5,development_rows=len(dev),
        development_ids_sha256=audit.row_hash(dev),development_targets_sha256=audit.row_hash(y),
        source_commit=commit,components=components,combination='fixed arithmetic probability mean; exactly1/3 each',
        common_local_best_before_combination=best,source_sha256=evidence,
        automatic_eligibility=False,online_submission_authorized=False,
        disclosure='Reused development labels; no independent holdout or stable-superiority claim. NPZ has no per-row IDs; alignment verified against original development order, frozen CV code, fold ID hashes, fold scores and exact v6 controls. Historical Mitra query hash absent; query IDs are the fixed fold complement of verified training IDs.')
    out.mkdir(parents=True,exist_ok=False)
    audit.json_write(out/'protocol.json',protocol)
    for relative,digest in evidence.items():
        if sha(ROOT/relative)!=digest: raise RuntimeError('Input changed while freezing.')
        if relative.startswith('data/raw/') or relative.endswith('split_indices.npz'): continue
        target = out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    started = time.monotonic()
    p = fixed_mean(probabilities)
    fold_scores = [accuracy_score(y[b],p[b]>=.5) for _,b in splits]
    fold_losses = [log_loss(y[b],p[b]) for _,b in splits]
    comparisons = []
    for i,j in itertools.combinations(range(3),2):
        comparisons.append(dict(left=components[i]['version'],right=components[j]['version'],
            disagreement_rows=int(np.sum((probabilities[i]>=.5)!=(probabilities[j]>=.5))),
            mean_absolute_probability_difference=float(np.mean(np.abs(probabilities[i]-probabilities[j])))))
    result = dict(version=NAME,accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p)),
        fold_std=float(np.std(fold_scores)),seconds=time.monotonic()-started)
    np.savez(out/f'{NAME}_seed{SEED}.npz',prob=p,fold_accuracy=fold_scores,seconds=result['seconds'],development=dev)
    pd.DataFrame([result]).to_csv(out/'screen_results.csv',index=False)
    audit.json_write(out/'assessment.json',dict(status='complete',components=components,
        fold_accuracy=fold_scores,fold_log_loss=fold_losses,pairwise_disagreement=comparisons,
        common_local_best_before_combination=best,automatic_eligibility=False,
        conclusion='One fixed practical-band comparison only; not S-level or an upload decision.'))
    audit.json_write(out/'selection.json',dict(version=NAME,components=[c['version'] for c in components],
        screen_accuracy=result['accuracy'],stage='screen_only_pending_review',automatic_eligibility=False))
    report.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','screen_results.csv','assessment.json','selection.json'):
        shutil.copy2(out/name,report/name)
    print(json.dumps(result))


if __name__=='__main__': screen()
