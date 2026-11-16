"""Fixed raw/adapted Mitra OOF interpolation after observed component results."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

import three_family_mean_screen as common

ROOT = Path(__file__).resolve().parent
audit,sha = common.audit,common.sha
PLAN = 'reports/online_iteration_20261116/fixed_prior_adapt_plan.json'
NAME,RUN,SEED = 'v86_Mitra_prior_adapt_equal2','run_050',20261112
RAW,ADAPT = 'v71_Mitra2_zeroshot','v81_Mitra_full_episode50'
COMPONENTS = [dict(version=RAW,run='run_038',weight=.5),dict(version=ADAPT,run='run_046',weight=.5)]
OWN = ['prior_adapt_mean_screen.py','test_prior_adapt_mean_screen.py',
       'three_family_mean_screen.py','audit_top5_references.py',PLAN,'reports/official_data_verification.json']


def fixed_mean(raw, adapted):
    raw,adapted = np.asarray(raw,dtype=float),np.asarray(adapted,dtype=float)
    if raw.ndim!=1 or raw.shape!=adapted.shape or not len(raw):
        raise ValueError('Exactly two nonempty aligned probability vectors required.')
    if any(not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in (raw,adapted)):
        raise ValueError('Finite probabilities in[0,1] required.')
    return .5*raw+.5*adapted


def verified_report(run, name, evidence):
    artifact,report = ROOT/'artifacts'/run/name,ROOT/'reports'/run/name
    common.require_committed([str(report.relative_to(ROOT))])
    if sha(artifact)!=sha(report): raise RuntimeError('Artifact differs from committed report: '+name)
    for path in (artifact,report): evidence[str(path.relative_to(ROOT))]=sha(path)
    return artifact


def check_adaptation_log(log, train_ids, query_ids, labels, seed):
    if (log.get('status')!='fit_complete' or log.get('seed')!=seed
            or log.get('training_rows')!=len(train_ids) or log.get('prediction_rows')!=len(query_ids)
            or log.get('training_ids_sha256')!=audit.row_hash(train_ids)
            or log.get('prediction_ids_sha256')!=audit.row_hash(query_ids)
            or log.get('full_prediction_support') is not True
            or log.get('fresh_refit_from_original_checkpoint') is not True):
        raise RuntimeError('v81 fold is incomplete or misaligned.')
    inner,valid = train_test_split(np.arange(len(labels)),test_size=.2,stratify=labels,random_state=seed)
    if (log.get('inner_train_ids_sha256')!=audit.row_hash(train_ids[inner])
            or log.get('inner_validation_ids_sha256')!=audit.row_hash(train_ids[valid])):
        raise RuntimeError('v81 inner split differs.')
    history,refit = log.get('inner_history',[]),log.get('refit_history',[])
    steps = log.get('selected_steps')
    if ([r.get('step') for r in history]!=list(range(51))
            or not all(np.isfinite(r['validation_log_loss']) for r in history)
            or steps!=min(history,key=lambda r:(r['validation_log_loss'],r['step']))['step']
            or log.get('inner_optimizer_steps')!=50 or log.get('refit_optimizer_steps')!=steps
            or [r.get('step') for r in refit]!=list(range(1,steps+1))
            or any(r.get('finite_gradients') is not True or r.get('disjoint_ids') is not True
                   for r in history[1:]+refit)):
        raise RuntimeError('v81 fixed50 selection/refit history incomplete.')


def read_adapt(dev,y,splits,baseline,raw):
    directory = ROOT/'artifacts/run_046'
    common.check_complete(directory,ADAPT)
    protocol,evidence = common.verify_protocol(directory)
    for name in ('protocol.json','selection.json','screen_results.csv'):
        verified_report('run_046',name,evidence)
    path = directory/f'{ADAPT}_seed{SEED}.npz'
    with np.load(path,allow_pickle=False) as cache:
        p,stored = np.asarray(cache['prob'],dtype=float),cache['fold_accuracy']
    scores = common.cached_fold_scores(p,stored,y,splits)
    evidence[str(path.relative_to(ROOT))]=sha(path)
    rows = pd.read_csv(directory/'screen_results.csv');row=rows[rows.version==ADAPT]
    if (len(row)!=1 or abs(float(row.iloc[0].accuracy)-accuracy_score(y,p>=.5))>1e-12
            or abs(float(row.iloc[0].log_loss)-log_loss(y,p))>1e-7):
        raise RuntimeError('v81 cache metrics differ from completed screen.')
    for version,expected in [('v6_no_id_features',baseline),(RAW,raw)]:
        path = directory/f'{version}_seed{SEED}.npz'
        with np.load(path,allow_pickle=False) as cache:
            if not np.array_equal(cache['prob'],expected): raise RuntimeError('run046 control differs from run038.')
        evidence[str(path.relative_to(ROOT))]=sha(path)
    for fold,(a,b) in enumerate(splits):
        name = f'{ADAPT}_seed{SEED+fold}_{audit.row_hash(dev[a])[:12]}.json'
        path = verified_report('run_046',name,evidence)
        record=json.loads(path.read_text())
        check_adaptation_log(record,dev[a],dev[b],y[a],SEED+fold)
        if record.get('recipe')!=protocol['full_episode_policy']['settings']:
            # Protocol settings add runtime/budget declarations; recipe is the fixed core subset.
            settings=protocol['full_episode_policy']['settings']
            if any(settings.get(k)!=v for k,v in record.get('recipe',{}).items()) or not record.get('recipe'):
                raise RuntimeError('v81 logged recipe differs from frozen protocol.')
    env = verified_report('run_046','full_episode_environment.lock.txt',evidence)
    if sha(env)!=protocol['full_episode_policy']['environment_sha256']:
        raise RuntimeError('v81 historical environment differs from its protocol.')
    return p,dict(version=ADAPT,run='run_046',weight=.5,accuracy=float(row.iloc[0].accuracy),
        log_loss=float(row.iloc[0].log_loss),fold_accuracy=scores.tolist()),evidence


def paired(candidate,reference,y,splits):
    def one(indices):
        a,b,t=candidate[indices],reference[indices],y[indices]
        ac,bc=(a>=.5)==t,(b>=.5)==t
        return dict(accuracy_delta=float(np.mean(ac)-np.mean(bc)),
            log_loss_delta=float(log_loss(t,a)-log_loss(t,b)),
            repaired=int(np.sum(ac&~bc)),broken=int(np.sum(~ac&bc)),
            disagreements=int(np.sum(ac!=bc)))
    return dict(overall=one(np.arange(len(y))),folds=[one(b) for _,b in splits])


def exploratory_continuation(result,components,best,folds):
    checks=dict(accuracy_not_below_weaker_component=result['accuracy']>=min(c['accuracy'] for c in components)-1e-12,
        logloss_not_above_worse_component=result['log_loss']<=max(c['log_loss'] for c in components)+1e-12,
        within_common_1pp=best['accuracy']-result['accuracy']<=.01+1e-12,
        all_three_folds_complete=len(folds)==3 and bool(np.isfinite(folds).all()))
    return dict(passed=all(checks.values()),checks=checks,
        interpretation='Fixed exploratory continuation heuristic only; not stable superiority, S-tier, or upload authorization.')


def screen():
    out,report=ROOT/'artifacts'/RUN,ROOT/'reports'/RUN
    if out.exists() or report.exists(): raise RuntimeError('Reserved run exists; no overwrite or resume.')
    commit=common.require_committed(OWN)
    plan=json.loads((ROOT/PLAN).read_text())
    if (plan.get('candidate')!=NAME or plan.get('reserved_run')!=RUN
            or plan.get('components')!=COMPONENTS or plan.get('threshold')!=.5
            or plan.get('no_search') is not True or 'already observed' not in plan.get('selection_disclosure','')):
        raise RuntimeError('Fixed post-selection plan differs or fails to disclose prior observations.')
    for relative,digest in plan['known_inputs'].items():
        if sha(ROOT/relative)!=digest: raise RuntimeError('Prospective frozen component input changed.')
    for c in COMPONENTS: common.check_complete(ROOT/'artifacts'/c['run'],c['version'])
    evidence={p:sha(ROOT/p) for p in OWN}
    data=json.loads((ROOT/'reports/official_data_verification.json').read_text())
    for name in ('train.csv','test.csv'):
        path=ROOT/'data/raw'/name
        if sha(path)!=data['files'][name]['official_sha256']: raise RuntimeError('Official data changed.')
        evidence[str(path.relative_to(ROOT))]=sha(path)
    path=ROOT/'artifacts/run_002/split_indices.npz'
    with np.load(path,allow_pickle=False) as split: dev=np.sort(split['development']).astype(np.int64)
    if len(dev)!=6954 or audit.row_hash(dev)!=audit.DEV_HASH: raise RuntimeError('Development membership changed.')
    evidence[str(path.relative_to(ROOT))]=sha(path)
    _,_,y=audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev)
    splits=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,y))
    with np.load(ROOT/f'artifacts/run_038/v6_no_id_features_seed{SEED}.npz',allow_pickle=False) as cache: baseline=cache['prob']
    raw,c0,files=common.read_component('run_038',RAW,'mitra',dev,y,splits,baseline);evidence.update(files);c0['weight']=.5
    adapted,c1,files=read_adapt(dev,y,splits,baseline,raw);evidence.update(files)
    env=verified_report('run_038','requirements-mitra.lock.txt',evidence)
    if sha(env)!=sha(ROOT/'artifacts/run_046/full_episode_environment.lock.txt'):
        raise RuntimeError('Historical component environments differ; explicit review required.')
    components=[c0,c1];best,files=common.common_local_best();evidence.update(files)
    if any(best['accuracy']-c['accuracy']>.01+1e-12 for c in components):
        raise RuntimeError('Component outside current common1pp band.')
    protocol=dict(candidate=NAME,seed=SEED,folds=3,threshold=.5,development_rows=6954,
        development_ids_sha256=audit.row_hash(dev),development_targets_sha256=audit.row_hash(y),
        source_commit=commit,source_sha256=evidence,components=components,
        selection_disclosure=plan['selection_disclosure'],known_component_online_scores=plan['known_component_online_scores'],
        common_local_best_before_comparison=best,automatic_eligibility=False,online_submission_authorized=False,
        alignment_note='Legacy NPZ has no row IDs. Frozen CV source, dev membership, fold hashes, scores and exact controls verify alignment. Raw Mitra lacks query hash; its query is the verified training fold complement.')
    out.mkdir(parents=True,exist_ok=False);audit.json_write(out/'protocol.json',protocol)
    for relative,digest in evidence.items():
        if sha(ROOT/relative)!=digest: raise RuntimeError('Input changed before freezing.')
        if relative.startswith('data/raw/') or relative.endswith('split_indices.npz'): continue
        target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    started=time.monotonic();p=fixed_mean(raw,adapted)
    folds=[accuracy_score(y[b],p[b]>=.5) for _,b in splits]
    result=dict(version=NAME,accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p)),
        fold_std=float(np.std(folds)),seconds=time.monotonic()-started)
    continuation=exploratory_continuation(result,components,best,folds)
    assessment=dict(status='complete',components=components,fold_accuracy=folds,
        fold_log_loss=[log_loss(y[b],p[b]) for _,b in splits],
        paired_vs_components={RAW:paired(p,raw,y,splits),ADAPT:paired(p,adapted,y,splits)},
        exploratory_continuation=continuation,automatic_eligibility=False,S_supported=False,
        practical_tier='same_common_1pp_band' if abs(best['accuracy']-result['accuracy'])<=.01 else 'outside_common_1pp_band_requires_review',
        conclusion='Observed-component adaptive comparison; no new independent validation or stable-superiority claim.')
    np.savez(out/f'{NAME}_seed{SEED}.npz',prob=p,fold_accuracy=folds,seconds=result['seconds'],development=dev)
    pd.DataFrame([result]).to_csv(out/'screen_results.csv',index=False)
    audit.json_write(out/'assessment.json',assessment)
    audit.json_write(out/'selection.json',dict(version=NAME,components=[RAW,ADAPT],
        screen_accuracy=result['accuracy'],stage='screen_only_pending_review' if continuation['passed'] else 'archived_no_upload',
        automatic_eligibility=False))
    report.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','screen_results.csv','assessment.json','selection.json'): shutil.copy2(out/name,report/name)
    print(json.dumps(dict(**result,exploratory_continuation=continuation)))


if __name__=='__main__': screen()
