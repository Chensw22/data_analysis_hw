"""Exactly one frozen .5*v44+.5*Mitra cached-OOF experiment; no model fits."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from sklearn.metrics import accuracy_score,log_loss

import audit_top5_references as audit
import top5_group_evaluate as group
import top5_validate as old_validation

ROOT=Path(__file__).resolve().parent
PHASE=ROOT/'artifacts/top5_crossfamily_phase'
NAME='v80_MitraCat_mean'
POLICY='reports/top5_crossfamily_review.json'
AUTH='reports/top5_crossfamily_authorization.json'
REFERENCES={'v44':('run_019','v44_seed_ensemble'),'mitra':('run_038','v71_Mitra2_zeroshot')}
RANDOM_SEEDS=[20261113,20261114]
GROUP_SEEDS=[20261112,20261115,20261116]
KEYS=['screen_20261112',*[f'random_{s}' for s in RANDOM_SEEDS],*[f'group_{s}' for s in GROUP_SEEDS]]
OWN_SOURCES=sorted(set(group.OWN_SOURCES+['top5_group_evaluate.py','test_top5_group_evaluate.py',
    'crossfamily_mean_evaluate.py','test_crossfamily_mean_evaluate.py',POLICY,AUTH]))


def read(path):return json.loads(Path(path).read_text())
def digest(path):return group.digest(path)
def write(path,value):audit.json_write(Path(path),value)


def check_policy():
    policy,auth=read(ROOT/POLICY),read(ROOT/AUTH)
    inherited=read(ROOT/group.POLICY)
    if (policy['candidate']!=NAME or policy['weights']!={'v44':.5,'mitra':.5}
        or policy['threshold']!=.5 or policy['development_rows']!=6954
        or policy['random_seeds']!=RANDOM_SEEDS or policy['group_seeds']!=GROUP_SEEDS
        or policy['references']!={key:spec[1] for key,spec in REFERENCES.items()}
        or policy['inherited_policy_sha256']!=digest(ROOT/group.POLICY)
        or policy['budget_seconds']!=600
        or policy['additional_A_conditions']!=['Both matched random13/14 accuracies must be at least the matched Mitra reference accuracy.']):
        raise RuntimeError('Fixed cross-family policy differs.')
    for field in ['A_research_online_gate','S_stable_local_superiority_gate','bootstrap_recipe']:
        if policy[field]!=inherited[field]:raise RuntimeError('Inherited group gate/bootstrap must not change.')
    if (auth['candidate']!=NAME or auth['eligibility_policy']!=POLICY
        or auth['eligibility_policy_sha256']!=digest(ROOT/POLICY)
        or auth['no_duplicate_or_blind_retry_upload'] is not True):
        raise RuntimeError('Cross-family authorization differs.')
    return policy


def inventory():
    """Authenticate source metadata and collect hashes; compute no blend or score."""
    record=group.assert_phase('mitra')
    queue=group.complete_queue(record)
    decision_path=group.PHASE/'eligibility.json'
    decision=read(decision_path)
    if (digest(decision_path)!=(group.PHASE/'eligibility.sha256').read_text().strip()
        or decision['status']!='complete' or decision['fixed_queue_complete'] is not True
        or decision['phase_sha256']!=digest(group.PHASE/'protocol.json')):
        raise RuntimeError('Historical group phase not complete or changed.')
    files={};sources={name:digest(ROOT/name) for name in OWN_SOURCES}
    def pin(path,expected=None):
        path=Path(path);actual=digest(path)
        if expected is not None and actual!=expected:raise RuntimeError('Input SHA changed: '+str(path))
        files[str(path.relative_to(ROOT))]=actual
        return actual
    for path in [group.PHASE/'protocol.json',group.PHASE/'protocol.sha256',decision_path,group.PHASE/'eligibility.sha256']:
        pin(path)
    for relative,expected in record['historical_files'].items():pin(ROOT/relative,expected)
    sources.update(record['sources'])
    for (kind,key),summary in queue.items():
        pin(group.PHASE/kind/key/'summary.json',decision['component_summary_sha256'][f'{kind}/{key}'])
        pin(group.PHASE/kind/key/'protocol.json',summary['execution_protocol_sha256'])
    plans={key:dict(kind=key.split('_')[0],seed=int(key.split('_')[1]),references={}) for key in KEYS}
    for ref,(run,version) in REFERENCES.items():
        folder=ROOT/'artifacts/top5_reference_audit'/ref
        summary,protocol=read(folder/'summary.json'),read(folder/'protocol.json')
        history=ROOT/'artifacts'/run/'protocol.json'
        original=read(history)
        if (summary['status']!='complete' or summary['reference']!=ref or summary['version']!=version
            or summary['historical_run_unchanged'] is not True or protocol['reference']!=ref
            or protocol['version']!=version or protocol['historical_run']!=run
            or protocol['historical_protocol_sha256']!=digest(history)
            or protocol['split_recipe']!=[list(item) for item in audit.SPLITS]
            or original['seed']!=20261112 or original['folds']!=3 or original['threshold']!=.5
            or original['development_rows']!=6954):raise RuntimeError('Historical reference identity/splits differ.')
        audit.check_historical_sources(original)
        audit.check_historical_sources(dict(sources=protocol['historical_sources']))
        audit.check_historical_sources(dict(sources=protocol['audit_sources']))
        sources.update(original['sources']);sources.update(protocol['audit_sources'])
        pin(history);pin(folder/'summary.json');pin(folder/'protocol.json')
        screen=summary['original_screen'];expected_path=f'artifacts/{run}/{version}_seed20261112.npz'
        if screen['oof_path']!=expected_path or screen['seed']!=20261112:
            raise RuntimeError('Original screen path/seed differs.')
        pin(ROOT/screen['oof_path'],screen['oof_sha256']);pin(ROOT/screen['csv_path'],screen['csv_sha256'])
        plans['screen_20261112']['references'][ref]=dict(path=screen['oof_path'],sha256=screen['oof_sha256'],
            accuracy=screen['recorded_metrics']['accuracy'],log_loss=screen['recorded_metrics']['log_loss'],
            fold_accuracy=screen['recorded_fold_accuracy'],audit_protocol=str((folder/'protocol.json').relative_to(ROOT)))
        for seed in RANDOM_SEEDS:
            entries=[entry for entry in summary['audits'] if entry['kind']=='random' and entry['seed']==seed]
            if len(entries)!=1:raise RuntimeError('Exactly one audit result per fixed random seed required.')
            entry=entries[0];expected_path=str((folder/f'random_{seed}.npz').relative_to(ROOT))
            if entry['oof_path']!=expected_path:raise RuntimeError('Random OOF path differs.')
            pin(ROOT/entry['oof_path'],entry['oof_sha256'])
            plans[f'random_{seed}']['references'][ref]=dict(path=entry['oof_path'],sha256=entry['oof_sha256'],
                accuracy=entry['accuracy'],log_loss=entry['log_loss'],audit_protocol=str((folder/'protocol.json').relative_to(ROOT)))
        for entry in queue[('refs',ref)]['results']:
            pin(ROOT/entry['oof_path'],entry['oof_sha256'])
            plans[f'group_{entry["seed"]}']['references'][ref]=dict(path=entry['oof_path'],sha256=entry['oof_sha256'],
                accuracy=entry['accuracy'],log_loss=entry['log_loss'],
                folds=entry['folds'],legacy_reused=entry['legacy_reused'])
    for seed in GROUP_SEEDS:pin(group.PHASE/f'folds_{seed}.npz',record['fold_hashes'][str(seed)])
    pin(ROOT/'artifacts/run_002/split_indices.npz')
    return sources,files,plans,record


def freeze():
    policy=check_policy()
    if PHASE.exists():raise RuntimeError('Phase already exists; no overwrite or retry.')
    sources,files,plans,old_phase=inventory()
    commit=group.require_committed(sources)
    import iterate
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    identity=group.data_identity(matrices,dev,y,groups,passengers)
    if identity!=old_phase['data']:raise RuntimeError('Historical phase data identity differs.')
    PHASE.mkdir(parents=True)
    for name in sources:
        target=PHASE/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    protocol=dict(status='frozen_before_any_candidate_score',candidate=NAME,weights={'v44':.5,'mitra':.5},threshold=.5,
        sources=sources,input_files=files,plans=plans,data=identity,environment=group.environment(),
        committed_at=commit,policy_sha256=digest(ROOT/POLICY),authorization_sha256=digest(ROOT/AUTH),
        disclosure=policy['disclosure'],screen_identity='Legacy screen NPZ has no row IDs: order is authenticated by frozen audit dev IDs, fixedseed3fold source, and per-fold accuracy reproduction.')
    write(PHASE/'protocol.json',protocol)
    (PHASE/'protocol.sha256').write_text(digest(PHASE/'protocol.json')+'\n')
    print('Frozen inputs; candidate score not calculated.',flush=True)


def assert_frozen():
    check_policy()
    if digest(PHASE/'protocol.json')!=(PHASE/'protocol.sha256').read_text().strip():raise RuntimeError('Phase manifest changed.')
    record=read(PHASE/'protocol.json')
    if (record['candidate']!=NAME or record['weights']!={'v44':.5,'mitra':.5} or record['threshold']!=.5
        or set(record['plans'])!=set(KEYS) or record['environment']!=group.environment()
        or record['policy_sha256']!=digest(ROOT/POLICY) or record['authorization_sha256']!=digest(ROOT/AUTH)):
        raise RuntimeError('Frozen recipe/environment/authorization differs.')
    for name,expected in {**record['sources'],**record['input_files']}.items():
        if digest(ROOT/name)!=expected:raise RuntimeError('Frozen source/input changed: '+name)
    return record


def assignment_for(kind,seed,y,groups):
    assignment=np.empty(len(y),dtype=np.int8)
    for fold,(_,b) in enumerate(audit.audit_splits('random' if kind=='screen' else kind,seed,y,groups)):
        assignment[b]=fold
    return assignment


def load_probability(plan,ref,dev,y,groups,passengers,assignment):
    entry=plan['references'][ref];path=ROOT/entry['path']
    if plan['kind']!='screen':
        p=group.artifact_probability(path,entry['sha256'],dev,groups,passengers,assignment)
        if plan['kind']=='group' and not entry['legacy_reused']:
            for fold,(a,b) in enumerate(group.checked_partition(assignment,groups)):
                item=entry['folds'][fold]
                if (item['fit_seed']!=plan['seed']+fold or item['group_overlap']!=0
                    or item['train_ids_sha256']!=audit.row_hash(dev[a])
                    or item['query_ids_sha256']!=audit.row_hash(dev[b])):
                    raise RuntimeError('Reference fit seed/train/query context differs from frozen fold.')
    else:
        if digest(path)!=entry['sha256']:raise RuntimeError('Screen OOF SHA changed.')
        protocol=read(ROOT/entry['audit_protocol'])
        if (protocol['development_ids_sha256']!=audit.row_hash(dev)
            or protocol['group_ids_sha256']!=audit.string_hash(groups)
            or protocol['passenger_ids_sha256']!=audit.string_hash(passengers)):
            raise RuntimeError('Legacy screen row-order provenance differs.')
        with np.load(path,allow_pickle=False) as saved:
            p=group.probabilities(saved['prob'].copy(),len(dev))
            folds=[accuracy_score(y[assignment==fold],p[assignment==fold]>=.5) for fold in range(3)]
            np.testing.assert_allclose(folds,saved['fold_accuracy'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(folds,entry['fold_accuracy'],atol=1e-12,rtol=0)
    if (abs(accuracy_score(y,p>=.5)-entry['accuracy'])>1e-12
        or abs(log_loss(y,p,labels=[0,1])-entry['log_loss'])>1e-12):
        raise RuntimeError('Reference metrics do not reproduce frozen OOF.')
    return p


def fixed_mean(v44,mitra):
    v44=group.probabilities(v44,len(v44));mitra=group.probabilities(mitra,len(v44))
    return .5*v44+.5*mitra


def score_row(kind,seed,y,p,references):
    correct=(p>=.5)==y
    row=dict(kind=kind,seed=seed,accuracy=float(correct.mean()),log_loss=float(log_loss(y,p,labels=[0,1])))
    differences={}
    for ref,q in references.items():
        other=(q>=.5)==y;diff=correct.astype(float)-other.astype(float);differences[ref]=diff
        row[f'vs_{ref}']=dict(gain=float(diff.mean()),reference_accuracy=float(other.mean()),
            corrections=int((correct&~other).sum()),regressions=int((~correct&other).sum()))
    return row,differences


def make_decision(screen,random_rows,group_rows,intervals):
    if [row['seed'] for row in random_rows]!=RANDOM_SEEDS:raise RuntimeError('Both fixed random checks required.')
    result=group.make_decision(screen,group_rows,intervals)
    result['A_checks']['both_random_gains_mitra_nonnegative']=all(row['vs_mitra']['gain']>=-1e-12 for row in random_rows)
    result['A_eligible']=bool(all(result['A_checks'].values()))
    result['S_supported']=bool(result['A_eligible'] and all(result['S_checks'].values()))
    return result


def evaluate():
    record=assert_frozen()
    if (PHASE/'eligibility.json').exists() or (PHASE/'failure.json').exists():raise RuntimeError('Already evaluated/failed; no retry.')
    import iterate
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    if group.data_identity(matrices,dev,y,groups,passengers)!=record['data']:raise RuntimeError('Frozen dev inputs changed.')
    rows=[];paired={};started=time.monotonic()
    def timeout(*_):raise TimeoutError('Frozen cached-OOF evaluation budget exceeded.')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    try:
        for key in KEYS:
            plan=record['plans'][key];kind,seed=plan['kind'],plan['seed']
            assignment=assignment_for(kind,seed,y,groups)
            if kind=='group':
                frozen_assignment=group.fold_assignment(read(group.PHASE/'protocol.json'),seed,dev,groups,passengers)
                np.testing.assert_array_equal(assignment,frozen_assignment)
            refs={ref:load_probability(plan,ref,dev,y,groups,passengers,assignment) for ref in REFERENCES}
            p=fixed_mean(refs['v44'],refs['mitra'])
            row,difference=score_row(kind,seed,y,p,refs)
            rows.append(row);paired[key]=difference
            np.savez(PHASE/f'{key}.npz',prob=p,fold_id=assignment,dev_ids=dev,group_ids=groups.astype(str),passenger_ids=passengers.astype(str))
            row['oof_sha256']=digest(PHASE/f'{key}.npz')
            write(PHASE/f'{key}.json',row)
        intervals={}
        for ref in REFERENCES:
            for seed in GROUP_SEEDS:
                intervals[f'{seed}_vs_{ref}']=old_validation.cluster_interval(paired[f'group_{seed}'][ref],groups)
            intervals[f'new_mean_vs_{ref}']=old_validation.cluster_interval(
                np.mean([paired[f'group_{s}'][ref] for s in GROUP_SEEDS[1:]],axis=0),groups)
        result=make_decision(rows[0]['accuracy'],rows[1:3],rows[3:],intervals)
        assert_frozen()
        result.update(status='complete',candidate=NAME,results=rows,paired_cluster_sensitivity=intervals,
            phase_sha256=digest(PHASE/'protocol.json'),policy_sha256=digest(ROOT/POLICY),
            authorization_sha256=digest(ROOT/AUTH),weights={'v44':.5,'mitra':.5},threshold=.5,
            disclosure=record['disclosure'],seconds=time.monotonic()-started,model_fits=0,upload_attempted=False)
        write(PHASE/'eligibility.json',result)
        (PHASE/'eligibility.sha256').write_text(digest(PHASE/'eligibility.json')+'\n')
        print(json.dumps(result),flush=True)
        return result
    except Exception as error:
        (PHASE/'eligibility.json').unlink(missing_ok=True);(PHASE/'eligibility.sha256').unlink(missing_ok=True)
        write(PHASE/'failure.json',dict(status='failed_inconclusive',error=repr(error)));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','evaluate'])
    args=parser.parse_args();freeze() if args.stage=='freeze' else evaluate()


if __name__=='__main__':main()
