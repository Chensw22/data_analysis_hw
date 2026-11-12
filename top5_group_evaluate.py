"""One committed, frozen group-primary phase on the reused development rows.

Freeze before any new fit; run each reference/candidate in a fresh process and
its original environment. Entire fixed queue must finish before eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import audit_top5_references as audit
import top5_validate as old_validation

ROOT = Path(__file__).resolve().parent
PHASE = ROOT/'artifacts/top5_group_phase'
POLICY = 'reports/top5_group_validation_review.json'
AUTH = 'reports/top5_group_authorization.json'
SEEDS = [20261112, 20261115, 20261116]
CANDIDATES = {
    'mirror': dict(version='v74_Mitra_mirror4', module='mitra_mirror_ensemble', run='run_040'),
    'context': dict(version='v77_Mitra_context20', module='mitra_context_candidate', run='run_043')}
REFERENCES = {
    'mitra': dict(version='v71_Mitra2_zeroshot', run='run_038'),
    'v44': dict(version='v44_seed_ensemble', run='run_019')}
DISCLOSURE = ('One prospective adaptive revision motivated by previously documented '
    'train/test group separation and with old group12 results already observed. '
    'All6954 development labels are reused; new partitions are not an independent '
    'holdout. Old random-validation failures remain unchanged.')
OWN_SOURCES = ['top5_group_evaluate.py', 'test_top5_group_evaluate.py',
    'audit_top5_references.py', 'test_audit_top5_references.py',
    'top5_validate.py', 'test_top5_validate.py', POLICY, AUTH]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    audit.json_write(Path(path), value)


def environment():
    return {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}


def check_policy():
    policy, auth = json.loads((ROOT/POLICY).read_text()), json.loads((ROOT/AUTH).read_text())
    scope = policy['fixed_scope_before_new_scores']
    candidates = [v['version'] for v in CANDIDATES.values()]
    references = [v['version'] for v in REFERENCES.values()]
    if (scope['candidates'] != candidates or scope['references'] != references
            or scope['group_split_seeds'] != SEEDS or scope['new_primary_seeds'] != SEEDS[1:]
            or scope['folds_per_seed'] != 3):
        raise RuntimeError('Fixed candidate/reference/seed queue differs from the reviewed policy.')
    if (auth['eligibility_policy'] != POLICY or auth['eligibility_policy_sha256'] != digest(ROOT/POLICY)
            or auth['fixed_candidates'] != candidates or auth['fixed_references'] != references):
        raise RuntimeError('New group-primary authorization is missing or differs.')
    budgets = auth['budgets_seconds']
    if set(budgets) != {'refs', 'candidates'} or set(budgets['refs']) != set(REFERENCES) or set(budgets['candidates']) != set(CANDIDATES):
        raise RuntimeError('All fixed components require their own declared budgets.')
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for section in budgets.values() for v in section.values()):
        raise RuntimeError('Positive integer execution budgets required.')
    if policy['bootstrap_recipe']['replicates'] != 2000 or policy['bootstrap_recipe']['seed'] != 20261119:
        raise RuntimeError('Bootstrap recipe differs.')
    return policy, auth


def checked_partition(assignment, groups):
    assignment, groups = np.asarray(assignment), np.asarray(groups).astype(str)
    if assignment.shape != groups.shape or set(assignment.tolist()) != {0, 1, 2}:
        raise RuntimeError('A complete exactly-three-fold assignment is required.')
    folds = []
    for fold in range(3):
        a, b = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        if not len(a) or not len(b) or set(groups[a]) & set(groups[b]):
            raise RuntimeError('Group crosses train/query boundary or fold is empty.')
        folds.append((a, b))
    if not np.array_equal(np.sort(np.concatenate([b for _, b in folds])), np.arange(len(groups))):
        raise RuntimeError('OOF rows are not a strict partition.')
    return folds


def probabilities(values, n):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise RuntimeError('Incomplete or invalid probabilities.')
    return values


def data_identity(matrices, dev, y, groups, passengers):
    frame = matrices[False][0]
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return dict(development_ids_sha256=audit.row_hash(dev), targets_sha256=audit.row_hash(y),
        group_ids_sha256=audit.string_hash(groups), passenger_ids_sha256=audit.string_hash(passengers),
        engineered_schema=list(frame.columns), engineered_values_sha256=hashlib.sha256(values).hexdigest())


def artifact_probability(path, expected_hash, dev, groups, passengers, assignment):
    if digest(path) != expected_hash:
        raise RuntimeError('OOF SHA differs from its frozen evidence.')
    with np.load(path, allow_pickle=False) as saved:
        for key, actual in [('dev_ids', dev), ('group_ids', groups), ('passenger_ids', passengers), ('fold_id', assignment)]:
            expected = saved[key]
            if key in ('group_ids', 'passenger_ids'):
                expected, actual = expected.astype(str), np.asarray(actual).astype(str)
            if not np.array_equal(expected, actual):
                raise RuntimeError(f'OOF row/group/fold identity differs: {key}')
        return probabilities(saved['prob'].copy(), len(dev))


def legacy_manifest(kind, key):
    """Only the three explicitly permitted old group12 caches are considered."""
    if kind == 'refs':
        folder = ROOT/'artifacts/top5_reference_audit'/key
        summary, protocol = json.loads((folder/'summary.json').read_text()), json.loads((folder/'protocol.json').read_text())
        spec = REFERENCES[key]
        if (summary['status'] != 'complete' or summary['reference'] != key
                or summary['version'] != spec['version'] or not summary['historical_run_unchanged']
                or protocol['reference'] != key or protocol['version'] != spec['version']):
            raise RuntimeError('Legacy reference identity or completion differs.')
        historical = ROOT/'artifacts'/spec['run']/'protocol.json'
        if protocol['historical_protocol_sha256'] != digest(historical):
            raise RuntimeError('Legacy reference historical protocol differs.')
        audit.check_historical_sources(dict(sources=protocol['historical_sources']))
        for name, expected in protocol['audit_sources'].items():
            if digest(ROOT/name) != expected:
                raise RuntimeError('Legacy reference audit implementation differs.')
        entry = next(r for r in summary['audits'] if r['kind'] == 'group' and r['seed'] == SEEDS[0])
        path = ROOT/entry['oof_path']
        if digest(path) != entry['oof_sha256']:
            raise RuntimeError('Legacy reference OOF differs from its original summary SHA.')
        files = [folder/'summary.json', folder/'protocol.json', historical, path]
        return dict(oof=str(path.relative_to(ROOT)), files={str(p.relative_to(ROOT)): digest(p) for p in files},
                    recorded_accuracy=entry['accuracy'], kind=kind, key=key, seed=SEEDS[0])
    if (kind, key) != ('candidates', 'mirror'):
        raise RuntimeError('No legacy group cache is authorized for this component.')
    folder = ROOT/'artifacts/run_040/stability'
    protocol = json.loads((folder/'protocol.json').read_text())
    eligibility = json.loads((folder/'eligibility.json').read_text())
    historical = ROOT/'artifacts/run_040/protocol.json'
    if (eligibility['status'] != 'complete' or eligibility['candidate'] != CANDIDATES['mirror']['version']
            or protocol['candidate'] != CANDIDATES['mirror']['version']
            or protocol['original_screen_protocol_sha256'] != digest(historical)
            or eligibility['original_screen_protocol_sha256'] != digest(historical)
            or protocol['own_source_sha256'] != digest(ROOT/'top5_validate.py')):
        raise RuntimeError('Legacy mirror protocol/model identity differs.')
    for field, source in [('validation_policy_sha256', 'reports/top5_validation_review.json'),
                          ('authorization_sha256', 'reports/top5_authorization.json')]:
        if protocol[field] != digest(ROOT/source) or eligibility[field] != digest(ROOT/source):
            raise RuntimeError('Legacy mirror policy/auth evidence differs.')
    entry = next(r for r in eligibility['results'] if r['kind'] == 'group' and r['seed'] == SEEDS[0])
    path = folder/'group_20261112.npz'
    # Original validator did not store this OOF hash. Pin its bytes now, before
    # new fits, alongside the original completed protocol and aligned rows.
    files = [folder/'protocol.json', folder/'eligibility.json', historical,
             folder/'group_20261112.json', path]
    return dict(oof=str(path.relative_to(ROOT)), files={str(p.relative_to(ROOT)): digest(p) for p in files},
        recorded_accuracy=entry['accuracy'], kind=kind, key=key, seed=SEEDS[0],
        sha_provenance='OOF SHA pinned at new phase freeze, not retrospectively claimed as an old signed hash')


def require_committed(paths):
    paths = sorted(paths)
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *paths], cwd=ROOT).decode().split('\0')
    if set(filter(None, tracked)) != set(paths):
        raise RuntimeError('Commit every phase source/policy/auth before freeze.')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=ROOT).decode()
    if dirty:
        raise RuntimeError('Phase sources differ from committed versions; unrelated worktree files are allowed.')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()


def freeze():
    policy, auth = check_policy()
    if PHASE.exists():
        raise RuntimeError('Phase exists; no overwrite or retrospective freeze.')
    source_hashes = {}
    historical = {}
    for spec in [*REFERENCES.values(), *CANDIDATES.values()]:
        path = ROOT/'artifacts'/spec['run']/'protocol.json'
        record = json.loads(path.read_text())
        audit.check_historical_sources(record)
        source_hashes.update(record['sources'])
        historical[str(path.relative_to(ROOT))] = digest(path)
        if spec in CANDIDATES.values():
            screen = path.parent/'screen_results.csv'
            historical[str(screen.relative_to(ROOT))] = digest(screen)
            for old_name in ['stability/eligibility.json', 'stability/protocol.json']:
                old_path=path.parent/old_name
                if old_path.exists():
                    historical[str(old_path.relative_to(ROOT))]=digest(old_path)
    for name in OWN_SOURCES:
        source_hashes[name] = digest(ROOT/name)
    commit = require_committed(source_hashes)
    import iterate
    matrices, dev, y, groups, passengers = audit.load_audit_data(iterate)
    screen_evidence={}
    for key,spec in CANDIDATES.items():
        folder=ROOT/'artifacts'/spec['run']
        table=pd.read_csv(folder/'screen_results.csv')
        rows=table.loc[table.version.eq(spec['version'])]
        if len(rows)!=1:raise RuntimeError('Original screen must contain exactly one candidate row.')
        screen_path=folder/f'{spec["version"]}_seed20261112.npz'
        with np.load(screen_path,allow_pickle=False) as saved:
            p=probabilities(saved['prob'],len(dev))
        accuracy=float(accuracy_score(y,p>=.5))
        loss=float(log_loss(y,p,labels=[0,1]))
        if abs(accuracy-float(rows.iloc[0].accuracy))>1e-12 or abs(loss-float(rows.iloc[0].log_loss))>1e-12:
            raise RuntimeError('Original screen CSV does not reproduce saved candidate OOF.')
        relative=str(screen_path.relative_to(ROOT))
        historical[relative]=digest(screen_path)
        screen_evidence[key]=dict(version=spec['version'],oof_path=relative,oof_sha256=historical[relative],
            accuracy=accuracy,log_loss=loss,development_ids_sha256=audit.row_hash(dev))
    assignments = {}
    for seed in SEEDS:
        assignment = np.empty(len(dev), dtype=np.int8)
        for fold, (_, b) in enumerate(audit.audit_splits('group', seed, y, groups)):
            assignment[b] = fold
        checked_partition(assignment, groups)
        assignments[seed] = assignment
    caches = {}
    for kind, key in [('refs','mitra'), ('refs','v44'), ('candidates','mirror')]:
        cache = legacy_manifest(kind, key)
        p = artifact_probability(ROOT/cache['oof'], cache['files'][cache['oof']],
            dev, groups, passengers, assignments[SEEDS[0]])
        if abs(accuracy_score(y, p >= .5)-cache['recorded_accuracy']) > 1e-12:
            raise RuntimeError('Legacy score cannot be reproduced from provenance-matched OOF.')
        caches[f'{kind}/{key}'] = cache
    envs = {}
    for name, executable in [('v44','.venv/bin/python'), ('mitra','.venv-mitra/bin/python')]:
        code = 'import json,importlib.metadata as m; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()},sort_keys=True))'
        envs[name] = json.loads(subprocess.check_output([str(ROOT/executable), '-c', code], cwd=ROOT))
    PHASE.mkdir(parents=True)
    (PHASE/'source').mkdir()
    for name in source_hashes:
        target = PHASE/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    fold_hashes = {}
    for seed, assignment in assignments.items():
        path = PHASE/f'folds_{seed}.npz'
        np.savez(path, fold_id=assignment, dev_ids=dev, group_ids=groups.astype(str), passenger_ids=passengers.astype(str))
        fold_hashes[str(seed)] = digest(path)
    record = dict(status='frozen_before_new_group_fits', committed_at=commit,
        sources=source_hashes, historical_files=historical, candidates=CANDIDATES, references=REFERENCES,
        seeds=SEEDS, budgets_seconds=auth['budgets_seconds'], environments=envs,
        data=data_identity(matrices,dev,y,groups,passengers), fold_hashes=fold_hashes,
        legacy_caches=caches, disclosure=DISCLOSURE,
        screen_evidence=screen_evidence,
        queue_completion_required_before_any_eligibility=True,
        fixed_primary_metric='accuracy at0.5', bootstrap_replicates=2000, bootstrap_seed=20261119)
    write_json(PHASE/'protocol.json', record)
    (PHASE/'protocol.sha256').write_text(digest(PHASE/'protocol.json')+'\n')
    print('Frozen phase', commit, flush=True)


def assert_phase(env=None):
    check_policy()
    if digest(PHASE/'protocol.json') != (PHASE/'protocol.sha256').read_text().strip():
        raise RuntimeError('Phase protocol changed.')
    record = json.loads((PHASE/'protocol.json').read_text())
    if record['candidates'] != CANDIDATES or record['references'] != REFERENCES or record['seeds'] != SEEDS:
        raise RuntimeError('Fixed queue changed; no candidate can be removed.')
    for name, expected in {**record['sources'], **record['historical_files']}.items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f'Frozen source/history changed: {name}')
    for cache in record['legacy_caches'].values():
        for name, expected in cache['files'].items():
            if digest(ROOT/name)!=expected:
                raise RuntimeError(f'Original cache evidence changed: {name}')
    if env is not None and environment() != record['environments'][env]:
        raise RuntimeError('Use the exact frozen environment for this component.')
    return record


def fold_assignment(record, seed, dev, groups, passengers):
    path = PHASE/f'folds_{seed}.npz'
    if digest(path) != record['fold_hashes'][str(seed)]:
        raise RuntimeError('Frozen fold file changed.')
    with np.load(path, allow_pickle=False) as saved:
        for key, value in [('dev_ids',dev), ('group_ids',groups.astype(str)), ('passenger_ids',passengers.astype(str))]:
            if not np.array_equal(saved[key],value):
                raise RuntimeError('Current rows do not match frozen folds.')
        assignment=saved['fold_id'].copy()
    checked_partition(assignment,groups)
    return assignment


def execute(kind, key):
    specs = REFERENCES if kind == 'refs' else CANDIDATES
    if key not in specs:
        raise ValueError('Component not in the fixed queue.')
    env = 'v44' if (kind,key)==('refs','v44') else 'mitra'
    record=assert_phase(env)
    output=PHASE/kind/key
    output.mkdir(parents=True,exist_ok=False)
    budget=record['budgets_seconds'][kind][key]
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('Frozen group-phase component budget exceeded.')
    previous=signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,budget)
    rows=[]
    try:
        if kind=='refs':
            framework,wrapper,fit,_,version=audit.initialize_reference(key,output)
        else:
            spec=specs[key]
            wrapper=importlib.import_module(spec['module'])
            wrapper.configure_runtime(output,stage='stability')
            wrapper.assert_frozen(ROOT/'artifacts'/spec['run'])
            framework=importlib.import_module('iterate')
            fit,version=wrapper.fit,spec['version']
        if version!=specs[key]['version']:
            raise RuntimeError('Loaded component identity differs.')
        # All Mitra wrappers share this deadline; model/source configuration is unchanged.
        runtime=getattr(wrapper,'mitra',wrapper)
        if hasattr(runtime,'DEADLINE'):runtime.DEADLINE=started+budget
        matrices,dev,y,groups,passengers=audit.load_audit_data(framework)
        if data_identity(matrices,dev,y,groups,passengers)!=record['data']:
            raise RuntimeError('Development data/attributes changed after freeze.')
        write_json(output/'protocol.json',dict(kind=kind,key=key,version=version,
            phase_sha256=digest(PHASE/'protocol.json'), environment=env,budget_seconds=budget,
            execution='CPU4' if env=='v44' else 'MPS',disclosure=DISCLOSURE))
        for seed in SEEDS:
            assignment=fold_assignment(record,seed,dev,groups,passengers)
            cache=record['legacy_caches'].get(f'{kind}/{key}') if seed==SEEDS[0] else None
            if cache:
                for relative, expected in cache['files'].items():
                    if digest(ROOT/relative)!=expected:
                        raise RuntimeError('Legacy cache evidence changed after freeze.')
                p=artifact_probability(ROOT/cache['oof'],cache['files'][cache['oof']],dev,groups,passengers,assignment)
                fold_logs=[]
            else:
                p=np.full(len(y),np.nan)
                fold_logs=[]
                for fold,(a,b) in enumerate(checked_partition(assignment,groups)):
                    tick=time.monotonic()
                    predicted,model=fit(version,matrices,dev[a],y[a],dev[b],seed+fold)
                    try:
                        p[b]=probabilities(predicted,len(b))
                        fold_logs.append(dict(fold=fold,fit_seed=seed+fold,
                            train_ids_sha256=audit.row_hash(dev[a]),query_ids_sha256=audit.row_hash(dev[b]),
                            group_overlap=0,seconds=time.monotonic()-tick))
                        write_json(output/f'group_{seed}_fold{fold}.json',fold_logs[-1])
                    finally:
                        if hasattr(model,'release'):model.release()
                        del model
                    print(kind,key,seed,'fold',fold+1,'complete',flush=True)
                probabilities(p,len(y))
            path=output/f'group_{seed}.npz'
            np.savez(path,prob=p,fold_id=assignment,dev_ids=dev,group_ids=groups.astype(str),passenger_ids=passengers.astype(str))
            row=dict(seed=seed,accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
                oof_path=str(path.relative_to(ROOT)),oof_sha256=digest(path),
                legacy_reused=bool(cache),legacy_evidence=cache,folds=fold_logs)
            rows.append(row)
            write_json(output/'progress.json',dict(status='incomplete',results=rows))
            print(kind,key,json.dumps(row),flush=True)
        assert_phase(env)
        write_json(output/'summary.json',dict(status='complete',kind=kind,key=key,version=version,
            phase_sha256=digest(PHASE/'protocol.json'),execution_protocol_sha256=digest(output/'protocol.json'),
            results=rows,total_seconds=time.monotonic()-started))
    except Exception as error:
        write_json(output/'failure.json',dict(status='incomplete',error=repr(error),completed_splits=[r['seed'] for r in rows]))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


def make_decision(screen, rows, intervals):
    if [r['seed'] for r in rows]!=SEEDS:
        raise RuntimeError('Every candidate requires all three fixed group splits.')
    old,new=rows[0],rows[1:]
    a=dict(original_screen_in_band=screen>=.8198159332758125-.01-1e-12,
        both_new_gains_mitra_positive=all(r['vs_mitra']['gain']>1e-12 for r in new),
        mean_new_gain_mitra_at_least_point1pp=np.mean([r['vs_mitra']['gain'] for r in new])>=.001-1e-12,
        old_group_gain_mitra_nonnegative=old['vs_mitra']['gain']>=-1e-12,
        all_three_within_one_pp_v44=all(r['vs_v44']['gain']>=-.01-1e-12 for r in rows))
    s=dict(original_screen_above83=screen>.83,new_mean_above83=np.mean([r['accuracy'] for r in new])>.83,
        both_new_gains_v44_positive=all(r['vs_v44']['gain']>1e-12 for r in new),
        mean_new_gain_v44_above_one_pp=np.mean([r['vs_v44']['gain'] for r in new])>.01,
        old_group_gain_v44_positive=old['vs_v44']['gain']>1e-12,
        new_mean_cluster_lower_positive=intervals['new_mean_vs_v44']['lower95']>0)
    return dict(A_eligible=bool(all(a.values())),S_supported=bool(all(a.values()) and all(s.values())),
        A_checks={k:bool(v) for k,v in a.items()},S_checks={k:bool(v) for k,v in s.items()},
        automatic_upload_ready=False,submission_checks_pending=['export/source identity','duplicate file','official allowance'])


def complete_queue(record):
    summaries={}
    for kind,specs in [('refs',REFERENCES),('candidates',CANDIDATES)]:
        for key,spec in specs.items():
            folder=PHASE/kind/key
            if (folder/'failure.json').exists() or not (folder/'summary.json').exists():
                raise RuntimeError(f'Entire fixed queue required; incomplete {kind}/{key}')
            value=json.loads((folder/'summary.json').read_text())
            if (value['status']!='complete' or value['kind']!=kind or value['key']!=key
                    or value['version']!=spec['version'] or value['phase_sha256']!=digest(PHASE/'protocol.json')
                    or [r['seed'] for r in value['results']]!=SEEDS
                    or value['execution_protocol_sha256']!=digest(folder/'protocol.json')):
                raise RuntimeError('Completed component identity/split queue differs.')
            execution=json.loads((folder/'protocol.json').read_text())
            expected_env='v44' if (kind,key)==('refs','v44') else 'mitra'
            if (execution['kind']!=kind or execution['key']!=key or execution['version']!=spec['version']
                    or execution['phase_sha256']!=digest(PHASE/'protocol.json')
                    or execution['environment']!=expected_env
                    or execution['budget_seconds']!=record['budgets_seconds'][kind][key]
                    or execution['execution']!=('CPU4' if expected_env=='v44' else 'MPS')):
                raise RuntimeError('Execution protocol identity/environment/budget differs.')
            for row in value['results']:
                expected_path=folder/f'group_{row["seed"]}.npz'
                if row['oof_path']!=str(expected_path.relative_to(ROOT)):
                    raise RuntimeError('Component OOF path does not belong to this fixed task.')
                cache=record['legacy_caches'].get(f'{kind}/{key}') if row['seed']==SEEDS[0] else None
                if (row['legacy_reused']!=bool(cache) or row['legacy_evidence']!=cache
                        or (not cache and [f['fold'] for f in row['folds']]!=[0,1,2])):
                    raise RuntimeError('Required complete folds or allowed cache identity differ.')
            summaries[(kind,key)]=value
    return summaries


def verified_queue(record):
    summaries=complete_queue(record)
    import iterate
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    if data_identity(matrices,dev,y,groups,passengers)!=record['data']:raise RuntimeError('Data changed.')
    loaded={}
    for component,summary in summaries.items():
        for row in summary['results']:
            assignment=fold_assignment(record,row['seed'],dev,groups,passengers)
            p=artifact_probability(ROOT/row['oof_path'],row['oof_sha256'],dev,groups,passengers,assignment)
            if abs(accuracy_score(y,p>=.5)-row['accuracy'])>1e-12:raise RuntimeError('Recorded metric differs from OOF.')
            if not row['legacy_reused']:
                for fold,(a,b) in enumerate(checked_partition(assignment,groups)):
                    item=row['folds'][fold]
                    if (item['fit_seed']!=row['seed']+fold or item['group_overlap']!=0
                            or item['train_ids_sha256']!=audit.row_hash(dev[a])
                            or item['query_ids_sha256']!=audit.row_hash(dev[b])):
                        raise RuntimeError('Completed fit boundary does not match frozen folds.')
            loaded[(*component,row['seed'])]=p
    return summaries,loaded,y,groups


def assert_eligibility(candidate_key):
    """Read-only exporter gate; verifies the whole queue, files and frozen scope.

Returns (phase_manifest, phase_eligibility, candidate_decision). This does not
perform export, duplicate-file checks, allowance checks or an online upload.
"""
    if candidate_key not in CANDIDATES:raise ValueError('Candidate outside fixed queue.')
    record=assert_phase()
    summaries,_,_,_=verified_queue(record)
    path=PHASE/'eligibility.json'
    if digest(path)!=(PHASE/'eligibility.sha256').read_text().strip():
        raise RuntimeError('New-phase decision changed after finalization.')
    decision=json.loads(path.read_text())
    if (decision['status']!='complete' or not decision['fixed_queue_complete']
            or set(decision['candidates'])!=set(CANDIDATES)
            or decision['phase_sha256']!=digest(PHASE/'protocol.json')
            or decision['policy_sha256']!=digest(ROOT/POLICY)
            or decision['authorization_sha256']!=digest(ROOT/AUTH)):
        raise RuntimeError('Group eligibility phase/policy/queue differs.')
    for kind,key in summaries:
        if decision['component_summary_sha256'][f'{kind}/{key}']!=digest(PHASE/kind/key/'summary.json'):
            raise RuntimeError('Completed component changed after decision.')
    candidate=decision['candidates'][candidate_key]
    if candidate['candidate']!=CANDIDATES[candidate_key]['version'] or not candidate['A_eligible']:
        raise RuntimeError('Candidate did not qualify under the new group-primary policy.')
    return record,decision,candidate


def finalize():
    record=assert_phase()
    if (PHASE/'eligibility.json').exists():raise RuntimeError('New-phase eligibility already recorded.')
    summaries,loaded,y,groups=verified_queue(record)  # Both candidates must finish.
    decisions={}
    for key,spec in CANDIDATES.items():
        rows,differences=[],{}
        for seed in SEEDS:
            correct=(loaded[('candidates',key,seed)]>=.5)==y
            row=dict(seed=seed,accuracy=float(correct.mean()))
            for ref in REFERENCES:
                other=(loaded[('refs',ref,seed)]>=.5)==y
                difference=correct.astype(float)-other.astype(float)
                differences[(seed,ref)]=difference
                row[f'vs_{ref}']=dict(gain=float(difference.mean()),reference_accuracy=float(other.mean()),
                    corrections=int((correct&~other).sum()),regressions=int((~correct&other).sum()))
            rows.append(row)
        intervals={}
        for ref in REFERENCES:
            for seed in SEEDS:
                intervals[f'{seed}_vs_{ref}']=old_validation.cluster_interval(differences[(seed,ref)],groups)
            intervals[f'new_mean_vs_{ref}']=old_validation.cluster_interval(
                np.mean([differences[(seed,ref)] for seed in SEEDS[1:]],axis=0),groups)
        screen=record['screen_evidence'][key]['accuracy']
        decision=make_decision(screen,rows,intervals)
        decision.update(candidate=spec['version'],original_screen_accuracy=screen,results=rows,
            paired_cluster_sensitivity=intervals,disclosure=DISCLOSURE,old_eligibility_unchanged=True)
        decisions[key]=decision
    result=dict(status='complete',fixed_queue_complete=True,phase_sha256=digest(PHASE/'protocol.json'),
        policy_sha256=digest(ROOT/POLICY),authorization_sha256=digest(ROOT/AUTH),
        component_summary_sha256={f'{kind}/{key}':digest(PHASE/kind/key/'summary.json') for kind,key in summaries},
        candidates=decisions,disclosure=DISCLOSURE)
    write_json(PHASE/'eligibility.json',result)
    (PHASE/'eligibility.sha256').write_text(digest(PHASE/'eligibility.json')+'\n')
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','reference','candidate','finalize'])
    parser.add_argument('component',nargs='?')
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='finalize':finalize()
    else:execute('refs' if args.stage=='reference' else 'candidates',args.component)


if __name__=='__main__':main()
