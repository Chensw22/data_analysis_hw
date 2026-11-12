"""Fixed Mitra-generation batch under the unchanged group-primary rules.

The old phase and its two failed decisions are immutable. Its matched reference
OOF is copied with provenance; no reference models are trained by this driver.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import types

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import top5_group_evaluate as previous

ROOT = Path(__file__).resolve().parent
PHASE = ROOT/'artifacts/top5_generation_phase'
POLICY = 'reports/top5_generation_validation_review.json'
AUTH = 'reports/top5_generation_authorization.json'
CANDIDATES = {
    'v1': dict(version='v78_Mitra1_zeroshot',module='mitra_generation_candidate',run='run_044'),
    'mean': dict(version='v79_Mitra12_mean',module='mitra_generation_candidate',run='run_044')}
REFERENCES = copy.deepcopy(previous.REFERENCES)
SEEDS = list(previous.SEEDS)
DISCLOSURE = ('A new fixed model batch under unchanged group-primary rules, not '
    'another validation-design revision. All6954 development labels and group '
    'partitions12/15/16 have already been used. Previous mirror/context A_false '
    'decisions remain unchanged. No independent confirmation is claimed.')

# An isolated namespace reuses the reviewed driver without mutating its module
# globals, source files, runtime state or old phase. It never invokes old freeze.
_impl = types.ModuleType('_top5_generation_reused_helpers')
_impl.__file__ = str(ROOT/'top5_group_evaluate.py')
exec(compile((ROOT/'top5_group_evaluate.py').read_text(),_impl.__file__,'exec'),_impl.__dict__)
_impl.PHASE, _impl.POLICY, _impl.AUTH = PHASE, POLICY, AUTH
_impl.CANDIDATES, _impl.REFERENCES = CANDIDATES, REFERENCES
_impl.DISCLOSURE = DISCLOSURE
OWN_SOURCES = sorted(set(previous.OWN_SOURCES + [
    'top5_generation_evaluate.py','test_top5_generation_evaluate.py',
    POLICY,AUTH,'reports/top5_mitra_v1_review.json']))
_impl.OWN_SOURCES = OWN_SOURCES
digest,write_json = previous.digest,previous.write_json


def check_policy():
    policy,auth = _impl.check_policy()
    old = json.loads((ROOT/previous.POLICY).read_text())
    for section in ['A_research_online_gate','S_stable_local_superiority_gate','bootstrap_recipe']:
        if policy[section] != old[section]:
            raise RuntimeError('This batch may not change an existing gate or bootstrap rule.')
    scope=policy['fixed_scope_before_new_scores']
    for key in ['group_split_seeds','new_primary_seeds','folds_per_seed','splitter','group_key','fit_seed']:
        if scope[key] != old['fixed_scope_before_new_scores'][key]:
            raise RuntimeError('Validation design changed instead of adding the fixed model batch.')
    if policy['batch_spec']['reference_training_allowed'] is not False:
        raise RuntimeError('Reference retraining is not part of this batch.')
    return policy,auth


# Preserve the inherited hash/data/model checks, adding the unchanged-rule gate.
_base_assert_phase = _impl.assert_phase


def assert_phase(env=None):
    check_policy()
    record=_base_assert_phase(env)
    if set(record['reference_copy_manifest'])!=set(REFERENCES):
        raise RuntimeError('Both matched reference copy manifests are required.')
    for key,entries in record['reference_copy_manifest'].items():
        if [entry['seed'] for entry in entries]!=SEEDS:
            raise RuntimeError('Reference copy manifest omits a required seed.')
        for entry in entries:
            target=PHASE/'refs'/key/f'group_{entry["seed"]}.npz'
            if digest(target)!=entry['oof_sha256'] or digest(ROOT/entry['original_oof_path'])!=entry['oof_sha256']:
                raise RuntimeError('Copied reference differs from the authenticated original.')
    return record


_impl.assert_phase = assert_phase


def reference_evidence():
    """Verify the entire old completed queue before authenticating its refs."""
    record=previous.assert_phase()
    summaries,_,_,_=previous.verified_queue(record)
    path=previous.PHASE/'eligibility.json'
    if digest(path)!=(previous.PHASE/'eligibility.sha256').read_text().strip():
        raise RuntimeError('Previous phase decision hash differs.')
    decision=json.loads(path.read_text())
    if (decision['status']!='complete' or not decision['fixed_queue_complete']
            or decision['phase_sha256']!=digest(previous.PHASE/'protocol.json')
            or set(decision['candidates'])!=set(previous.CANDIDATES)):
        raise RuntimeError('Previous fixed phase is incomplete or changed.')
    for (kind,key), summary in summaries.items():
        if decision['component_summary_sha256'][f'{kind}/{key}']!=digest(previous.PHASE/kind/key/'summary.json'):
            raise RuntimeError('Prior component changed after its decision.')
    return record,summaries


def freeze():
    policy,auth=check_policy()
    if PHASE.exists():raise RuntimeError('New phase exists; no retrospective overwrite.')
    old,summaries=reference_evidence()
    protocol_path=ROOT/'artifacts/run_044/protocol.json'
    screen_protocol=json.loads(protocol_path.read_text())
    previous.audit.check_historical_sources(screen_protocol)
    # Screen must have frozen this new batch policy, evaluator and authorization.
    for source in ['top5_generation_evaluate.py','test_top5_generation_evaluate.py',POLICY,AUTH]:
        if screen_protocol['sources'].get(source)!=digest(ROOT/source):
            raise RuntimeError(f'run044 did not prefreeze required batch source: {source}')
    sources={**old['sources'],**screen_protocol['sources']}
    for source in OWN_SOURCES:sources[source]=digest(ROOT/source)
    commit=previous.require_committed(sources)
    import iterate
    matrices,dev,y,groups,passengers=previous.audit.load_audit_data(iterate)
    if previous.data_identity(matrices,dev,y,groups,passengers)!=old['data']:
        raise RuntimeError('Reference and new batch development attributes differ.')
    historical=dict(old['historical_files'])
    for path in [protocol_path,protocol_path.parent/'screen_results.csv',
                 previous.PHASE/'protocol.json',previous.PHASE/'protocol.sha256',
                 previous.PHASE/'eligibility.json',previous.PHASE/'eligibility.sha256']:
        historical[str(path.relative_to(ROOT))]=digest(path)
    screen={}
    table=pd.read_csv(protocol_path.parent/'screen_results.csv')
    for key,spec in CANDIDATES.items():
        rows=table.loc[table.version.eq(spec['version'])]
        if len(rows)!=1:raise RuntimeError('Both fixed screen recipes must be recorded exactly once.')
        path=protocol_path.parent/f'{spec["version"]}_seed20261112.npz'
        with np.load(path,allow_pickle=False) as saved:p=previous.probabilities(saved['prob'],len(y))
        accuracy=float(accuracy_score(y,p>=.5));loss=float(log_loss(y,p,labels=[0,1]))
        if abs(accuracy-float(rows.iloc[0].accuracy))>1e-12 or abs(loss-float(rows.iloc[0].log_loss))>1e-12:
            raise RuntimeError('Screen CSV does not match source-bound OOF.')
        relative=str(path.relative_to(ROOT));historical[relative]=digest(path)
        screen[key]=dict(version=spec['version'],oof_path=relative,oof_sha256=historical[relative],
            accuracy=accuracy,log_loss=loss,development_ids_sha256=previous.audit.row_hash(dev))
    refs={}
    for key in REFERENCES:
        summary=summaries[('refs',key)]
        refs[key]=copy.deepcopy(summary)
        folder=previous.PHASE/'refs'/key
        for path in [folder/'summary.json',folder/'protocol.json']:
            historical[str(path.relative_to(ROOT))]=digest(path)
        for row in summary['results']:
            seed=row['seed']
            assignment=previous.fold_assignment(old,seed,dev,groups,passengers)
            path=ROOT/row['oof_path']
            p=previous.artifact_probability(path,row['oof_sha256'],dev,groups,passengers,assignment)
            if abs(accuracy_score(y,p>=.5)-row['accuracy'])>1e-12:
                raise RuntimeError('Reference score does not match SHA-authenticated OOF.')
            historical[row['oof_path']]=row['oof_sha256']
    # Metadata-only environment inventory, identical to the completed phase.
    # Candidate execution later requires the exact frozen Mitra environment.
    if previous.environment()!=old['environments']['mitra']:
        raise RuntimeError('Freeze this batch in the unchanged .venv-mitra environment.')
    PHASE.mkdir(parents=True)
    for source in sources:
        target=PHASE/'source'/source;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,target)
    for seed in SEEDS:
        shutil.copy2(previous.PHASE/f'folds_{seed}.npz',PHASE/f'folds_{seed}.npz')
    record=dict(status='frozen_before_new_group_fits',committed_at=commit,sources=sources,
        historical_files=historical,candidates=CANDIDATES,references=REFERENCES,seeds=SEEDS,
        budgets_seconds=auth['budgets_seconds'],environments=old['environments'],data=old['data'],
        fold_hashes=old['fold_hashes'],screen_evidence=screen,
        legacy_caches={k:v for k,v in old['legacy_caches'].items() if k.startswith('refs/')},
        disclosure=DISCLOSURE,queue_completion_required_before_any_eligibility=True,
        fixed_primary_metric='accuracy at0.5',bootstrap_replicates=2000,bootstrap_seed=20261119,
        reference_copy_policy='All3seeds copied bitwise; legacy_reused describes original provenance, not new training.',
        reference_copy_manifest={key:[dict(seed=row['seed'],original_oof_path=row['oof_path'],
            oof_sha256=row['oof_sha256']) for row in summary['results']] for key,summary in refs.items()},
        reference_phase_sha256=digest(previous.PHASE/'protocol.json'),reference_training_allowed=False)
    write_json(PHASE/'protocol.json',record)
    (PHASE/'protocol.sha256').write_text(digest(PHASE/'protocol.json')+'\n')
    copy_references(refs,auth)
    assert_phase('mitra')
    print('Frozen generation batch; all reference OOF copied without fitting.',flush=True)


def copy_references(refs,auth):
    """Copy authenticated rows; preserve their original fit-boundary evidence."""
    for key,summary in refs.items():
        folder=PHASE/'refs'/key;folder.mkdir(parents=True)
        copied=copy.deepcopy(summary)
        for row in copied['results']:
            source=ROOT/row['oof_path'];target=folder/f'group_{row["seed"]}.npz'
            shutil.copy2(source,target)
            row.update(original_oof_path=str(source.relative_to(ROOT)),
                oof_path=str(target.relative_to(ROOT)),reference_copied_without_training=True)
            if digest(target)!=row['oof_sha256']:raise RuntimeError('Reference copy differs.')
        execution=dict(kind='refs',key=key,version=REFERENCES[key]['version'],
            phase_sha256=digest(PHASE/'protocol.json'),environment=key,
            budget_seconds=auth['budgets_seconds']['refs'][key],
            execution='CPU4' if key=='v44' else 'MPS',
            execution_mode='reference_copy_no_fit; execution field identifies original model backend',
            original_summary_sha256=digest(previous.PHASE/'refs'/key/'summary.json'))
        write_json(folder/'protocol.json',execution)
        copied.update(phase_sha256=digest(PHASE/'protocol.json'),
            execution_protocol_sha256=digest(folder/'protocol.json'),total_seconds=0,
            reference_copied_without_training=True,original_summary_sha256=execution['original_summary_sha256'])
        write_json(folder/'summary.json',copied)
def candidate(key):
    if key not in CANDIDATES:raise ValueError('Only the two fixed new recipes are allowed.')
    return _impl.execute('candidates',key)


def complete_queue(record):return _impl.complete_queue(record)
def make_decision(screen,rows,intervals):return _impl.make_decision(screen,rows,intervals)
def assert_eligibility(key):return _impl.assert_eligibility(key)
def finalize():return _impl.finalize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','candidate','finalize'])
    parser.add_argument('component',nargs='?')
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='finalize':finalize()
    else:candidate(args.component)


if __name__=='__main__':main()
