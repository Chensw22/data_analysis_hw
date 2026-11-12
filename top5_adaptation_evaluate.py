"""One existing max100 candidate, unchanged group protocol, no new screen fit.

The run042 screen was observed before this plan. This driver freezes the new
group evaluation prospectively and makes no retrospective preregistration claim.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import types

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT=Path(__file__).resolve().parent
PHASE=ROOT/'artifacts/top5_adaptation_phase'
POLICY='reports/top5_adaptation_review.json'
AUTH='reports/top5_adaptation_authorization.json'
CANDIDATES={'adapt':dict(version='v76_Mitra_lastblock100',module='mitra_lastblock100_candidate',run='run_042')}
DISCLOSURE=('One bounded adaptive evaluation of existing run042 after its random screen '
    'was observed. Same reused6954development labels, group12/15/16, A/S gates and '
    'max100 recipe. Not prescreen preregistration, an independent holdout or a new '
    'validation-design change. Prior failed phases remain unchanged.')

# Independent copies of the already-reviewed helper namespaces; no mutation of
# imported old modules, files, phases or original candidate runtime settings.
_g=types.ModuleType('_adaptation_generation_helpers')
_g.__file__=str(ROOT/'top5_generation_evaluate.py')
exec(compile((ROOT/'top5_generation_evaluate.py').read_text(),_g.__file__,'exec'),_g.__dict__)
REFERENCES,SEEDS=copy.deepcopy(_g.REFERENCES),list(_g.SEEDS)
OWN_SOURCES=sorted(set(_g.OWN_SOURCES+[
    'top5_adaptation_evaluate.py','test_top5_adaptation_evaluate.py',POLICY,AUTH]))
for module in [_g,_g._impl]:
    module.PHASE,module.POLICY,module.AUTH=PHASE,POLICY,AUTH
    module.CANDIDATES,module.DISCLOSURE=CANDIDATES,DISCLOSURE
    module.OWN_SOURCES=OWN_SOURCES
digest,write_json=_g.digest,_g.write_json
previous=_g.previous


def check_policy():
    policy,auth=_g.check_policy()
    if (auth['budgets_seconds']['candidates']!={'adapt':21600}
            or policy['resource_limit']['candidate_total_seconds']!=21600
            or policy['resource_limit']['weekly_reserve_percent']!=12
            or policy['batch_spec']['new_screen_training_allowed'] is not False):
        raise RuntimeError('Only the bounded21600second/no-new-screen follow-up is authorized.')
    return policy,auth


def assert_phase(env=None):
    check_policy()
    return _g.assert_phase(env)


_g._impl.assert_phase=assert_phase


def freeze():
    _,auth=check_policy()
    if PHASE.exists():raise RuntimeError('Existing adaptation phase must not be overwritten.')
    old,summaries=_g.reference_evidence()
    history=ROOT/'artifacts/run_042'
    old_screen=json.loads((history/'protocol.json').read_text())
    previous.audit.check_historical_sources(old_screen)
    version=CANDIDATES['adapt']['version']
    if version not in old_screen['configs']:
        raise RuntimeError('Expected unchanged existing max100 candidate.')
    sources={**old['sources'],**old_screen['sources']}
    for source in OWN_SOURCES:sources[source]=digest(ROOT/source)
    commit=previous.require_committed(sources)
    import iterate
    matrices,dev,y,groups,passengers=previous.audit.load_audit_data(iterate)
    if previous.data_identity(matrices,dev,y,groups,passengers)!=old['data']:
        raise RuntimeError('Data or original reference attributes changed.')
    historical=dict(old['historical_files'])
    for path in [history/'protocol.json',history/'screen_results.csv',
                 previous.PHASE/'protocol.json',previous.PHASE/'protocol.sha256',
                 previous.PHASE/'eligibility.json',previous.PHASE/'eligibility.sha256']:
        historical[str(path.relative_to(ROOT))]=digest(path)
    table=pd.read_csv(history/'screen_results.csv')
    rows=table.loc[table.version.eq(version)]
    if len(rows)!=1:raise RuntimeError('Original candidate screen row is missing/duplicated.')
    path=history/f'{version}_seed20261112.npz'
    with np.load(path,allow_pickle=False) as saved:p=previous.probabilities(saved['prob'],len(y))
    accuracy=float(accuracy_score(y,p>=.5));loss=float(log_loss(y,p,labels=[0,1]))
    if abs(accuracy-float(rows.iloc[0].accuracy))>1e-12 or abs(loss-float(rows.iloc[0].log_loss))>1e-12:
        raise RuntimeError('Existing screen CSV/OOF evidence differs.')
    relative=str(path.relative_to(ROOT));historical[relative]=digest(path)
    screen={'adapt':dict(version=version,oof_path=relative,oof_sha256=historical[relative],
        accuracy=accuracy,log_loss=loss,development_ids_sha256=previous.audit.row_hash(dev),
        observed_before_this_group_plan=True)}
    refs={}
    for key in REFERENCES:
        refs[key]=copy.deepcopy(summaries[('refs',key)])
        folder=previous.PHASE/'refs'/key
        for path in [folder/'summary.json',folder/'protocol.json']:
            historical[str(path.relative_to(ROOT))]=digest(path)
        for row in refs[key]['results']:
            assignment=previous.fold_assignment(old,row['seed'],dev,groups,passengers)
            probability=previous.artifact_probability(ROOT/row['oof_path'],row['oof_sha256'],
                dev,groups,passengers,assignment)
            if abs(accuracy_score(y,probability>=.5)-row['accuracy'])>1e-12:
                raise RuntimeError('Reference OOF metric differs.')
            historical[row['oof_path']]=row['oof_sha256']
    if previous.environment()!=old['environments']['mitra']:
        raise RuntimeError('Use unchanged .venv-mitra for phase freeze.')
    PHASE.mkdir(parents=True)
    for source in sources:
        target=PHASE/'source'/source;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,target)
    for seed in SEEDS:shutil.copy2(previous.PHASE/f'folds_{seed}.npz',PHASE/f'folds_{seed}.npz')
    record=dict(status='frozen_before_adaptation_group_fits',committed_at=commit,
        sources=sources,historical_files=historical,candidates=CANDIDATES,references=REFERENCES,
        seeds=SEEDS,budgets_seconds=auth['budgets_seconds'],environments=old['environments'],
        data=old['data'],fold_hashes=old['fold_hashes'],screen_evidence=screen,
        legacy_caches={k:v for k,v in old['legacy_caches'].items() if k.startswith('refs/')},
        disclosure=DISCLOSURE,queue_completion_required_before_any_eligibility=True,
        fixed_primary_metric='accuracy at0.5',bootstrap_replicates=2000,bootstrap_seed=20261119,
        reference_copy_policy='All3seeds copied bitwise; no reference fitting.',
        reference_copy_manifest={key:[dict(seed=row['seed'],original_oof_path=row['oof_path'],
            oof_sha256=row['oof_sha256']) for row in summary['results']] for key,summary in refs.items()},
        reference_training_allowed=False,new_screen_training_allowed=False,
        original_screen_observed_before_this_plan=True,weekly_reserve_percent=12)
    write_json(PHASE/'protocol.json',record)
    (PHASE/'protocol.sha256').write_text(digest(PHASE/'protocol.json')+'\n')
    _g.copy_references(refs,auth)
    assert_phase('mitra')
    print('Frozen bounded adaptation group check; refs copied, no models fitted.',flush=True)


def candidate(key='adapt'):
    if key!='adapt':raise ValueError('Only existing v76 is allowed.')
    return _g._impl.execute('candidates',key)


def complete_queue(record):return _g.complete_queue(record)
def make_decision(screen,rows,intervals):return _g.make_decision(screen,rows,intervals)
def assert_eligibility(key):return _g.assert_eligibility(key)
def finalize():return _g.finalize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','candidate','finalize'])
    parser.add_argument('component',nargs='?',default='adapt')
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='finalize':finalize()
    else:candidate(args.component)


if __name__=='__main__':main()
