"""Post-completion descriptive error audit; never selects weights or fits a model."""
import json
from pathlib import Path

import iltm_r128bn_cv_false as candidate

ROOT=Path(__file__).resolve().parent


def main():
    candidate.parent_environment()  # FALSE before importing numerical/data helpers.
    candidate.resource.require_committed(['iltm_error_complementarity.py'])
    out=candidate.OUTPUT/'random';public=candidate.REPORT/'random'
    evidence={}
    for name in ['protocol.json','summary.json','assessment.json',*[f'fold{k}.json' for k in range(3)]]:
        path=public/name;candidate.resource.require_committed([str(path.relative_to(ROOT))])
        if candidate.sha(path)!=candidate.sha(out/name):raise RuntimeError('Completed report/artifact differs.')
        evidence[str(path.relative_to(ROOT))]=candidate.sha(path)
    summary=candidate.read(out/'summary.json');protocol=candidate.read(out/'protocol.json')
    if (summary.get('status')!='complete' or summary.get('completed_folds')!=3
            or summary.get('candidate')!=candidate.NAME or protocol.get('phase')!='random'):
        raise RuntimeError('Only a completed fixed random three-fold result can be diagnosed.')
    candidate.assert_frozen(protocol['frozen'])
    for name,digest in summary['result_files_sha256'].items():
        if candidate.sha(out/name)!=digest:raise RuntimeError('Completed OOF evidence changed.')
        evidence[str((out/name).relative_to(ROOT))]=digest
    import numpy as np
    import error_overlap_diagnostic as diagnostic
    inputs,splits,folds,probabilities,files=diagnostic.load_authenticated();evidence.update(files)
    with np.load(out/'oof.npz',allow_pickle=False) as cache:
        for key,expected in [('dev_ids',inputs['dev']),('fold_id',folds),
                             ('passenger_ids',inputs['passengers']),('group_ids',inputs['groups'])]:
            if not np.array_equal(cache[key],expected):raise RuntimeError('New OOF alignment differs: '+key)
        p=cache['probability'].copy()
    for fold,(a,b) in enumerate(splits):
        folder=out/f'fold{fold}';record=candidate.read(out/f'fold{fold}.json')
        for name,digest in record['files_sha256'].items():
            if candidate.sha(folder/name)!=digest:raise RuntimeError('Native fold evidence changed.')
        meta=candidate.read(folder/'input.json');worker=candidate.read(folder/'worker_result.json')
        for key,expected in [('training_ids_sha256',inputs['dev'][a]),('query_ids_sha256',inputs['dev'][b]),
                             ('training_labels_sha256',inputs['y'][a].astype(np.int64))]:
            if meta[key]!=candidate.resource.array_hash(expected):raise RuntimeError('Native fold row/label mismatch.')
        q=candidate.validate_worker(worker,meta,folder)
        if (not np.array_equal(q,p[b]) or worker['entry_environment_snapshot']['duplicate_runtime_flag']!='FALSE'
                or worker['entry_environment_snapshot']['numeric_modules_already_loaded']
                or 'after_release' not in worker['runtime_flag_checks']):
            raise RuntimeError('Complete FALSE worker evidence/OOF mismatch.')
    y=inputs['y'];metrics=diagnostic.data.metrics(p,y,splits)
    if metrics!=summary['metrics']:raise RuntimeError('Candidate aggregate metrics differ.')
    assessment=candidate.assess('random',metrics,{})
    if assessment!=candidate.read(out/'assessment.json'):raise RuntimeError('Predeclared assessment differs.')
    old_wrong=np.column_stack([(probabilities[name]>=.5)!=y for name in diagnostic.FAMILIES]).sum(axis=1)
    if [int((old_wrong==k).sum()) for k in range(6)]!=[5263,248,182,158,189,914]:
        raise RuntimeError('Historical five-family grouping changed.')
    new_correct=(p>=.5)==y;rows=[]
    for name,mask in [('all_five_wrong',old_wrong==5),('all_five_correct',old_wrong==0),
                      ('families_disagree',(old_wrong>0)&(old_wrong<5))]:
        rows.append(dict(group=name,rows=int(mask.sum()),candidate_correct=int(new_correct[mask].sum()),
            candidate_wrong=int((~new_correct[mask]).sum()),candidate_accuracy=float(new_correct[mask].mean())))
    if sum(r['candidate_correct'] for r in rows)!=int(new_correct.sum()):raise RuntimeError('Group accounting differs.')
    result=dict(kind='descriptive_error_overlap_after_complete_fixed_recipe_not_candidate_selection',
        candidate=candidate.NAME,rows=len(y),metrics=metrics,predeclared_assessment=assessment,
        historical_family_groups=rows,
        paired_with_historical_families={name:candidate.paired(p,probabilities[name],y,splits) for name in diagnostic.FAMILIES},
        no_blend_or_threshold_search=True,no_upload_decision=True,old_holdout_targets_read=False,
        interpretation='Repairing shared errors is descriptive complementarity, not a validated ensemble or proof of stable superiority. These are reused development folds.',
        evidence_sha256=evidence,source_sha256=candidate.sha(Path(__file__)))
    candidate.write(public/'error_complementarity.json',result)
    print(json.dumps({k:result[k] for k in ['metrics','historical_family_groups']},indent=2))


if __name__=='__main__':main()
