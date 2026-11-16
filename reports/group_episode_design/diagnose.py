"""CPU-only sampling audit. Reads only original development labels; no fitting/scoring."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import audit_top5_references as audit

OUT = Path(__file__).resolve().parent

def run():
    split = ROOT/'artifacts/run_002/split_indices.npz'
    dev = np.sort(np.load(split)['development']).astype(np.int64)
    assert len(dev) == 6954 and audit.row_hash(dev) == audit.DEV_HASH
    train, test, y = audit.read_development_only(ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', dev)
    groups = train.PassengerId.iloc[dev].str.split('_').str[0].to_numpy()
    oof = ROOT/'artifacts/top5_reference_audit/mitra/group_20261112.npz'
    summary_path = oof.parent/'summary.json'
    summary = json.loads(summary_path.read_text())
    entry = next(row for row in summary['audits'] if row['kind'] == 'group' and row['seed'] == 20261112)
    assert audit.file_hash(oof) == entry['oof_sha256']
    z = np.load(oof)
    assert np.array_equal(z['dev_ids'], dev)
    assert np.array_equal(z['group_ids'].astype(str), groups)
    assert np.array_equal(z['passenger_ids'].astype(str), train.PassengerId.iloc[dev].astype(str))
    assignment = z['fold_id'].astype(np.int64)
    expected = np.full(len(dev), -1, dtype=np.int64)
    for fold, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=20261112).split(np.zeros(len(dev)), y, groups)):
        assert not set(groups[a]) & set(groups[b])
        expected[b] = fold
    np.testing.assert_array_equal(assignment, expected)
    result = dict(kind='sampling_only_no_model_fit_or_candidate_score', labels_read='original6954dev_only',
        dev_rows=len(dev), dev_ids_sha256=audit.row_hash(dev),
        group_ids_sha256=audit.string_hash(groups), fold_assignment_sha256=audit.row_hash(assignment),
        source_sha256={}, stages=[], episodes=[], cabin_group_audit={}, historical_runtime=[])
    sources = ['audit_top5_references.py', 'mitra_full_episode.py', 'mitra_full_episode_probe.py',
               'mitra_candidate.py', str(split.relative_to(ROOT)), str(oof.relative_to(ROOT)),
               str(summary_path.relative_to(ROOT))]
    for fold in range(3):
        outer = np.flatnonzero(assignment != fold)
        outer_query = np.flatnonzero(assignment == fold)
        seed = 20261112 + fold
        a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(outer)), y[outer], groups[outer]))
        inner, valid = outer[a], outer[b]
        assert not set(groups[inner]) & set(groups[valid])
        for stage, indices in [('inner', inner), ('refit', outer)]:
            rng = np.random.default_rng(seed)
            rows = []
            for step in range(1, 51):
                perm = rng.permutation(len(indices))
                query, old_support = perm[1024:1280], perm[:1024]
                blocked = np.isin(groups[indices][perm], groups[indices][query])
                support = perm[~blocked][:1024]
                assert len(support) == 1024
                assert not set(groups[indices][support]) & set(groups[indices][query])
                assert not set(dev[indices[support]]) & set(dev[indices[query]])
                rows.append([int((~blocked).sum()), int(np.isin(groups[indices][query], groups[indices][old_support]).sum()),
                             1024-len(set(support) & set(old_support))])
                result['episodes'].append(dict(fold=fold, stage=stage, step=step, seed=seed,
                    support_ids_sha256=audit.row_hash(dev[indices[support]]),
                    query_ids_sha256=audit.row_hash(dev[indices[query]]),
                    permutation_sha256=audit.row_hash(perm), eligible_support_rows=rows[-1][0],
                    old_query_with_support_group=rows[-1][1], replaced_support_rows=rows[-1][2],
                    disjoint_ids=True, disjoint_groups=True, query_positions_unchanged=True))
            arr = np.asarray(rows)
            result['stages'].append(dict(fold=fold, stage=stage, rows=len(indices), validation_rows=len(valid),
                train_ids_sha256=audit.row_hash(dev[indices]), inner_validation_ids_sha256=audit.row_hash(dev[valid]),
                outer_query_ids_sha256=audit.row_hash(dev[outer_query]),
                min_eligible_support=int(arr[:,0].min()), old_query_samegroup_support_fraction=float(arr[:,1].sum()/12800),
                mean_replaced_support=float(arr[:,2].mean()), max_replaced_support=int(arr[:,2].max())))
    for name, frame in [('train',train), ('development',train.iloc[dev]), ('test',test)]:
        known = frame.loc[frame.Cabin.notna(), ['PassengerId','Cabin']].copy()
        known['GroupId'] = known.PassengerId.str.split('_').str[0]
        counts = known.groupby('Cabin').GroupId.nunique()
        result['cabin_group_audit'][name] = dict(known_cabins=len(counts), cabins_spanning_groups=int((counts>1).sum()))
    result['cabin_group_audit']['train_test_group_overlap'] = len(set(train.PassengerId.str.split('_').str[0]) & set(test.PassengerId.str.split('_').str[0]))
    result['cabin_group_audit']['train_test_known_cabin_overlap'] = len(set(train.Cabin.dropna()) & set(test.Cabin.dropna()))
    cabins = train.Cabin.iloc[dev]
    result['cabin_group_audit']['outer_fold_known_cabin_overlap'] = [len(set(cabins[assignment!=f].dropna()) & set(cabins[assignment==f].dropna())) for f in range(3)]
    for path in sorted((ROOT/'reports/run_046').glob('v81*.json')):
        record=json.loads(path.read_text())
        result['historical_runtime'].append(dict(path=str(path.relative_to(ROOT)), selected_steps=record['selected_steps'],
            seconds=record['seconds'], inner_step_median_seconds=float(np.median([r['total_seconds'] for r in record['inner_history']]))))
        sources.append(str(path.relative_to(ROOT)))
    result['historical_total_hours'] = sum(r['seconds'] for r in result['historical_runtime'])/3600
    for path in sources:
        result['source_sha256'][path]=audit.file_hash(ROOT/path)
    result['source_sha256'][str(Path(__file__).relative_to(ROOT))]=audit.file_hash(__file__)
    result['episodes_checked']=len(result['episodes'])
    assert result['episodes_checked']==300
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['episodes_checked','stages','cabin_group_audit','historical_total_hours']},indent=2))

if __name__=='__main__':
    run()
