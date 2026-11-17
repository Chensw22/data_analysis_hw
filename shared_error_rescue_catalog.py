"""Authenticated historical OOF rescue catalog; no fits or deployable routing.

All labels/IDs/per-row masks stay in ignored artifacts. Representatives are fixed
by architecture and existing method_comparison, not their rescue counts. Original
ID-less caches are explicitly authenticated by provenance/control/fold metrics.
"""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import audit_top5_references as audit
import error_overlap_diagnostic as anchor_source
import three_family_mean_screen as common

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/error_rescue_20261117'
LOCAL=ROOT/'artifacts/error_rescue_20261117'
ANCHOR=ROOT/'artifacts/data_error_diagnosis_20261117/aligned_rows.csv'
# family, primary, run, version. Multiple recipes within a family are not votes.
REPRESENTATIVES=[
 ('LightGBM',True,'run_005','v18_lgb15'),
 ('SVM',True,'run_005','v20_svm3'),
 ('MLP',True,'run_005','v22_mlp'),
 ('EBM',True,'run_010','v29_ebm'),
 ('TabICL',True,'run_014','v37_TabICL_engineered4'),
 ('TabM',True,'run_016','v41_TabM'),
 ('TabPFN',False,'run_021','v48_TabPFN_finetuned1'),
 ('XGBoost',True,'run_022','v49_xgboost'),
 ('LightGBM',False,'run_027','v56_nested_linear_leaf'),
 ('TabPFN',True,'run_033','v65_TabPFN_default32'),
 ('FTTransformer',True,'run_035','v67_FTTransformer_default'),
 ('TabR',True,'run_036','v68_TabR_S'),
 ('MLP',False,'run_037','v69_matched_scratch_MLP'),
 ('MLP',False,'run_037','v70_SCARF'),
 ('AutoGluon_stack',True,'run_039','v73_AutoGluon_multilayer'),
 ('Mitra',False,'run_043','v77_Mitra_context20'),
 ('Mitra',False,'run_044','v78_Mitra1_zeroshot'),
]
ANCHORS=[('CatBoost_spatial',True,'v44'),('Mitra',True,'v81'),
 ('TabDPT',True,'TabDPT'),('Kumo',True,'Kumo'),('EXAONE',True,'EXAONE'),
 ('Mitra',False,'rawMitra'),('Mitra',False,'v93')]


def remember(path,evidence):
    evidence[str(Path(path).relative_to(ROOT))]=audit.file_hash(path)


def frozen_sources(folder,protocol,evidence):
    for relative,digest in protocol['sources'].items():
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid source path.')
        candidates=[folder/'source'/rel,folder/rel,folder/(rel.stem+'_source'+rel.suffix)]
        if relative=='iterate.py':candidates.append(folder/'iterate_source.py')
        if relative=='experiment.py':candidates.append(folder/'base_source.py')
        candidates.append(ROOT/rel)
        path=next((p for p in candidates if p.is_file() and audit.file_hash(p)==digest),None)
        if path is None:raise RuntimeError('Unavailable authenticated frozen source: '+relative)
        remember(path,evidence)


def function_ast(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return ast.dump(node,include_attributes=False)


def load_candidate(run,version,inputs,splits,baseline,evidence):
    folder=ROOT/'artifacts'/run
    common.check_complete(folder,version)
    for name in ('protocol.json','selection.json','screen_results.csv'):
        a,b=folder/name,ROOT/'reports'/run/name
        common.require_committed([str(b.relative_to(ROOT))])
        if audit.file_hash(a)!=audit.file_hash(b):raise RuntimeError('Public/artifact metadata differs: '+name)
        remember(a,evidence);remember(b,evidence)
    protocol=json.loads((folder/'protocol.json').read_text())
    if [protocol.get(k) for k in ('seed','folds','development_rows','threshold')]!=[20261112,3,6954,.5]:
        raise RuntimeError('Not original6954/SKF3/seed20261112/threshold.5.')
    if version not in protocol['configs']:raise RuntimeError('Missing configured recipe.')
    frozen_sources(folder,protocol,evidence)
    # This producer writes prob[b] for the identical SKF order; no row-ID guess.
    if function_ast(folder/'iterate_source.py','cross_validate')!=function_ast(ROOT/'artifacts/run_038/iterate_source.py','cross_validate'):
        raise RuntimeError('Cross-validation producer AST differs; manual row provenance needed.')
    control=folder/'v6_no_id_features_seed20261112.npz'
    with np.load(control,allow_pickle=False) as cache:
        if not np.array_equal(cache['prob'],baseline):raise RuntimeError('Complete v6 OOF not bitwise identical.')
    remember(control,evidence)
    path=folder/f'{version}_seed20261112.npz'
    with np.load(path,allow_pickle=False) as cache:
        p=np.asarray(cache['prob'],dtype=float)
        common.cached_fold_scores(p,cache['fold_accuracy'],inputs['y'],splits)
        embedded_ids='dev_ids' in cache.files
        if embedded_ids and not np.array_equal(cache['dev_ids'],inputs['dev']):raise RuntimeError('Embedded devIDs mismatch.')
    remember(path,evidence)
    row=pd.read_csv(folder/'screen_results.csv').query('version == @version')
    if (len(row)!=1 or abs(float(row.iloc[0].accuracy)-accuracy_score(inputs['y'],p>=.5))>1e-12
            or abs(float(row.iloc[0].log_loss)-log_loss(inputs['y'],p))>1e-7):
        raise RuntimeError('Aggregate metrics disagree with aligned cache.')
    return p,dict(run=run,version=version,cache_path=str(path.relative_to(ROOT)),cache_sha256=audit.file_hash(path),
        config=protocol['configs'][version],embedded_dev_ids=embedded_ids,
        alignment_assurance='Frozen producer AST + seed/fold protocol + bitwise identical full v6 control + all3 reconstructed fold scores; not embedded rowIDs' if not embedded_ids else 'Embedded devIDs plus frozen producer/control/metrics',
        interpretation='Historical development OOF only; no new fit or independent validation.')


def summarize(p,y,common_wrong,common_correct,v81_wrong,folds):
    correct=(p>=.5)==y
    repaired=int((correct&common_wrong).sum());broken=int((~correct&common_correct).sum())
    return dict(accuracy=float(correct.mean()),errors=int((~correct).sum()),
        repaired_five_common_wrong=repaired,repair_fraction_of914=repaired/914,
        damaged_five_common_correct=broken,damage_fraction_of5263=broken/5263,
        v81_errors_repaired=int((correct&v81_wrong).sum()),v81_correct_damaged=int((~correct&~v81_wrong).sum()),
        net_correct_vs_v81=int(correct.sum()-(~v81_wrong).sum()),
        rescue_by_fold=[int((correct&common_wrong&(folds==f)).sum()) for f in range(3)],
        collateral_by_fold=[int((~correct&common_correct&(folds==f)).sum()) for f in range(3)])


def main():
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    inputs,splits,folds,anchor_prob,evidence=anchor_source.load_authenticated()
    anchor=pd.read_csv(ANCHOR,dtype={'PassengerId':str})
    if (not np.array_equal(anchor.row_id,inputs['dev']) or not np.array_equal(anchor.y,inputs['y'])
            or not np.array_equal(anchor.fold,folds) or not np.array_equal(anchor.PassengerId,inputs['passengers'])):
        raise RuntimeError('Anchor row/label/fold identity differs.')
    for short,p in anchor_prob.items():
        if not np.allclose(anchor['p_'+short],p,rtol=0,atol=1e-15):raise RuntimeError('Anchor probabilities changed: '+short)
    y=inputs['y'];five_wrong=np.column_stack([(anchor_prob[k]>=.5)!=y for k in anchor_source.FAMILIES])
    wrong=five_wrong.all(axis=1);right=(~five_wrong).all(axis=1);v81wrong=(anchor_prob['v81']>=.5)!=y
    if wrong.sum()!=914 or right.sum()!=5263 or not np.array_equal(five_wrong.sum(axis=1),anchor.family_wrong_count):
        raise RuntimeError('Expected fixed914wrong/5263correct anchor.')
    for path in [ANCHOR,ROOT/'reports/data_error_diagnosis_20261117/error_overlap.json',ROOT/'reports/method_tiers.csv',ROOT/'reports/method_comparison.csv',Path(__file__),ROOT/'error_overlap_diagnostic.py',ROOT/'three_family_mean_screen.py',ROOT/'exaone_validation_data.py',ROOT/'prior_adapt_mean_screen.py',ROOT/'audit_top5_references.py']:
        remember(path,evidence)
    tiers=pd.read_csv(ROOT/'reports/method_tiers.csv');comparison=pd.read_csv(ROOT/'reports/method_comparison.csv')
    with np.load(ROOT/'artifacts/run_038/v6_no_id_features_seed20261112.npz') as z:baseline=z['prob'].copy()
    records=[];probabilities={};provenance=[];excluded=[]
    for family,primary,short in ANCHORS:
        p=anchor_prob[short];probabilities[short]=p
        records.append(dict(method=short,family=family,primary_family_representative=primary,anchor=True,
            **summarize(p,y,wrong,right,v81wrong,folds)))
    for family,primary,run,version in REPRESENTATIVES:
        try:
            p,record=load_candidate(run,version,inputs,splits,baseline,evidence)
            if version not in set(tiers.version)|set(comparison.version):raise RuntimeError('Not in existing method inventories.')
            probabilities[version]=p;provenance.append(record)
            records.append(dict(method=version,family=family,primary_family_representative=primary,anchor=False,
                **summarize(p,y,wrong,right,v81wrong,folds)))
        except (RuntimeError,ValueError,KeyError,FileNotFoundError,StopIteration) as error:
            excluded.append(dict(run=run,version=version,family=family,reason=str(error)))
    primary=[r['method'] for r in records if r['primary_family_representative']]
    primary_wrong=np.column_stack([(probabilities[n]>=.5)!=y for n in primary])
    all_wrong=np.column_stack([(p>=.5)!=y for p in probabilities.values()])
    rows=pd.DataFrame(records)
    # Unique rescue is descriptive relative to this declared set, not a selector.
    primary_rescue=np.column_stack([wrong&((probabilities[n]>=.5)==y) for n in primary])
    total=primary_rescue.sum(axis=1)
    rows['unique_rescue_among_primary_representatives']=[int((primary_rescue[:,primary.index(n)]&(total==1)).sum()) if n in primary else None for n in rows.method]
    rows.sort_values(['repaired_five_common_wrong','damaged_five_common_correct'],ascending=[False,True]).to_csv(OUT/'catalog_methods.csv',index=False)
    family_rows=[]
    for family in rows.family.unique():
        group=rows[rows.family==family];names=group.method.tolist()
        residual=np.column_stack([(probabilities[n]>=.5)!=y for n in names]).all(axis=1)
        family_rows.append(dict(family=family,primary=group.loc[group.primary_family_representative,'method'].tolist(),
            recipe_count=len(names),recipes=names,any_recipe_truth_aware_rescue=int((wrong&~residual).sum()),
            note='Union includes related variants; not independent votes or achievable router.'))
    pd.DataFrame(family_rows).to_csv(OUT/'catalog_families.csv',index=False)
    result=dict(status='complete_descriptive_only',rows=6954,threshold=.5,seed=20261112,folds=3,
        anchor_common_wrong=914,anchor_common_correct=5263,anchor_disagreement_rows=777,
        included_recipes=len(records),primary_representatives=primary,family_count=len(family_rows),
        primary_representative_intersection=dict(all_still_wrong=int(primary_wrong.all(axis=1).sum()),
            anchor914_rescued_by_any=int((wrong&~primary_wrong.all(axis=1)).sum()),
            truth_aware_oracle_accuracy=float(1-primary_wrong.all(axis=1).mean())),
        all_included_recipe_sensitivity=dict(all_still_wrong=int(all_wrong.all(axis=1).sum()),
            anchor914_rescued_by_any=int((wrong&~all_wrong.all(axis=1)).sum()),
            truth_aware_oracle_accuracy=float(1-all_wrong.all(axis=1).mean())),
        families=family_rows,included_provenance=provenance,excluded=excluded,
        scope_exclusions=['Other seeds/GroupCV/old holdout/in-sample/fullfit test predictions excluded.',
            'Other87 variations are not independent votes; this is a predeclared representative catalog.',
            'No isolated RF/XT OOF is claimed from the combined AutoGluon stack.',
            'Online-only v95/v96/v99 lack compatible development OOF; they cannot enter this comparison.'],
        limitations=['Oracle chooses using true labels, is not deployable or a target accuracy.',
            'More diverse but weaker methods may rescue914 while damaging many5263; rescue counts alone do not justify blending.',
            'These same development labels have been reused adaptively; no independent validation or new performance claim.',
            'ID-less old caches rely on frozen provenance, identical baseline and fold metrics, not embedded passenger identities.',
            'Family groups summarize methods, not statistical independence; AutoGluon contains some other listed learners.'],
        conclusions=[
            'The914 shared failures are not an established data ceiling: some are correct under other authenticated recipes.',
            'Every newly catalogued primary non-anchor representative damages more of the5263 common-correct rows than it rescues from914. Do not rank deployability by rescue count.',
            'SCARF/scratchMLP/TabR/FTTransformer show larger descriptive disagreement, but their overall errors exceed v81; this is evidence of complementary predictions, not a safe router.',
            'Extra within-family variants reduce the all-wrong intersection further, but increasing recipes inflates the truth-aware oracle; no extra independent votes are claimed.',
            'Full-data online-only count/peerCryo candidates have no matched development OOF and are excluded rather than inferred from their online scores.'
        ],evidence_sha256=evidence)
    primary_new=rows[rows.primary_family_representative & ~rows.anchor]
    if not (primary_new.damaged_five_common_correct>primary_new.repaired_five_common_wrong).all():
        raise RuntimeError('Qualitative conclusion no longer agrees with computed catalog.')
    audit.json_write(OUT/'catalog.json',result)
    local=anchor[['row_id','PassengerId','fold','y','family_wrong_count']].copy()
    for name,p in probabilities.items():local['correct_'+name]=(p>=.5)==y
    local['all_primary_still_wrong']=primary_wrong.all(axis=1);local['all_recipes_still_wrong']=all_wrong.all(axis=1)
    local.to_csv(LOCAL/'catalog_rows.csv',index=False)
    summary_lines=[
        '# 历史OOF共同错例修复目录', '',
        f'已认证{len(records)}个配方、{len(primary)}个预先指定模型族代表；全部同6954开发行、原SKF3及seed20261112。未训练模型。', '',
        f'原五族共同错914条中，按每族一个代表的描述性并集可修复{result["primary_representative_intersection"]["anchor914_rescued_by_any"]}条，仍共同错{result["primary_representative_intersection"]["all_still_wrong"]}条。纳入同族补充变体后，分别为{result["all_included_recipe_sensitivity"]["anchor914_rescued_by_any"]}和{result["all_included_recipe_sensitivity"]["all_still_wrong"]}条。', '',
        '这需要事后知道每行真实标签才能选择方法，绝不是可达准确率、拟合路由结果或数据上限。', '',
        '修复较多的SCARF、MLP、TabR、FTTransformer也损坏更多原五族共同正确行。每个新增主代表在这两个固定集合上的损坏数均大于修复数，故不支持按救错数盲目融合。', '',
        '完整修复/损坏、各折计数、相对v81配对差见catalog_methods.csv；模型族与同族变体见catalog_families.csv。只有ignored artifacts保存逐行ID、标签和正确性mask。', '',
        '旧NPZ多数不含ID；其认证依赖冻结源、相同CV producer AST、完整v6控制逐位一致和重建三折指标，明确不声称旧文件自带身份。线上only新配方没有同划分OOF，不纳入。', ''
    ]
    (OUT/'catalog_summary.md').write_text('\n'.join(summary_lines))
    print(json.dumps(dict(included=len(records),excluded=excluded,primary_intersection=result['primary_representative_intersection'],
        all_recipe_intersection=result['all_included_recipe_sensitivity'],top_rescuers=rows.sort_values('repaired_five_common_wrong',ascending=False).head(6).to_dict('records')),indent=2))


if __name__=='__main__':main()
