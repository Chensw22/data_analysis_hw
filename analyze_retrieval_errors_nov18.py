"""Read-only error accounting for the fixed completed retrieval fold; no models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/error_followup_20261118/retrieval_error_analysis.json'
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result_path=ROOT/'reports/error_followup_20261118/retrieval_fold0/result.json'
    result=json.loads(result_path.read_text());npz=ROOT/'artifacts/error_followup_20261118/retrieval_fold0/predictions.npz'
    if result['status']!='complete_single_fold_adaptive_diagnostic' or sha(npz)!=result['predictions_sha256']:raise RuntimeError('Incomplete or changed prediction evidence.')
    with np.load(npz,allow_pickle=False) as z:cache={k:z[k].copy() for k in z.files}
    catalog_path=ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv'
    catalog=pd.read_csv(catalog_path,dtype={'PassengerId':str}).set_index('row_id').loc[cache['dev_ids']]
    np.testing.assert_array_equal(catalog.PassengerId.to_numpy(),cache['passenger_ids'])
    np.testing.assert_array_equal(catalog.fold.to_numpy(),cache['fold_id'])
    np.testing.assert_array_equal(catalog.all_recipes_still_wrong.to_numpy(bool),cache['hard546'])
    y=catalog.y.to_numpy();p=cache['retrieval256_oneview'];raw=cache['rawMitra']
    good=(p>=.5)==y;old=(raw>=.5)==y
    repaired=good&~old;broken=~good&old;hard=cache['hard546']&good
    assert len(y)==2318 and repaired.sum()==70 and broken.sum()==80 and hard.sum()==3
    ids=set(cache['dev_ids'].tolist());train_path=ROOT/'data/raw/train.csv'
    # Read only these query rows and non-name attributes; all labels come from existing dev catalog.
    frame=pd.read_csv(train_path,usecols=['PassengerId','HomePlanet','CryoSleep','Destination','Age','VIP','Cabin',*SPEND],
        dtype={'PassengerId':str},skiprows=lambda line:line>0 and line-1 not in ids)
    frame=frame.set_index('PassengerId').loc[cache['passenger_ids']].reset_index()
    frame['Deck']=frame.Cabin.str.split('/').str[0]
    missing=frame[SPEND].isna().any(axis=1).to_numpy();total=frame[SPEND].sum(axis=1,min_count=5).to_numpy()
    complete_zero=~missing&(total==0)
    age=frame.Age.to_numpy()
    categories={name:frame[name].fillna('__MISSING__').astype(str).to_numpy() for name in ['HomePlanet','CryoSleep','Deck','Destination']}
    categories['age_band']=np.select([np.isnan(age),age<=12,age<=17,age<=29],['missing','0-12','13-17','18-29'],default='30+')
    categories['spend_status']=np.where(missing,'any_missing',np.where(complete_zero,'complete_zero','complete_positive'))
    earth=frame.HomePlanet.eq('Earth').to_numpy();sleep=frame.CryoSleep.eq(True).to_numpy();awake=frame.CryoSleep.eq(False).to_numpy()
    np.testing.assert_array_equal(earth&sleep,cache['Earth_sleep']);np.testing.assert_array_equal(earth&awake,cache['Earth_awake'])
    categories['planet_sleep']=np.where(earth&sleep,'Earth_sleep',np.where(earth&awake,'Earth_awake','other_or_unknown'))
    def summarise(mask):
        n=int(mask.sum());rc=np.maximum(raw,1-raw);pc=np.maximum(p,1-p)
        return dict(rows=n,actual_false=int((mask&(y==0)).sum()),actual_true=int((mask&(y==1)).sum()),
            complete_zero=int((mask&complete_zero).sum()),any_bill_missing=int((mask&missing).sum()),
            Earth_sleep=int((mask&earth&sleep).sum()),Earth_awake=int((mask&earth&awake).sum()),
            raw_confidence=dict(median=float(np.median(rc[mask])),q25=float(np.quantile(rc[mask],.25)),q75=float(np.quantile(rc[mask],.75)),at_least_08=int((mask&(rc>=.8)).sum())),
            retrieval_confidence=dict(median=float(np.median(pc[mask])),q25=float(np.quantile(pc[mask],.25)),q75=float(np.quantile(pc[mask],.75)),at_least_08=int((mask&(pc>=.8)).sum())))
    tables={}
    for feature,values in categories.items():
        tables[feature]=[]
        for value in sorted(set(values)):
            mask=values==value;r=int((mask&repaired).sum());b=int((mask&broken).sum())
            tables[feature].append(dict(value=value,query_rows=int(mask.sum()),repairs=r,breaks=b,net_correct=r-b,
                fraction_of70_repairs=r/70,fraction_of80_breaks=b/80,
                raw_errors=int((mask&~old).sum()),raw_correct=int((mask&old).sum())))
    hard_cases=[]
    for i in np.flatnonzero(hard):
        row=frame.iloc[i]
        hard_cases.append(dict(row_id=int(cache['dev_ids'][i]),PassengerId=row.PassengerId,
            HomePlanet=None if pd.isna(row.HomePlanet) else row.HomePlanet,
            CryoSleep=None if pd.isna(row.CryoSleep) else bool(row.CryoSleep),Age=None if pd.isna(row.Age) else float(row.Age),
            Cabin=None if pd.isna(row.Cabin) else row.Cabin,actual_transported=bool(y[i]),
            raw_probability=float(raw[i]),retrieval_probability=float(p[i]),
            complete_zero=bool(complete_zero[i]),any_bill_missing=bool(missing[i]),
            total_spend=None if not np.isfinite(total[i]) else float(total[i])))
    source_paths=[result_path,npz,catalog_path,train_path,Path(__file__).resolve()]
    report=dict(status='complete_readonly_accounting_not_new_candidate',source_sha256={str(path.relative_to(ROOT)):sha(path) for path in source_paths},
        scope='Only saved fold0 predictions, existing development catalog labels, and query-only raw attributes excluding Name/Transported. No fit, model load, threshold optimization or online routing.',
        rows=len(y),metrics=result['metrics'],relative_raw=dict(repairs=70,breaks=80,net_correct=-10,
            accuracy_delta_percentage_points=100*(-10/2318),logloss_delta=result['metrics']['retrieval256_oneview']['log_loss']-result['metrics']['rawMitra']['log_loss']),
        repair_profile=summarise(repaired),break_profile=summarise(broken),hard178_rescues=hard_cases,
        hard_accounting=dict(subset_rows=178,rescued=3,still_wrong=175,additional_repairs_outside_hard178=int((repaired&~cache['hard546']).sum()),breaks_outside_hard178=int((broken&~cache['hard546']).sum())),
        feature_tables=tables,explanation=[
            'The3 hard rescues are a subset of70 repairs;80 other predictions changed from correct to incorrect, so overall net is-10.',
            'Errors do not concentrate uniquely in Earth sleepers: that group repairs27/breaks23(net+4), Earth awake28/35(net-7), other/unknown15/22(net-7).',
            'Complete-positive-spend rows account for27 repairs and43 breaks(net-16); complete-zero30/29(net+1), any-bill-missing13/8(net+5). These are descriptive accounting identities, not a conditional deployment rule.',
            'Most changed cases lie near the decision boundary: median raw confidence .531 for repairs and .562 for breaks; retrieval medians .554 and .544. This is consistent with unstable local decision changes, not proof of a particular causal mechanism.',
            'The3 hard rescues are all actually False and Earth, but only1 is asleep/zero-spend;2 are awake with complete positive bills. No single zero-spend rule explains all3.'
        ],confidence_note='Confidence means max(p,1-p), a model probability magnitude, not evidence that estimates are calibrated.',
        limits=['Rescued hard cases are selected by outcomes after inference; their traits cannot define a reliable router or justify targeting only those rows.',
                'RawMitra is a different architecture and context recipe. This comparison does not isolate retrieval versus fullcontext; cached TabDPT-full8 also differs in ensemble count.',
                'Feature counts are descriptive within this repeatedly used single fold; sample counts and denominators are explicit, not causal explanations.',
                'Three rescues disprove universal irreparability of those cases, but do not offset80 new mistakes or establish a generalizable hidden rule.'])
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['relative_raw','repair_profile','break_profile','hard178_rescues','hard_accounting']},indent=2))


if __name__=='__main__':main()
