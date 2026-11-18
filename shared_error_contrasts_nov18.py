"""Fixed descriptive contrasts of historical shared errors; never fit a model."""
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pandas as pd

import error_overlap_diagnostic as authenticated

ROOT = Path(__file__).resolve().parent
PLAN = ROOT/'reports/error_targeted_20261118/contrast_protocol.json'
OUT = ROOT/'reports/error_targeted_20261118/contrast_result.json'
CATALOG = ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv'
STRATA = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'raw_prediction']
NUMERIC = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','CabinNumber']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def keys(frame, columns):
    # No rounding, quantile binning, numerical tolerance, or approximate match.
    return pd.Series([tuple(None if pd.isna(v) else v for v in row)
        for row in frame[columns].itertuples(index=False,name=None)],index=frame.index)


def matched_contrast(frame, cases, controls, columns=STRATA):
    k = keys(frame, columns)
    counts = k[cases].value_counts(); available = k[controls].value_counts()
    common = set(counts.index) & set(available.index)
    cm = cases & k.isin(common).to_numpy(); tm = controls & k.isin(common).to_numpy()
    # Each stratum's total control weight equals its case count. No fitted propensity.
    w = np.array([float(counts[key]/available[key]) if keep else 0.
                  for key,keep in zip(k,tm)])
    result = dict(case_rows=int(cases.sum()), control_pool_rows=int(controls.sum()),
        matched_case_rows=int(cm.sum()), matched_control_rows=int(tm.sum()),
        unmatched_case_rows=int((cases & ~cm).sum()), matched_strata=len(common),
        control_weight_sum=float(w.sum()),
        control_effective_n=float(w.sum()**2/(w@w)) if (w@w)>0 else 0., fields={})
    if not cm.any():return result
    cw = cm.astype(float)
    for col in frame.columns:
        if col == 'raw_prediction':continue
        if col in NUMERIC:
            values = frame[col].to_numpy(dtype=float)
            known = np.isfinite(values)
            def stat(weights):
                denom = weights.sum(); kw = weights*known
                return dict(missing_fraction=float(weights[~known].sum()/denom),
                    observed_mean=float(np.nansum(values*kw)/kw.sum()) if kw.sum() else None)
            result['fields'][col] = dict(cases=stat(cw), weighted_controls=stat(w))
        else:
            values = frame[col].fillna('__MISSING__').astype(str)
            result['fields'][col] = {value:dict(case_fraction=float(cw[values.eq(value)].sum()/cw.sum()),
                weighted_control_fraction=float(w[values.eq(value)].sum()/w.sum()))
                for value in sorted(values[cm|tm].unique())}
    return result


def exact_conflicts(frame, y, hard, raw_correct, columns):
    k = keys(frame, columns)
    table = pd.DataFrame({'key':k,'y':y,'hard':hard,'raw_correct':raw_correct})
    grouped = table.groupby('key',sort=False,dropna=False).agg(rows=('y','size'), positives=('y','sum'),
        hard_rows=('hard','sum'), correct_rows=('raw_correct','sum'))
    conflict = (grouped.positives>0)&(grouped.positives<grouped.rows)
    shared = (grouped.hard_rows>0)&(grouped.correct_rows>0)
    # Same raw predicted class is required separately, not silently inferred.
    pred = frame.raw_prediction.to_numpy()
    joint = table.assign(pred=pred).groupby(['key','pred'],sort=False,dropna=False).agg(
        hard_rows=('hard','sum'), correct_rows=('raw_correct','sum'))
    matched = (joint.hard_rows>0)&(joint.correct_rows>0)
    return dict(columns=columns, distinct_patterns=len(grouped),
        repeated_patterns=int((grouped.rows>1).sum()), repeated_pattern_rows=int(grouped.loc[grouped.rows>1,'rows'].sum()),
        conflicting_patterns=int(conflict.sum()), conflicting_pattern_rows=int(grouped.loc[conflict,'rows'].sum()),
        hard_rows_in_conflicting_patterns=int(grouped.loc[conflict,'hard_rows'].sum()),
        hard_rows_with_exact_attribute_raw_correct_peer=int(grouped.loc[shared,'hard_rows'].sum()),
        hard_rows_with_exact_attributes_and_same_raw_prediction_correct_peer=int(joint.loc[matched,'hard_rows'].sum()),
        note='Observed label conflicts within this representation, not mislabel proof or a Bayes/performance ceiling. Dropping CabinNumber deliberately coarsens information.')


def run():
    if OUT.exists():raise RuntimeError('No overwrite of completed diagnostic.')
    plan = json.loads(PLAN.read_text())
    names=[Path(__file__).name,str(PLAN.relative_to(ROOT)),*plan['source_dependencies']]
    tracked=set(filter(None,subprocess.check_output(['git','ls-files','-z','--',*names],cwd=ROOT).decode().split('\0')))
    if tracked!=set(names) or subprocess.check_output(['git','status','--porcelain','--',*names],cwd=ROOT).strip():
        raise RuntimeError('Commit fixed protocol/script/dependencies before execution.')
    for name,digest in plan['pinned_evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Pinned input changed: '+name)
    if plan['strata']!=STRATA or plan['numeric_columns']!=NUMERIC:raise RuntimeError('Fixed contrasts differ.')
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('CPU diagnostic exceeded120seconds.')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        inputs,splits,folds,prob,evidence=authenticated.load_authenticated()
        rows=pd.read_csv(CATALOG,dtype={'PassengerId':str})
        for column,values in [('row_id',inputs['dev']),('PassengerId',inputs['passengers']),('fold',folds),('y',inputs['y'])]:
            np.testing.assert_array_equal(rows[column].to_numpy(),values)
        y=np.asarray(inputs['y'])
        for name,p in prob.items():np.testing.assert_array_equal(rows['correct_'+name],(p>=.5)==y)
        correct_columns=[c for c in rows if c.startswith('correct_')]
        if len(correct_columns)!=24:raise RuntimeError('Expected historical24 recipes.')
        hard=~rows[correct_columns].to_numpy(dtype=bool).any(axis=1)
        np.testing.assert_array_equal(hard,rows.all_recipes_still_wrong)
        if hard.sum()!=546:raise RuntimeError('Historical all24 error population changed.')
        frame=inputs['features'].loc[inputs['dev']].reset_index(drop=True).copy()
        columns=list(frame.columns);frame['raw_prediction']=(prob['rawMitra']>=.5).astype(int)
        raw_correct=(prob['rawMitra']>=.5)==y
        all_correct=rows[correct_columns].to_numpy(dtype=bool).all(axis=1)
        sleep=frame.HomePlanet.eq('Earth').to_numpy() & frame.CryoSleep.eq('True').to_numpy()
        result=dict(status='complete_descriptive_only', rows=6954, hard_rows=546,
            raw_prediction_comparator='Authenticated rawMitra OOF threshold0.5, same originalSKF3seed20261112',
            matched_hard_vs_raw_correct=matched_contrast(frame,hard,raw_correct),
            matched_hard_vs_all24_correct=matched_contrast(frame,hard,all_correct),
            earth_asleep=dict(rows=int(sleep.sum()),negative_labels=int((sleep&(y==0)).sum()),
                positive_labels=int((sleep&(y==1)).sum()),hard_negative_rows=int((hard&sleep&(y==0)).sum()),
                descriptive_negative_vs_positive=matched_contrast(frame,sleep&(y==0),sleep&(y==1)),
                hard_negative_vs_raw_correct_positive=matched_contrast(frame,hard&sleep&(y==0),sleep&raw_correct&(y==1))),
            exact_conflicts=[exact_conflicts(frame,y,hard,raw_correct,cols)
                for cols in [columns,[c for c in columns if c!='CabinNumber']]],
            evidence_sha256={**evidence,**plan['pinned_evidence_sha256'],
                **{n:sha(ROOT/n) for n in names}}, protocol_sha256=sha(PLAN),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            seconds=time.monotonic()-started,limitations=plan['limitations'],
            next_information_hypothesis=plan['next_information_hypothesis'],
            recommendation='No automatic new candidate: descriptive contrasts require review against prior coverage. Error membership was never a training target or inference route.')
        # Do not emit IDs, labels, feature tuples, per-row values or a purported attainable oracle.
        for name,digest in plan['pinned_evidence_sha256'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Evidence changed during diagnostic.')
        OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'],hard_rows=546,seconds=result['seconds'],output=str(OUT.relative_to(ROOT)))))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':run()
