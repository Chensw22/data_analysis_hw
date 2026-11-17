"""Read-only Earth/G spatial residual audit of six frozen group OOF arrays.

Run: .venv-mitra/bin/python spatial_residual_diagnostic.py
No fit, new bins, cutpoint search, independent-holdout claim, or correction.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import audit_top5_references as audit
import top5_group_evaluate as group
import iterate
from data_quality_diagnostic import attributes

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/data_error_diagnosis_20261117/spatial_residual.json'
QUALITY=ROOT/'reports/data_error_diagnosis_20261117/data_quality.json'
SEEDS=[20261112,20261115,20261116]


def cluster_summary(y,p,groups):
    y,p=np.asarray(y,float),np.asarray(p,float)
    n=len(y)
    if not n:return {'rows':0,'groups':0,'target_rate':None,'mean_probability':None,'mean_residual_y_minus_p':None,'cluster_se':None,'descriptive_normal95':None}
    residual=y-p; mean=float(residual.mean())
    _,inv=np.unique(np.asarray(groups).astype(str),return_inverse=True)
    count=int(inv.max()+1)
    scores=np.bincount(inv,weights=residual-mean)
    se=float(np.sqrt(count/(count-1)*np.dot(scores,scores))/n) if count>1 else None
    return {'rows':n,'groups':count,'target_rate':float(y.mean()),'mean_probability':float(p.mean()),
            'predicted_positive_rows_at_half':int((p>=.5).sum()),
            'errors_at_fixed_half':int(((p>=.5)!=y).sum()),
            'mean_residual_y_minus_p':mean,'cluster_se':se,
            'descriptive_normal95':None if se is None else [mean-1.96*se,mean+1.96*se]}


def contrast_summary(y,p,groups,a,b):
    assert not np.any(a&b)
    r=y-p; na,nb=int(a.sum()),int(b.sum())
    if not na or not nb:raise ValueError('Fixed contrast has an empty side')
    actual=float(y[a].mean()-y[b].mean()); predicted=float(p[a].mean()-p[b].mean())
    delta=actual-predicted
    keep=a|b; _,inv=np.unique(groups[keep],return_inverse=True); ng=int(inv.max()+1)
    influence=np.zeros(len(y)); influence[a]=(r[a]-r[a].mean())/na; influence[b]=-(r[b]-r[b].mean())/nb
    scores=np.bincount(inv,weights=influence[keep]); se=float(np.sqrt(ng/(ng-1)*np.dot(scores,scores))) if ng>1 else None
    return {'third_bin_rows':na,'other_three_bins_rows':nb,'actual_rate_difference':actual,
            'predicted_probability_difference':predicted,'unexplained_rate_difference':delta,
            'cluster_se':se,'descriptive_normal95':None if se is None else [delta-1.96*se,delta+1.96*se]}


def load_verified():
    record=group.assert_phase()  # Source/history/protocol/policy/auth identity, not a fit.
    summaries=group.complete_queue(record)
    matrices,dev,y,groups,passengers=audit.load_audit_data(iterate)
    if group.data_identity(matrices,dev,y,groups,passengers)!=record['data']:
        raise RuntimeError('Historical data identity changed')
    eligibility_path=group.PHASE/'eligibility.json'
    if group.digest(eligibility_path)!=(group.PHASE/'eligibility.sha256').read_text().strip():
        raise RuntimeError('Frozen eligibility hash changed')
    eligibility=json.loads(eligibility_path.read_text())
    if (eligibility['status']!='complete' or not eligibility['fixed_queue_complete']
            or eligibility['phase_sha256']!=group.digest(group.PHASE/'protocol.json')
            or eligibility['policy_sha256']!=group.digest(ROOT/group.POLICY)
            or eligibility['authorization_sha256']!=group.digest(ROOT/group.AUTH)):
        raise RuntimeError('Phase decision/policy/auth identity changed')
    predictions={}; evidence={}
    for reference in ['mitra','v44']:
        summary=summaries[('refs',reference)]
        path=group.PHASE/'refs'/reference/'summary.json'
        if eligibility['component_summary_sha256']['refs/'+reference]!=group.digest(path):
            raise RuntimeError('Reference summary differs from frozen decision')
        for row in summary['results']:
            seed=row['seed']; assignment=group.fold_assignment(record,seed,dev,groups,passengers)
            p=group.artifact_probability(ROOT/row['oof_path'],row['oof_sha256'],dev,groups,passengers,assignment)
            if abs(float(accuracy_score(y,p>=.5))-row['accuracy'])>1e-12 or abs(float(log_loss(y,p))-row['log_loss'])>1e-12:
                raise RuntimeError('Reference metrics do not reproduce')
            if not row['legacy_reused']:
                for fold,(a,b) in enumerate(group.checked_partition(assignment,groups)):
                    item=row['folds'][fold]
                    if (item['fit_seed']!=seed+fold or item['group_overlap']!=0
                            or item['train_ids_sha256']!=audit.row_hash(dev[a])
                            or item['query_ids_sha256']!=audit.row_hash(dev[b])):
                        raise RuntimeError('Reference fit boundary mismatch')
            predictions[(reference,seed)]=p
            evidence[f'{reference}/{seed}']={'oof_path':row['oof_path'],'oof_sha256':row['oof_sha256'],
                'fold_assignment_sha256':record['fold_hashes'][str(seed)],'summary_sha256':group.digest(path),
                'execution_protocol_sha256':summary['execution_protocol_sha256'],'legacy_reused':row['legacy_reused']}
    return record,dev,y,groups,passengers,predictions,evidence


def main():
    quality=json.loads(QUALITY.read_text())
    if quality['source_sha256']['data_quality_diagnostic.py']!=group.digest(ROOT/'data_quality_diagnostic.py'):
        raise RuntimeError('Fixed diagnostic bin source changed')
    record,dev_ids,y,groups,passengers,predictions,evidence=load_verified()
    train,test,y_again=audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev_ids)
    assert np.array_equal(y,y_again)
    roster=attributes(pd.concat([train,test],ignore_index=True))
    rank=roster.groupby(['Deck','Side'],dropna=False).CabinNumber.rank(method='average',pct=True)
    bins=np.minimum(np.floor(rank.fillna(0).to_numpy()*4).astype(int),3)
    bins[rank.isna().to_numpy()]=-1
    dev=roster.iloc[dev_ids].reset_index(drop=True); dev['SpatialQuartile']=bins[dev_ids]
    earth_g=dev.HomePlanet.eq('Earth') & dev.Deck.eq('G')
    # No rows/cutpoints selected based on residuals. All fixed Side/Cryo/quartile cells retained.
    fixed_cells=[]
    for (side,cryo,quartile),part in dev.loc[earth_g].groupby(['Side','CryoSleep','SpatialQuartile'],dropna=False,sort=True):
        idx=part.index.to_numpy(); key={'HomePlanet':'Earth','Deck':'G','Side':None if pd.isna(side) else str(side),
                                    'CryoSleep':None if pd.isna(cryo) else str(cryo),'SpatialQuartile':str(quartile)}
        # Assert exact reproduction of the previously reported target strata.
        old=[v for v in quality['confounding_and_space']['spatial_label_tables'] if all(v[k]==value for k,value in key.items())]
        if len(old)!=1 or old[0]['rows']!=len(idx) or old[0]['positive_labels']!=int(y[idx].sum()):
            raise RuntimeError('Spatial bins or target counts differ from original diagnostic')
        results={}
        for reference in ['mitra','v44']:
            per_seed=[{'seed':seed,**cluster_summary(y[idx],predictions[(reference,seed)][idx],groups[idx])} for seed in SEEDS]
            values=[row['mean_residual_y_minus_p'] for row in per_seed]
            results[reference]={'per_seed':per_seed,'residual_direction_by_seed':[int(np.sign(v)) for v in values],
                                 'all_three_residuals_same_nonzero_direction':all(v>0 for v in values) or all(v<0 for v in values),
                                 'mean_across_reused_partitions':float(np.mean(values))}
        fixed_cells.append({'slice':key,'models':results})
    contrasts=[]
    # Requested G/S third bin vs other three, separately for the two observed sleep states.
    for cryo in [False,True]:
        eligible=(earth_g & dev.Side.eq('S') & dev.CryoSleep.eq(cryo) & dev.SpatialQuartile.ge(0)).to_numpy()
        a=eligible & dev.SpatialQuartile.eq(2).to_numpy(); b=eligible & dev.SpatialQuartile.ne(2).to_numpy()
        contrasts.append({'slice':{'HomePlanet':'Earth','Deck':'G','Side':'S','CryoSleep':cryo},
                          'comparison':'zero-based rank bin2 (third of four) minus bins0,1,3',
                          'models':{ref:[{'seed':seed,**contrast_summary(y,predictions[(ref,seed)],groups,a,b)} for seed in SEEDS] for ref in ['mitra','v44']}})
    report={'status':'descriptive_frozen_group_oof_residual_audit_no_new_model',
      'label_policy':'Only original6954dev labels through audit.read_development_only; no unused1739 targets or test targets.',
      'fixed_bins_source':'reports/data_error_diagnosis_20261117/data_quality.json',
      'fixed_bin_definition':quality['confounding_and_space']['spatial_rank_definition'],
      'models':group.REFERENCES,'group_split_seeds':SEEDS,'development_identity':record['data'],
      'verification':'Phase policy/auth/source/history, complete summaries bound to eligibility SHA, OOF SHA, passenger/group/dev IDs, frozen fold assignment, group separation, new-fit seed/train/query hashes, accuracy/logloss verified before residual calculations.',
      'reference_source_evidence':evidence,'Earth_G_cells':fixed_cells,'requested_G_S_third_bin_contrasts':contrasts,
      'uncertainty':'Analytic finite-group corrected cluster SE of fixed OOF residual mean; clusters=GroupId. Normal95 intervals are descriptive, unadjusted for many cells and adaptive examination. They do not include refitting uncertainty/dependence between models trained on overlapping folds. Three partitions reuse identical labels and are not three independent samples.',
      'limitations':['This is not a new candidate score or independent holdout test.','High observed target-rate differences alone do not demonstrate unused predictive signal.','No calibration offset, smoother, thresholds, bin search or model fitted.','Residual means may change confidence without changing threshold0.5 decisions; do not imply attainable accuracy gains.'],
      'source_sha256':{p:group.digest(ROOT/p) for p in ['spatial_residual_diagnostic.py','data_quality_diagnostic.py','audit_top5_references.py','top5_group_evaluate.py','iterate.py',group.POLICY,group.AUTH,'reports/data_error_diagnosis_20261117/data_quality.json','artifacts/top5_group_phase/protocol.json','artifacts/top5_group_phase/eligibility.json']},
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
      'decision':{
        'headline':'Conditional support for one separately frozen Earth/G/awake spatial-calibration experiment if strict nested base probabilities can be reused at low cost; no direct correction from these diagnostic OOF arrays. The sleeping third-bin pattern does not justify another spatial model.',
        'sleeping_third_bin':'Observed rate125/126=.9921; raw Mitra mean probability .9641/.9711/.9683, versus v44 .8712/.8736/.8753. Mitra third-minus-other-bin unexplained differences .0431/.0337/.0387 have all three descriptive group-cluster95 intervals crossing zero. Do not motivate another model by the raw .992 rate alone.',
        'awake_pattern':'Observed third-bin83/112=.7411 vs Mitra .6708/.6742/.6736; the preceding bin is observed .2835 vs Mitra .3704/.3722/.3773. Third-minus-other-bin residual contrast stays about+.10 in all three partitions; this is more specific evidence of residual spatial compression than the sleeping high-rate region.',
        'classification_counterevidence':'Both models already predict all126 sleeping third-bin rows positive at0.5 in all three splits and each makes one error. Raising their probabilities cannot fix that error. In the awake third bin, Mitra predicts82/84/83 positives vs83 actual positives, with21/19/18 errors: average probability underestimation alone does not identify which individuals should flip.',
        'caution':'The bins/focal region arose from exploratory reused labels; repeated partitions share all targets, model-training dependence is omitted by the simple group SE, and many cells were viewed. These intervals are not independent confirmation or a multiple-testing-corrected significance claim.',
        'if_root_pursues':'Freeze eligibility Earth & DeckG & observed CryoSleep=False, both Sides and all four existing rank bins before any candidate score, with no third-bin-only override or cutpoint search. Train the calibration from strictly nested outertrain-only base OOF probabilities; compare with a matched self-only calibration control. Specify shrinkage/objective once, not against these residuals. Existing sleeping-only spatial expert is not an equivalent awake test.',
        'nested_cache_reuse_boundary':'Root reports a separately planned raw-Mitra/Kumo nested gate may supply raw-Mitra innerOOF without additional GPU. Reuse only if outertrain/query IDs, fold construction, base recipe/preprocessing and fit seeds match the spatial experiment exactly and artifact/source hashes verify. Random-outer gate innerOOF is not reusable as group-outer meta training merely because both contain the same6954development rows. Cache availability is not asserted here.',
        'accuracy_limit':'Calibration residual alone need not change0.5 decisions; the reported focal classification counts expose this distinction. No claim of attainable extra correct predictions, superiority to v81/v93, or irreducible accuracy ceiling.',
        'authorization':'This report does not authorize fitting, scoring a correction, or uploading.'}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'sha256':group.digest(OUT),'focal_contrasts':contrasts},indent=2))


if __name__=='__main__':main()
