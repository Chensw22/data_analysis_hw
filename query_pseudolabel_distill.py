"""Fixed transductive query-pseudolabel CatBoost CE student; no query targets in fit."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
import iterate
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/query_pseudolabel_distill'
PARAMS=dict(iterations=700,depth=6,learning_rate=.045,l2_leaf_reg=5,loss_function='CrossEntropy',
    thread_count=4,verbose=False,allow_writing_files=False)


def fit_student(train_x,train_y,unlabeled_x,teacher_p,seed,use_pseudo=True):
    """Public reusable API: no argument can carry unlabeled/query true targets."""
    y=np.asarray(train_y,dtype=float);p=np.asarray(teacher_p,dtype=float)
    if y.shape!=(len(train_x),) or p.shape!=(len(unlabeled_x),) or not np.isin(y,[0,1]).all():raise ValueError('Invalid training labels/shapes')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid teacher probabilities')
    if not train_x.index.is_unique or not unlabeled_x.index.is_unique or len(train_x.index.intersection(unlabeled_x.index)):raise ValueError('Train/query ID overlap')
    if list(train_x.columns)!=list(unlabeled_x.columns) or any(c in train_x for c in ['Name','Transported','PassengerId','GroupId']):raise ValueError('Invalid feature boundary')
    chosen=(p>=.9)|(p<=.1)
    if use_pseudo:
        fit_x=pd.concat([train_x,unlabeled_x.iloc[np.flatnonzero(chosen)]],axis=0)
        targets=np.r_[y,p[chosen]];weight=np.r_[np.ones(len(y)),np.full(chosen.sum(),.35)]
    else:fit_x=train_x;targets=y;weight=np.ones(len(y))
    cats=fit_x.select_dtypes(include=['object','str','bool']).columns.tolist()
    model=CatBoostClassifier(**PARAMS,random_seed=seed,cat_features=cats)
    model.fit(fit_x,targets,sample_weight=weight)
    return model,dict(hard_training_rows=len(y),unlabeled_query_rows=len(p),pseudo_selected_rows=int(chosen.sum()) if use_pseudo else 0,
        available_high_confidence=int(chosen.sum()),pseudo_positive=int(np.sum(p[chosen]>=.9)) if use_pseudo else 0,
        pseudo_negative=int(np.sum(p[chosen]<=.1)) if use_pseudo else 0,pseudo_weight=.35 if use_pseudo else None,
        fit_rows=len(targets),hard_weight_mass=len(y),pseudo_weight_mass=float(weight[len(y):].sum()),
        training_ids_sha256=r.nested.ids_hash(train_x.index.to_numpy()),query_ids_sha256=r.nested.ids_hash(unlabeled_x.index.to_numpy()),
        pseudo_ids_sha256=r.nested.ids_hash(unlabeled_x.index.to_numpy()[chosen]) if use_pseudo else None)


def run():
    if REPORT.exists():raise FileExistsError('One fixed experiment, no overwrite')
    REPORT.mkdir(parents=True);start=time.monotonic()
    paths=['query_pseudolabel_distill.py','iterate.py','experiment.py','error_overlap_diagnostic.py','relational_error_corrector.py',
        'exaone_validation_data.py','reports/v96_oof_completion/result.json','artifacts/v96_oof_completion/result.json','artifacts/v96_oof_completion/oof.npz',
        'artifacts/error_rescue_20261117/catalog_rows.csv']
    sources={p:r.sha(ROOT/p) for p in paths}
    plan=dict(status='fixed_before_fit',params=PARAMS,seed='20261112+outerfold',folds=3,dev_rows=6954,
        teacher='Authenticated original rawMitra outerfold OOF: teacher trained only thisfolda, prediction thisfoldb. Never full8693teacher predictions of dev.',
        pseudo_rule='Select outerquery p<=.1 or>=.9, use originalsoftp withsampleweight.35; all trueoutertrain labelsweight1. Querytruey neverpassedtofit.',
        controls='Matched sameCrossEntropyCat depth6/700/.045/L2=5, onlyhardtrain; both independently mix0.5 withmatchedv96OOF. No weight/confidence sweep.',
        features='30engineered_no_ids rowlocal, rawNameexcluded beforeMissingCount; train.csv attributesonly foractualstudentX; officialtest neverentersstudent/teacherpseudo/trainingpool.',
        evaluation='Transductive query-specific adaptation to the exact scoredunlabeled fold; not ordinary inductiveCV. Original546 reportingonlyafterallpredictions.',
        budget_seconds=600,sources_sha256=sources,
        limitations=['Teachererrors may be reinforced at highconfidence.','Futuredeployment mustrepeatteacher/studentprocedure onfulltrain+actualtestquery; no guaranteefromsame-dev adaptiveexperiment.','Evaluation legitimately usesqueryX/teacherp, not queryy; not comparabletofullyinductive dataaccess withoutdisclosure.'])
    r.write(REPORT/'protocol.json',plan)
    handler=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU600s')));signal.setitimer(signal.ITIMER_REAL,600)
    try:
        data,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y'];teacher=bases['rawMitra']
        attrs=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c not in ['Transported','Name'])
        x,cats=iterate.build_features(attrs,False);assert x.shape==(8693,30)
        np.testing.assert_array_equal(attrs.PassengerId.iloc[dev].to_numpy(),data['passengers'])
        hist=json.loads((ROOT/'reports/v96_oof_completion/result.json').read_text())
        if hist!=json.loads((ROOT/'artifacts/v96_oof_completion/result.json').read_text()) or hist['status']!='complete_v96_fixed15_original3fold_oof':raise RuntimeError('v96 incomplete')
        path=ROOT/'artifacts/v96_oof_completion/oof.npz'
        if r.sha(path)!=hist['oof_sha256']:raise RuntimeError('v96 SHA mismatch')
        with np.load(path,allow_pickle=False) as z:
            for k,w in [('dev_ids',dev),('fold_id',folds),('passenger_ids',data['passengers'])]:np.testing.assert_array_equal(z[k],w)
            v96=z['prob'].copy()
        evidence.update(sources);r.write(REPORT/'evidence.json',evidence)
        pred={k:np.full(len(y),np.nan) for k in ['hard_control','pseudo_student','hard_half_v96','pseudo_half_v96']};pred['raw_teacher']=teacher;pred['v96']=v96;records=[]
        with threadpool_limits(limits=4):
            for fold,(a,b) in enumerate(splits):
                record=dict(fold=fold,models={});t=time.monotonic()
                for key,use in [('hard_control',False),('pseudo_student',True)]:
                    model,metadata=fit_student(x.iloc[dev[a]],y[a],x.iloc[dev[b]],teacher[b],20261112+fold,use_pseudo=use)
                    p=model.predict_proba(x.iloc[dev[b]])[:,1]
                    pred[key][b]=p;mixture='pseudo_half_v96' if use else 'hard_half_v96';pred[mixture][b]=.5*p+.5*v96[b]
                    path=ROOT/f'artifacts/query_pseudolabel_fold{fold}_{key}.cbm';model.save_model(str(path))
                    restored=CatBoostClassifier();restored.load_model(str(path));np.testing.assert_array_equal(p,restored.predict_proba(x.iloc[dev[b]])[:,1])
                    record['models'][key]=dict(**metadata,model_sha256=r.sha(path),reload_max_diff=0.,metrics=r.score(p,y[b]))
                    del model,restored
                record['seconds']=time.monotonic()-t;records.append(record);r.write(REPORT/f'fold{fold}.json',record)
                print(json.dumps(dict(fold=fold,models=record['models'])),flush=True)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for k,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[k],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        masks=dict(all=np.ones(len(y),bool),hard546=hard,hard914=rows.family_wrong_count.to_numpy()==5,commoncorrect5263=rows.family_wrong_count.to_numpy()==0)
        def stat(m):return {k:dict(correct=int((((p>=.5)==y)&m).sum()),vs_raw=r.pair(p,teacher,y,m),vs_hard=r.pair(p,pred['hard_control'],y,m),vs_hard_half=r.pair(p,pred['hard_half_v96'],y,m),vs_v96=r.pair(p,v96,y,m)) for k,p in pred.items()}
        for i,record in enumerate(records):record['paired']=stat(folds==i);record['hard546']=stat((folds==i)&hard);r.write(REPORT/f'fold{i}.json',record)
        cache=ROOT/'artifacts/query_pseudolabel_oof.npz';np.savez_compressed(cache,**pred,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str))
        for p,digest in evidence.items():
            if r.sha(ROOT/p)!=digest:raise RuntimeError('Sourcechanged '+p)
        result=dict(status='complete',metrics={k:r.score(p,y) for k,p in pred.items()},subsets={k:stat(m) for k,m in masks.items()},folds=records,
            seconds=time.monotonic()-start,protocol_sha256=r.sha(REPORT/'protocol.json'),oof_sha256=r.sha(cache),no_upload=True,old_holdout_labels_read=False)
        r.write(REPORT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
if __name__=='__main__':run()
