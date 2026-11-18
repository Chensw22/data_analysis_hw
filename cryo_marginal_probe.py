"""Fixed query-only marginalization of missing CryoSleep, matched Cat backbone."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import time
import signal
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
import iterate
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/cryo_marginal_probe'
TARGET=dict(iterations=700,depth=6,learning_rate=.045,l2_leaf_reg=5)
ATTRIBUTE=dict(iterations=200,depth=4,learning_rate=.06,l2_leaf_reg=5)
QCOLS=['HomePlanet','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']

def posterior_fit(x,seed):
    # Transported never passed to attribute predictor.
    known=x.CryoSleep.isin(['True','False']).to_numpy();q=x[QCOLS]
    cats=q.select_dtypes(include=['object','str','bool']).columns.tolist()
    model=CatBoostClassifier(**ATTRIBUTE,loss_function='Logloss',cat_features=cats,random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False)
    label=x.CryoSleep.eq('True').to_numpy(int)
    if set(label[known])!={0,1}:raise ValueError('Two observed states required')
    model.fit(q.loc[known],label[known]);return model,known

def marginal(model,posterior,x):
    original=model.predict_proba(x)[:,1];p=original.copy();missing=~x.CryoSleep.isin(['True','False']).to_numpy()
    q=posterior.predict_proba(x[QCOLS])[:,1]
    if missing.any():
        branch=x.loc[missing].copy();branch['CryoSleep']='False';awake=model.predict_proba(branch)[:,1]
        branch['CryoSleep']='True';sleep=model.predict_proba(branch)[:,1]
        p[missing]=(1-q[missing])*awake+q[missing]*sleep
    np.testing.assert_array_equal(p[~missing],original[~missing])
    return original,p,q,missing

def run():
    if OUT.exists():raise FileExistsError('No overwrite or parameter retry')
    OUT.mkdir(parents=True);start=time.monotonic()
    names=['cryo_marginal_probe.py','iterate.py','experiment.py','error_overlap_diagnostic.py','relational_error_corrector.py','exaone_validation_data.py','planet_marginal.py','reports/local_iterations.csv','artifacts/error_rescue_20261117/catalog_rows.csv']
    sources={n:r.sha(ROOT/n) for n in names}
    plan=dict(status='fixed_before_fit',target=TARGET,attribute_model=ATTRIBUTE,seed='20261112+fold',threshold=.5,folds=3,devrows=6954,
        features='Original engineered_no_ids30 excluding Name before construction, including MissingCount without Name; same target model for both predictions.',
        posterior_features=QCOLS,posterior_training='Only observedCryo outertrain, binaryCryo as label; noTransported input/labels.',
        rule='Only missingCryo outerquery: qTrue*target(Cryo=True)+(1-qTrue)*target(Cryo=False). Keep original MissingCount, all other columns and full observed-query probabilities unchanged.',
        control='Same fitted target model predicting original missingCryo input, no altered target training/sample weights.',
        prior='HomePlanet posterior-weighted training expansion v58 already tested (run029), not claimed newfamily. This fixes different missing attribute and query-only same-model integration.',
        fixed_budget_seconds=300,no_gpu=True,no_new_base_inference=True,
        limitations=['Conditional distribution of missingCryo assumed transportable from observed outertrain; MNAR can invalidate it.','This is fitted predictive marginalization, not exact Bayesian inference or causal intervention.','Missingness retained via original MissingCount; two categorical branches may be off observed joint support.','Small missing subset and adapted same dev labels; no independent generalization claim.'],sources_sha256=sources)
    r.write(OUT/'protocol.json',plan)
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU300s')));signal.setitimer(signal.ITIMER_REAL,300)
    try:
        inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y']
        attrs=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=lambda c:c not in ('Transported','Name')) for f in ['train.csv','test.csv']],ignore_index=True)
        x,cats=iterate.build_features(attrs,False);assert x.shape==(12970,30)
        np.testing.assert_array_equal(attrs.PassengerId.iloc[dev].to_numpy(),inputs['passengers'])
        for f in ['train.csv','test.csv']:evidence['data/raw/'+f]=r.sha(ROOT/'data/raw'/f)
        evidence.update(sources);r.write(OUT/'evidence.json',evidence)
        pred={k:np.full(len(y),np.nan) for k in ['original','marginal']};qall=np.full(len(y),np.nan);missing_all=np.zeros(len(y),bool);records=[]
        with threadpool_limits(limits=4):
            for fold,(a,b) in enumerate(splits):
                train=x.iloc[dev[a]];query=x.iloc[dev[b]];seed=20261112+fold;t=time.monotonic()
                posterior,known=posterior_fit(train,seed)
                model=CatBoostClassifier(**TARGET,cat_features=cats,random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,loss_function='Logloss')
                model.fit(train,y[a]);p,m,q,missing=marginal(model,posterior,query)
                pred['original'][b]=p;pred['marginal'][b]=m;qall[b]=q;missing_all[b]=missing
                observed=~missing;true=query.CryoSleep.eq('True').to_numpy(int)
                record=dict(fold=fold,train_ids_sha256=r.nested.ids_hash(dev[a]),query_ids_sha256=r.nested.ids_hash(dev[b]),
                    posterior_training_rows=int(known.sum()),query_missing_rows=int(missing.sum()),query_observed_rows=int(observed.sum()),
                    observed_attribute_accuracy=float(np.mean((q[observed]>=.5)==true[observed])),observed_attribute_brier=float(np.mean((q[observed]-true[observed])**2)),
                    observed_attribute_log_loss=float(log_loss(true[observed],q[observed])),
                    metrics={k:r.score(v[b],y[b]) for k,v in pred.items()},paired=r.pair(m,p,y[b]),seconds=time.monotonic()-t)
                records.append(record);r.write(OUT/f'fold{fold}.json',record)
                model.save_model(str(ROOT/f'artifacts/cryo_marginal_fold{fold}_target.cbm'))
                posterior.save_model(str(ROOT/f'artifacts/cryo_marginal_fold{fold}_posterior.cbm'))
                restored=CatBoostClassifier();restored.load_model(str(ROOT/f'artifacts/cryo_marginal_fold{fold}_target.cbm'))
                restored_q=CatBoostClassifier();restored_q.load_model(str(ROOT/f'artifacts/cryo_marginal_fold{fold}_posterior.cbm'))
                np.testing.assert_array_equal(m,marginal(restored,restored_q,query)[1]);record['reload_max_diff']=0.
                del model,posterior,restored,restored_q
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for k,want in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[k],want)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        def summary(mask):return dict(rows=int(mask.sum()),original_correct=int((((pred['original']>=.5)==y)&mask).sum()),marginal_correct=int((((pred['marginal']>=.5)==y)&mask).sum()),paired=r.pair(pred['marginal'],pred['original'],y,mask))
        for i,rec in enumerate(records):rec['hard546']=summary((folds==i)&hard);r.write(OUT/f'fold{i}.json',rec)
        np.testing.assert_array_equal(pred['marginal'][~missing_all],pred['original'][~missing_all])
        path=ROOT/'artifacts/cryo_marginal_oof.npz'
        np.savez_compressed(path,**pred,posterior=qall,missing=missing_all,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        for name,digest in evidence.items():
            if r.sha(ROOT/name)!=digest:raise RuntimeError('Frozen evidence changed '+name)
        result=dict(status='complete',metrics={k:r.score(v,y) for k,v in pred.items()},all=summary(np.ones(len(y),bool)),missing=summary(missing_all),hard546=summary(hard),hard546_missing=summary(hard&missing_all),folds=records,
            seconds=time.monotonic()-start,protocol_sha256=r.sha(OUT/'protocol.json'),oof_sha256=r.sha(path),no_upload=True,old_holdout_labels_read=False)
        r.write(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:r.write(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
