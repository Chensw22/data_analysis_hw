"""Fixed two-round mixed-type forest imputation, target-free and outertrain-only."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import sys
import json
from pathlib import Path
import signal
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
import iterate
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/mixed_type_forest_imputation'
CAT=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
NUM=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','CabinNumber']
COLS=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
CATPARAMS=dict(iterations=700,depth=6,learning_rate=.045,l2_leaf_reg=5,loss_function='Logloss',thread_count=4,verbose=False,allow_writing_files=False)
FOREST=dict(n_estimators=50,max_depth=10,min_samples_leaf=2,n_jobs=4)
if __name__=='__main__':sys.modules['mixed_type_forest_imputation']=sys.modules[__name__]

def semantic(frame):
    x=frame[COLS].copy()
    for c in CAT:x[c]=x[c].astype(object).replace('__MISSING__',np.nan)
    for c in NUM:x[c]=pd.to_numeric(x[c],errors='raise')
    return x

class ForestChain:
    def fit_transform(self,original,seed):
        if list(original.columns)!=COLS:raise ValueError('Exact13attrs only; noName/ID/target')
        x=original.copy();missing=x.isna();self.fill={};self.steps=[];self.seed=seed
        for c in COLS:
            values=x[c].dropna()
            self.fill[c]=(str(values.mode().iloc[0]) if len(values) else '__MISSING__') if c in CAT else (float(values.median()) if len(values) else 0.)
            x[c]=x[c].fillna(self.fill[c])
        for round_index in range(2):
            for j,c in enumerate(COLS):
                observed=~missing[c].to_numpy()
                if not missing[c].any() or not observed.any():continue
                predictors=[v for v in COLS if v!=c];cc=[v for v in predictors if v in CAT];nn=[v for v in predictors if v in NUM]
                encoder=ColumnTransformer([('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cc),('numeric','passthrough',nn)])
                z=encoder.fit_transform(x[predictors]);ys=original.loc[observed,c]
                cls=RandomForestClassifier if c in CAT else RandomForestRegressor
                forest=cls(**FOREST,random_state=seed+round_index*100+j)
                forest.fit(z[observed],ys.astype(str).to_numpy() if c in CAT else ys.to_numpy(float))
                x.loc[~observed,c]=forest.predict(z[~observed])
                self.steps.append((c,predictors,encoder,forest))
        self.training_missing_counts={c:int(missing[c].sum()) for c in COLS}
        for c in COLS:np.testing.assert_array_equal(x.loc[~missing[c],c].to_numpy(),original.loc[~missing[c],c].to_numpy())
        return x
    def transform(self,original):
        if list(original.columns)!=COLS:raise ValueError('Exact13attrs')
        x=original.copy();missing=x.isna()
        for c in COLS:x[c]=x[c].fillna(self.fill[c])
        for c,predictors,encoder,forest in self.steps:
            m=missing[c].to_numpy()
            if m.any():x.loc[m,c]=forest.predict(encoder.transform(x.loc[m,predictors]))
        for c in COLS:np.testing.assert_array_equal(x.loc[~missing[c],c].to_numpy(),original.loc[~missing[c],c].to_numpy())
        return x
ForestChain.__module__='mixed_type_forest_imputation'

def repaired_features(original,filled):
    x=original.copy();x[COLS]=filled[COLS]
    for c in CAT:x[c]=x[c].fillna('__MISSING__').astype(str)
    for c in NUM:x[c]=x[c].astype(float)
    bills=x[SPEND]
    x['TotalSpend']=bills.sum(axis=1);x['AllSpendingZero']=bills.eq(0).all(axis=1).astype(int)
    x['PaidServices']=bills.gt(0).sum(axis=1);x['LuxurySpend']=x[['RoomService','Spa','VRDeck']].sum(axis=1)
    for c in SPEND:x[c+'_log1p']=np.log1p(x[c])
    flags=['MissingCount','SpendMissingCount','Age_missing']+[c+'_missing' for c in SPEND]
    pd.testing.assert_frame_equal(x[flags],original[flags],check_exact=True)
    if x[NUM].isna().any().any() or x[SPEND].lt(0).any().any():raise RuntimeError('Invalid reconstructed attrs')
    return x

class Predictor:
    """Reusable fitted full-data or outerfold pipeline; accepts original30 feature frame."""
    def __init__(self,model,chain=None):self.model=model;self.chain=chain
    def predict_proba(self,x):
        if self.chain is not None:x=repaired_features(x,self.chain.transform(semantic(x)))
        return self.model.predict_proba(x)
Predictor.__module__='mixed_type_forest_imputation'

def fit_pair(train_x,train_y,seed):
    # Only current training rows, observed feature values as forest targets.
    chain=ForestChain();filled=chain.fit_transform(semantic(train_x),seed);completed=repaired_features(train_x,filled)
    cats=train_x.select_dtypes(include=['object','str','bool']).columns.tolist();models={}
    for key,x in [('native_missing',train_x),('forest_chain',completed)]:
        m=CatBoostClassifier(**CATPARAMS,cat_features=cats,random_seed=seed);m.fit(x,train_y)
        models[key]=Predictor(m,chain if key=='forest_chain' else None)
    return models,chain

def check():
    # Feature-only deterministic checks, no real labels/optimizer or model fit.
    raw=pd.DataFrame({c:[1.,np.nan] for c in NUM})
    for c in CAT:raw[c]=['X',np.nan]
    s=semantic(raw[COLS]);assert s.shape==(2,13) and s.iloc[1].isna().all()
    original=pd.DataFrame({c:[0.,0.] for c in COLS})
    for c in CAT:original[c]=['__MISSING__','X']
    for c in ['MissingCount','SpendMissingCount','Age_missing']+[c+'_missing' for c in SPEND]:original[c]=[1,0]
    filled=original[COLS].copy();filled['HomePlanet']=['Earth','X'];f=repaired_features(original,filled)
    assert f.loc[0,'MissingCount']==1 and f.loc[0,'HomePlanet']=='Earth'

def run():
    check()
    if REPORT.exists():raise FileExistsError('Fixedchain no overwrite')
    REPORT.mkdir(parents=True);start=time.monotonic()
    names=['mixed_type_forest_imputation.py','iterate.py','experiment.py','error_overlap_diagnostic.py','relational_error_corrector.py','exaone_validation_data.py','artifacts/error_rescue_20261117/catalog_rows.csv']
    sources={n:r.sha(ROOT/n) for n in names}
    plan=dict(status='fixed_before_fit',rounds=2,forest=FOREST,target=CATPARAMS,seed='20261112+fold',columns=COLS,
        references=['https://github.com/jclopezsDS/kaggle-spaceship-titanic/blob/main/src/tools.py','https://www.kaggle.com/competitions/spaceship-titanic/data','https://scikit-learn.org/stable/modules/impute.html'],
        attribution='Inspiredbyauthor iterative_imputation; notreplicatingclaimed0.82020 or entirepipeline. Ours mixedclassification/regression,2round singlechain, nohandrisk/Name/targetfeatureselection.',
        method='Outertrain initialmedians/modes, fixedcolumnorder2rounds; perfield onehotothercategoricals andnumericotherattrs. Fit forest only observedfieldrows, update only originallymissing entries. Query replays storedroundsequence, neverfit query.',
        missing_policy='All original missing indicators preserved; raw13completed then deterministic spendingaggregates/logs recomputed coherently. No changes to observed13 values.',
        control='Same30engineered_no_ids/noName fixedCat withnativeoriginalmissing; samey/seed/hyperparameters. Complete rows separatelyreported.',
        no_test_inputs=True,no_old_holdout_labels=True,hard546_report_only=True,budget_seconds=600,
        limitations=['Single imputation, not Bayesian multipleimputation or uncertaintyintegration.','MAR-type attribute predictability assumption can fail.','Trainingmissingvalues and querymissingvalues follow standard inductivechain fit/transform, not same-row targetleakage.','Changed target training may alter evencompletequerypredictions.','Adaptive same6954dev; nopublicscoreguarantee.'],sources_sha256=sources)
    r.write(REPORT/'protocol.json',plan)
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU600s')));signal.setitimer(signal.ITIMER_REAL,600)
    try:
        data,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y']
        attrs=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c not in ['Name','Transported']);x,cats=iterate.build_features(attrs,False)
        np.testing.assert_array_equal(attrs.PassengerId.iloc[dev].to_numpy(),data['passengers']);assert x.shape==(8693,30)
        evidence.update(sources);r.write(REPORT/'evidence.json',evidence)
        pred={k:np.full(len(y),np.nan) for k in ['native_missing','forest_chain']};records=[]
        with threadpool_limits(limits=4):
            for fold,(a,b) in enumerate(splits):
                t=time.monotonic();models,chain=fit_pair(x.iloc[dev[a]],y[a],20261112+fold)
                for key,model in models.items():pred[key][b]=model.predict_proba(x.iloc[dev[b]])[:,1]
                rec=dict(fold=fold,training_rows=len(a),query_rows=len(b),attribute_forests=len(chain.steps),training_missing_counts=chain.training_missing_counts,
                    train_ids_sha256=r.nested.ids_hash(dev[a]),query_ids_sha256=r.nested.ids_hash(dev[b]),
                    metrics={k:r.score(p[b],y[b]) for k,p in pred.items()},paired=r.pair(pred['forest_chain'][b],pred['native_missing'][b],y[b]),seconds=time.monotonic()-t)
                records.append(rec);r.write(REPORT/f'fold{fold}.json',rec);print(json.dumps(rec),flush=True)
                del models,chain
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for k,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[k],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);complete=~semantic(x.iloc[dev]).isna().any(axis=1).to_numpy();assert hard.sum()==546
        masks=dict(all=np.ones(len(y),bool),hard546=hard,complete13=complete,any_missing13=~complete,hard546_complete=hard&complete,hard546_missing=hard&~complete)
        def summary(m):return dict(rows=int(m.sum()),correct={k:int((((p>=.5)==y)&m).sum()) for k,p in pred.items()},paired=r.pair(pred['forest_chain'],pred['native_missing'],y,m))
        for i,rec in enumerate(records):rec['complete13']=summary(complete&(folds==i));rec['hard546']=summary(hard&(folds==i));r.write(REPORT/f'fold{i}.json',rec)
        cache=ROOT/'artifacts/mixed_type_forest_imputation_oof.npz';np.savez_compressed(cache,**pred,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str))
        for p,digest in evidence.items():
            if r.sha(ROOT/p)!=digest:raise RuntimeError('Sourcechanged '+p)
        result=dict(status='complete',metrics={k:r.score(p,y) for k,p in pred.items()},subsets={k:summary(m) for k,m in masks.items()},folds=records,
            seconds=time.monotonic()-start,protocol_sha256=r.sha(REPORT/'protocol.json'),oof_sha256=r.sha(cache),no_upload=True,old_holdout_labels_read=False,
            reusable_entry='fit_pair(train_x,train_y,seed) -> native/forest Predictor; query predicts via frozen forest sequence. Full-data execution not performed.')
        r.write(REPORT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
