"""Cross-fitted low-dimensional spending logits as oblique tree features."""
import argparse
import hashlib
import json
import shutil
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import iterate
import spatial_expert

BASE_FIT=spatial_expert.ORIGINAL_FIT
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v39_spending_projection':dict(context=False,model='projection',inner_folds=3,C=1.),
    'v40_projection_spatial':dict(context=False,model='projection_spatial',inner_folds=3,C=1.),
}
iterate.BLEND=[]
NAMES=['RawSpendingLogit','LogSpendingLogit']
CACHE={}


def projection(log,seed):
    steps=([FunctionTransformer(np.log1p)] if log else [])
    return make_pipeline(*steps,SimpleImputer(strategy='median'),StandardScaler(),
                         LogisticRegression(C=1.,max_iter=2000,random_state=seed))


def crossfit(x,train_idx,y,predict_idx,seed):
    assert not set(train_idx)&set(predict_idx)
    spending=x[iterate.base.SPEND];y=np.asarray(y);train_idx=np.asarray(train_idx)
    train_features=np.full((len(train_idx),2),np.nan)
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+5000).split(train_idx,y)):
        assert not set(train_idx[a])&set(train_idx[b])
        for column in range(2):
            model=projection(column==1,seed+fold)
            model.fit(spending.iloc[train_idx[a]],y[a])
            train_features[b,column]=model.decision_function(spending.iloc[train_idx[b]])
    models=[];predict_features=[]
    for column in range(2):
        model=projection(column==1,seed)
        model.fit(spending.iloc[train_idx],y)
        predict_features.append(model.decision_function(spending.iloc[predict_idx]))
        models.append(model)
    assert np.isfinite(train_features).all()
    return train_features,np.column_stack(predict_features),models


class ProjectionPredictor:
    def __init__(self,models,model):
        self.projections=models;self.model=model

    def predict_proba(self,x):
        x=x.copy()
        for name,model in zip(NAMES,self.projections):
            x[name]=model.decision_function(x[iterate.base.SPEND])
        return self.model.predict_proba(x)


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name=='v25_spatial_blend':
        return spatial_expert.fit_spatial(name,matrices,train_idx,y,predict_idx,seed)
    x,cats=matrices[False]
    key=(seed,hashlib.sha256(np.asarray(train_idx).tobytes()+np.asarray(predict_idx).tobytes()).hexdigest())
    if key not in CACHE:
        CACHE[key]=crossfit(x,train_idx,y,predict_idx,seed)
    train_features,predict_features,models=CACHE[key]
    augmented=x.copy()
    for column,label in enumerate(NAMES):
        augmented[label]=np.nan
        augmented.iloc[train_idx,augmented.columns.get_loc(label)]=train_features[:,column]
        augmented.iloc[predict_idx,augmented.columns.get_loc(label)]=predict_features[:,column]
    changed={False:(augmented,cats)}
    if name=='v39_spending_projection':
        p,model=BASE_FIT(iterate.BASELINE,changed,train_idx,y,predict_idx,seed)
    else:
        p,model=spatial_expert.fit_spatial('v25_spatial_blend',changed,train_idx,y,predict_idx,seed)
    return p,ProjectionPredictor(models,model)


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_015');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['spending_projection.py','spatial_expert.py']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['projection_policy']='Only five spending columns; raw and log1p, median+StandardScaler+L2 LR C1. Training projection features generated via inner3fold OOF. Validation projections from full outertrain fit. Original CatBoost/spatial settings unchanged.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
