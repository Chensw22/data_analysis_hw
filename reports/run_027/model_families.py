"""Diversify model families on the unchanged development protocol."""
import os
import sys
import importlib.util
from pathlib import Path


def ensure_openmp():
    # Some macOS LightGBM wheels need libomp. Reuse the installed sklearn runtime
    # in this process only; no system library or user shell settings are modified.
    if sys.platform=='darwin':
        spec=importlib.util.find_spec('sklearn')
        library=Path(spec.origin).parent/'.dylibs'
        if (library/'libomp.dylib').exists() and str(library) not in os.environ.get('DYLD_FALLBACK_LIBRARY_PATH','').split(':'):
            env={**os.environ,'DYLD_FALLBACK_LIBRARY_PATH':str(library)+':'+os.environ.get('DYLD_FALLBACK_LIBRARY_PATH','')}
            os.execve(sys.executable,[sys.executable,*sys.argv],env)


if __name__=='__main__':
    ensure_openmp()

import argparse
import hashlib
import json
import shutil
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
import iterate

ORIGINAL_FIT=iterate.fit
LGB=dict(n_estimators=700,learning_rate=.035,min_child_samples=30,reg_lambda=5,colsample_bytree=.9,
         subsample=.85,subsample_freq=1)
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v18_lgb15':dict(context=False,model='lightgbm',**LGB,num_leaves=15),
    'v19_lgb31':dict(context=False,model='lightgbm',**LGB,num_leaves=31),
    'v20_svm3':dict(context=False,model='svm',C=3,gamma='scale'),
    'v21_svm10':dict(context=False,model='svm',C=10,gamma='scale'),
    'v22_mlp':dict(context=False,model='mlp',hidden_layer_sizes=[64,32],alpha=1.,learning_rate_init=.001,
                   early_stopping=True,validation_fraction=.15,max_iter=400,n_iter_no_change=25),
}
iterate.BLEND=[]


def fit_model(name,matrices,train_idx,y,predict_idx,seed):
    config=iterate.CONFIGS[name]
    if config['model']=='catboost':
        return ORIGINAL_FIT(name,matrices,train_idx,y,predict_idx,seed)
    x,cats=matrices[False];x=x.copy()
    params={k:v for k,v in config.items() if k not in ['context','model']}
    if config['model']=='lightgbm':
        from lightgbm import LGBMClassifier
        for c in cats:x[c]=x[c].astype('category')
        model=LGBMClassifier(**params,random_state=seed,n_jobs=4,verbosity=-1,force_col_wise=True)
    else:
        # Avoid raw heavy-tailed duplicated amount columns in distance-based models.
        x=x.drop(columns=iterate.base.SPEND)
        x['TotalSpend']=np.log1p(x.TotalSpend)
        x['LuxurySpend']=np.log1p(x.LuxurySpend)
        nums=[c for c in x if c not in cats]
        preprocess=ColumnTransformer([
            ('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),nums),
            ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),
        ])
        if config['model']=='svm':
            estimator=SVC(**params,probability=True,random_state=seed,cache_size=512)
        else:
            params['hidden_layer_sizes']=tuple(params['hidden_layer_sizes'])
            estimator=MLPClassifier(**params,random_state=seed)
        model=make_pipeline(preprocess,estimator)
    with threadpool_limits(limits=4):
        model.fit(x.iloc[train_idx],y)
        prob=model.predict_proba(x.iloc[predict_idx])[:,1]
    return prob,model


iterate.fit=fit_model

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_005');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        path=iterate.ROOT/'model_families.py'
        shutil.copy2(path,out/'model_families_source.py')
        protocol=json.loads((out/'protocol.json').read_text())
        protocol['sources']['model_families.py']=hashlib.sha256(path.read_bytes()).hexdigest()
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
