"""Isolate depth on the no-ID baseline; reuse the frozen spatial expert."""
import argparse
import hashlib
import json
import shutil
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import iterate
import spatial_expert

BASE_FIT=spatial_expert.ORIGINAL_FIT
DEEP={**iterate.CAT,'depth':8}
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v31_depth8':dict(context=False,model='catboost',**DEEP),
    'v32_depth8_spatial':dict(context=False,model='spatial_deep',knots=5,C=.1,weight=.5),
}
iterate.BLEND=[]


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name in [iterate.BASELINE,'v31_depth8']:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name=='v25_spatial_blend':
        return spatial_expert.fit_spatial(name,matrices,train_idx,y,predict_idx,seed)
    config=iterate.CONFIGS[name]
    _,base=BASE_FIT('v31_depth8',matrices,train_idx,y,predict_idx,seed)
    x,_=matrices[False];training=x.iloc[train_idx]
    mask=spatial_expert.SpatialPredictor.eligible(training);specialist=training.loc[mask]
    spline=make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=5,degree=3,include_bias=False))
    side=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    spline.fit(specialist[['CabinNumber']]);side.fit(specialist[['Side']])
    expert=LogisticRegression(C=.1,max_iter=1500,random_state=seed)
    model=spatial_expert.SpatialPredictor(base,spline,side,expert,.5)
    expert.fit(model.design(specialist),np.asarray(y)[mask.to_numpy()])
    return model.predict_proba(x.iloc[predict_idx])[:,1],model


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_011');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['depth_ablation.py','spatial_expert.py']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['depth_policy']='Only CatBoost depth changes 6 to 8. The optional spatial route, knots, regularization and blend weight remain exactly v25.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
