"""A smooth cabin-position expert for the development-identified Earth/CryoSleep slice."""
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

ORIGINAL_FIT=iterate.fit
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v23_spatial5':dict(context=False,model='spatial',knots=5,C=1.,weight=1.),
    'v24_spatial9':dict(context=False,model='spatial',knots=9,C=1.,weight=1.),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
}
iterate.BLEND=[]


class SpatialPredictor:
    def __init__(self,base_model,spline,side_encoder,expert,weight):
        self.base_model=base_model;self.spline=spline;self.side_encoder=side_encoder
        self.expert=expert;self.weight=weight

    @staticmethod
    def eligible(x):
        return x.HomePlanet.eq('Earth') & x.CryoSleep.eq('True') & x.Side.isin(['P','S']) & x.CabinNumber.notna()

    def design(self,x):
        smooth=self.spline.transform(x[['CabinNumber']])
        side=self.side_encoder.transform(x[['Side']])
        return np.concatenate([side,(smooth[:,:,None]*side[:,None,:]).reshape(len(x),-1)],axis=1)

    def predict_proba(self,x):
        p=self.base_model.predict_proba(x)[:,1]
        mask=self.eligible(x)
        if mask.any():
            specialist=self.expert.predict_proba(self.design(x.loc[mask]))[:,1]
            p[mask]=(1-self.weight)*p[mask]+self.weight*specialist
        return np.column_stack([1-p,p])


def fit_spatial(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return ORIGINAL_FIT(name,matrices,train_idx,y,predict_idx,seed)
    config=iterate.CONFIGS[name]
    _,base_model=ORIGINAL_FIT(iterate.BASELINE,matrices,train_idx,y,predict_idx,seed)
    x,_=matrices[False]
    training=x.iloc[train_idx]
    mask=SpatialPredictor.eligible(training)
    specialist=training.loc[mask]
    target=np.asarray(y)[mask.to_numpy()]
    spline=make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=config['knots'],degree=3,include_bias=False))
    side=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    spline.fit(specialist[['CabinNumber']]);side.fit(specialist[['Side']])
    expert=LogisticRegression(C=config['C'],max_iter=1500,random_state=seed)
    model=SpatialPredictor(base_model,spline,side,expert,config['weight'])
    expert.fit(model.design(specialist),target)
    return model.predict_proba(x.iloc[predict_idx])[:,1],model


iterate.fit=fit_spatial

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_006');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        path=iterate.ROOT/'spatial_expert.py';shutil.copy2(path,out/'spatial_expert_source.py')
        protocol=json.loads((out/'protocol.json').read_text())
        protocol['sources']['spatial_expert.py']=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['expert_policy']='Earth+CryoSleep True+known cabin/side only; all spline knots and expert fitting use outer training rows. This slice was identified using prior development OOF errors, not independent evidence.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
