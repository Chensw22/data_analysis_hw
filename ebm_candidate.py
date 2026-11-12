"""Additive nonlinear effects plus training-only selected interactions."""
import argparse
import hashlib
import json
import shutil
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
import iterate
import spatial_expert

BASE_FIT=spatial_expert.ORIGINAL_FIT
PARAMS=dict(max_bins=256,max_interaction_bins=32,interactions=10,outer_bags=8,
    inner_bags=0,learning_rate=.015,max_rounds=10000,early_stopping_rounds=100,
    validation_size=.15,n_jobs=4,smoothing_rounds=75,interaction_smoothing_rounds=75)
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v29_ebm':dict(context=False,model='ebm',**PARAMS),
    'v30_ebm_spatial_blend':dict(context=False,model='ebm_spatial',weight=.5,**PARAMS),
}
iterate.BLEND=[]
COLUMNS=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt',
         'ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side','TotalSpend','PaidServices']
CACHE={}


class EBMPredictor:
    def __init__(self,model,spatial=None,weight=1.):
        self.model=model;self.spatial=spatial;self.weight=weight

    def predict_proba(self,x):
        p=self.model.predict_proba(x[COLUMNS])
        if self.spatial is not None:
            p=self.weight*p+(1-self.weight)*self.spatial.predict_proba(x)
        return p


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name=='v25_spatial_blend':
        return spatial_expert.fit_spatial(name,matrices,train_idx,y,predict_idx,seed)
    config=iterate.CONFIGS[name];x,cats=matrices[False];x=x[COLUMNS]
    key=(seed,hashlib.sha256(np.asarray(train_idx).tobytes()+np.asarray(predict_idx).tobytes()).hexdigest())
    model=None
    if key in CACHE and len(train_idx)<8693:
        prob=CACHE[key].copy()
    else:
        model=ExplainableBoostingClassifier(**PARAMS,random_state=seed,
            feature_types=['nominal' if c in cats else 'continuous' for c in COLUMNS])
        model.fit(x.iloc[train_idx],np.asarray(y))
        prob=model.predict_proba(x.iloc[predict_idx])[:,1]
        CACHE[key]=prob.copy()
    spatial=None
    if config['model']=='ebm_spatial':
        sp,spatial=spatial_expert.fit_spatial('v25_spatial_blend',matrices,train_idx,y,predict_idx,seed)
        prob=config['weight']*prob+(1-config['weight'])*sp
    predictor=EBMPredictor(model,spatial,config.get('weight',1.)) if model is not None else None
    return prob,predictor


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_010');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['ebm_candidate.py','spatial_expert.py','requirements-ebm.lock.txt']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['ebm_policy']='15 fixed compact features; EBM binning, interaction selection and early-stopping validation all fitted inside each outer training fold. Fixed 50/50 EBM/spatial probability blend, no OOF weight tuning.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
