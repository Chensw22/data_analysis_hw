"""Fixed spending contrasts for oblique boundaries with axis-aligned trees."""
import argparse
import hashlib
import json
import shutil
import numpy as np
import iterate
import spatial_expert

BASE_FIT=spatial_expert.ORIGINAL_FIT
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v33_spend_contrasts':dict(context=False,model='catboost',**iterate.CAT),
    'v34_contrasts_spatial':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
}
iterate.BLEND=[]


def add_contrasts(x):
    x=x.copy()
    dining=x[['FoodCourt','ShoppingMall']].sum(axis=1,min_count=1)
    luxury=x[['RoomService','Spa','VRDeck']].sum(axis=1,min_count=1)
    x['DiningShoppingMinusLuxury']=dining-luxury
    x['LogDiningShoppingLuxuryRatio']=np.log1p(dining)-np.log1p(luxury)
    return x


class ContrastPredictor:
    def __init__(self,model):
        self.model=model

    def predict_proba(self,x):
        return self.model.predict_proba(add_contrasts(x))


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name=='v25_spatial_blend':
        return spatial_expert.fit_spatial(name,matrices,train_idx,y,predict_idx,seed)
    x,cats=matrices[False];changed={False:(add_contrasts(x),cats)}
    if name=='v33_spend_contrasts':
        p,model=BASE_FIT(iterate.BASELINE,changed,train_idx,y,predict_idx,seed)
    else:
        p,model=spatial_expert.fit_spatial('v25_spatial_blend',changed,train_idx,y,predict_idx,seed)
    return p,ContrastPredictor(model)


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_012');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['contrast_features.py','spatial_expert.py']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['contrast_policy']='Only two fixed row-local features: FoodCourt+ShoppingMall minus RoomService+Spa+VRDeck; difference of log1p of those two sums. No target-derived coefficients. All-missing sums stay NaN.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
