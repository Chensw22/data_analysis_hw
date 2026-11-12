"""Strict nested stacking. Built with PriorLabs-TabPFN; all inference is local."""
import argparse
import gc
import hashlib
import json
import shutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import iterate
import spatial_expert
import tabpfn_candidate as foundation

BASE_FIT=spatial_expert.ORIGINAL_FIT
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v35_nested_stack':dict(context=False,model='nested',inner_folds=3,C=1.,n_estimators=8),
}
iterate.BLEND=[]
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt',
         'ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']


def logits(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


class StackPredictor:
    def __init__(self,spatial,pfn,meta):
        self.spatial=spatial;self.pfn=pfn;self.meta=meta

    def predict_proba(self,x):
        features=np.column_stack([self.spatial.predict_proba(x)[:,1],
                                  self.pfn.predict_proba(x[COMPACT])[:,1]])
        return self.meta.predict_proba(logits(features))


def pair_fit(matrices,train_idx,y,predict_idx,seed,keep=False):
    assert not set(train_idx)&set(predict_idx)
    spatial_p,spatial=spatial_expert.fit_spatial('v25_spatial_blend',matrices,train_idx,y,predict_idx,seed)
    x,cats=matrices[False];x=x[COMPACT]
    pfn=foundation.TabPFNClassifier.create_default_for_version(foundation.ModelVersion.V2,
        model_path=foundation.WEIGHTS,n_estimators=8,device='mps',random_state=seed,
        n_preprocessing_jobs=1,categorical_features_indices=[x.columns.get_loc(c) for c in cats if c in COMPACT])
    pfn.fit(x.iloc[train_idx],np.asarray(y))
    probabilities=np.column_stack([spatial_p,pfn.predict_proba(x.iloc[predict_idx])[:,1]])
    if not keep:
        spatial=None;pfn=None;gc.collect();foundation.torch.mps.empty_cache()
    return probabilities,spatial,pfn


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name=='v25_spatial_blend':
        return spatial_expert.fit_spatial(name,matrices,train_idx,y,predict_idx,seed)
    y=np.asarray(y);train_idx=np.asarray(train_idx)
    oof=np.full((len(y),2),np.nan)
    for inner,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+1000).split(train_idx,y)):
        assert not set(train_idx[a])&set(predict_idx)
        oof[b]=pair_fit(matrices,train_idx[a],y[a],train_idx[b],seed+inner)[0]
        print('Nested stack inner fold',inner+1,'completed; meta labels remain inside outer train',flush=True)
    assert np.isfinite(oof).all()
    meta=LogisticRegression(C=1.,max_iter=2000,random_state=seed)
    meta.fit(logits(oof),y)
    probabilities,spatial,pfn=pair_fit(matrices,train_idx,y,predict_idx,seed,keep=len(train_idx)==8693)
    result=meta.predict_proba(logits(probabilities))[:,1]
    print('Nested meta coefficients',meta.coef_.tolist(),'intercept',meta.intercept_.tolist(),flush=True)
    return result,StackPredictor(spatial,pfn,meta) if spatial is not None else None


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_013');args=parser.parse_args()
    foundation.torch.set_num_threads(4)
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['nested_stack.py','spatial_expert.py','tabpfn_candidate.py','requirements-tabpfn.lock.txt']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['checkpoint_sha256']=hashlib.sha256(foundation.WEIGHTS.read_bytes()).hexdigest()
        protocol['stack_policy']='Each outer training fold has fresh inner 3fold OOF; two clipped logits train fixed C=1 L2 logistic. Base models refit on outer training rows before predicting outer validation. No passthrough, weights/threshold search, or reuse of full-development OOF.'
        protocol['attribution']='Built with PriorLabs-TabPFN'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
