"""Local TabICLv2 inference with public synthetic-pretrained weights."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import argparse
import gc
import hashlib
import json
import resource
import shutil
import time
import numpy as np
import torch
from tabicl import TabICLClassifier
from sklearn.model_selection import StratifiedKFold
import iterate
import spatial_expert

BASE_FIT=spatial_expert.ORIGINAL_FIT
WEIGHTS=iterate.ROOT/'artifacts/pretrained/tabicl-classifier-v2-20260212.ckpt'
EXPECTED='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v36_TabICL_compact4':dict(context=False,model='tabicl',compact=True,n_estimators=4),
    'v37_TabICL_engineered4':dict(context=False,model='tabicl',compact=False,n_estimators=4),
    'v38_TabICL_spatial_blend':dict(context=False,model='tabicl_spatial',compact=True,n_estimators=4,weight=.5),
}
iterate.BLEND=[]
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt',
         'ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']
CACHE={}


class ICLPredictor:
    def __init__(self,model,columns,spatial=None,weight=1.):
        self.model=model;self.columns=columns;self.spatial=spatial;self.weight=weight

    def predict_proba(self,x):
        p=self.model.predict_proba(x[self.columns])
        if self.spatial is not None:
            p=self.weight*p+(1-self.weight)*self.spatial.predict_proba(x)
        return p


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return BASE_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name=='v25_spatial_blend':
        return spatial_expert.fit_spatial(name,matrices,train_idx,y,predict_idx,seed)
    config=iterate.CONFIGS[name];x,_=matrices[False]
    if config['compact']:
        x=x[COMPACT]
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==EXPECTED
    key=(config['compact'],seed,hashlib.sha256(np.asarray(train_idx).tobytes()+np.asarray(predict_idx).tobytes()).hexdigest())
    model=None
    if key in CACHE and len(train_idx)<8693:
        p=CACHE[key].copy()
    else:
        model=TabICLClassifier(n_estimators=4,batch_size=1,model_path=WEIGHTS,
            checkpoint_version='tabicl-classifier-v2-20260212.ckpt',
            allow_auto_download=False,device='mps',use_amp=False,use_fa3=False,
            kv_cache=False,random_state=seed,n_jobs=1,offload_mode='auto')
        started=time.monotonic();model.fit(x.iloc[train_idx],np.asarray(y))
        print('TabICL fitted; predicting',len(predict_idx),'rows',flush=True)
        p=model.predict_proba(x.iloc[predict_idx])[:,1]
        print('TabICL seconds',round(time.monotonic()-started,2),'max RSS bytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        print('MPS driver bytes after prediction',torch.mps.driver_allocated_memory(),flush=True)
        CACHE[key]=p.copy()
    spatial=None
    if config['model']=='tabicl_spatial':
        sp,spatial=spatial_expert.fit_spatial('v25_spatial_blend',matrices,train_idx,y,predict_idx,seed)
        p=config['weight']*p+(1-config['weight'])*sp
    predictor=ICLPredictor(model,list(x.columns),spatial,config.get('weight',1.)) if model is not None else None
    if len(train_idx)<8693:
        predictor=None;model=None;spatial=None;gc.collect();torch.mps.empty_cache()
    return p,predictor


iterate.fit=fit


def probe():
    train,_,matrices,dev=iterate.dataset();y=train.Transported.iloc[dev].astype(int).to_numpy()
    a,b=next(StratifiedKFold(3,shuffle=True,random_state=iterate.SEEDS[0]).split(dev,y))
    started=time.monotonic()
    p,_=fit('v36_TabICL_compact4',matrices,dev[a],y[a],dev[b[:128]],iterate.SEEDS[0])
    assert p.shape==(128,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    result=dict(train_rows=len(a),predict_rows=128,seconds=time.monotonic()-started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checkpoint_sha256=EXPECTED,valid_probabilities=True,
        note='Resource probe, no accuracy computed. Does not validate full 8693-row final fit.')
    (iterate.ROOT/'reports/tabicl_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['probe','screen','confirm','export'])
    parser.add_argument('--run',default='run_014');args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='probe':
        probe()
    else:
        out=iterate.ROOT/'artifacts'/args.run
        {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
        if args.stage=='screen':
            protocol=json.loads((out/'protocol.json').read_text())
            for name in ['tabicl_candidate.py','spatial_expert.py','requirements-tabicl.lock.txt']:
                path=iterate.ROOT/name;shutil.copy2(path,out/name)
                protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            protocol['checkpoint_sha256']=EXPECTED
            protocol['tabicl_policy']='Full outer train fits encoding/imputation/normalization; local fixed synthetic-pretrained model. FP32,4 estimators,batch1,no KV cache. Fixed .5 threshold and .5 spatial blend. Weights hash checked before each fit.'
            (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
