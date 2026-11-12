"""Built with PriorLabs-TabPFN: local, fixed v2 weights; no cloud inference."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TABPFN_NO_BROWSER']='1'
os.environ['TABPFN_MODEL_VERSION']='v2'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

import argparse
import gc
import hashlib
import json
import shutil
import time
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.model_selection import StratifiedKFold
import iterate

WEIGHTS=iterate.ROOT/'artifacts/pretrained/tabpfn-v2-classifier-v2_default.ckpt'
ORIGINAL_FIT=iterate.fit
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'TabPFN_v2_2estimators':dict(context=False,model='tabpfn',n_estimators=2),
    'TabPFN_v2_catboost_blend':dict(context=False,model='tabpfn_blend',n_estimators=2,weight=.5),
}
iterate.BLEND=[]
PREDICTION_CACHE={}


class ProbabilityBlend:
    def __init__(self,foundation,catboost,weight):
        self.foundation=foundation;self.catboost=catboost;self.weight=weight

    def predict_proba(self,x):
        p=self.weight*self.foundation.predict_proba(x)+(1-self.weight)*self.catboost.predict_proba(x)
        return p


def fit_tabpfn(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return ORIGINAL_FIT(name,matrices,train_idx,y,predict_idx,seed)
    config=iterate.CONFIGS[name]
    x,cats=matrices[False]
    assert WEIGHTS.is_file(), 'Download the public v2 checkpoint first.'
    assert torch.backends.mps.is_available(), 'This run was planned for local Apple MPS.'
    key=(seed,hashlib.sha256(np.asarray(train_idx).tobytes()+np.asarray(predict_idx).tobytes()).hexdigest())
    foundation=None
    if key in PREDICTION_CACHE and len(train_idx)<8693:
        p=PREDICTION_CACHE[key].copy()
    else:
        foundation=TabPFNClassifier.create_default_for_version(ModelVersion.V2,model_path=WEIGHTS,
            n_estimators=config['n_estimators'],device='mps',random_state=seed,n_preprocessing_jobs=1,
            categorical_features_indices=[x.columns.get_loc(c) for c in cats])
        start=time.monotonic()
        foundation.fit(x.iloc[train_idx],np.asarray(y))
        print('TabPFN fit complete, predicting',len(predict_idx),'rows',flush=True)
        p=foundation.predict_proba(x.iloc[predict_idx])[:,1]
        print('TabPFN total seconds',round(time.monotonic()-start,2),flush=True)
        PREDICTION_CACHE[key]=p.copy()
    if config['model']=='tabpfn_blend':
        catprob,catmodel=ORIGINAL_FIT(iterate.BASELINE,matrices,train_idx,y,predict_idx,seed)
        p=config['weight']*p+(1-config['weight'])*catprob
        model=ProbabilityBlend(foundation,catmodel,config['weight']) if foundation is not None else None
    else:
        model=foundation
    if len(train_idx)<8693:
        # CV needs probabilities only; release each fold's accelerator state.
        model=None
        foundation=None
        if config['model']=='tabpfn_blend':
            catmodel=None
        gc.collect()
        torch.mps.empty_cache()
    return p,model


iterate.fit=fit_tabpfn


def probe():
    train,_,matrices,dev=iterate.dataset()
    y=train.Transported.iloc[dev].astype(int).to_numpy()
    a,b=next(StratifiedKFold(3,shuffle=True,random_state=iterate.SEEDS[0]).split(dev,y))
    started=time.monotonic()
    p,_=fit_tabpfn('TabPFN_v2_2estimators',matrices,dev[a],y[a],dev[b[:128]],iterate.SEEDS[0])
    assert len(p)==128 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    result=dict(training_rows=len(a),prediction_rows=128,seconds=time.monotonic()-started,
                mps_available=True,torch_version=torch.__version__,valid_probabilities=True,
                note='Compatibility/resource probe only. No accuracy calculated; not evidence of meeting the gate.')
    path=iterate.ROOT/'reports/tabpfn_probe.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['probe','screen','confirm','export'])
    parser.add_argument('--run',default='run_007');args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='probe':
        probe()
    else:
        out=iterate.ROOT/'artifacts'/args.run
        {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
        if args.stage=='screen':
            path=iterate.ROOT/'tabpfn_candidate.py';shutil.copy2(path,out/'tabpfn_candidate_source.py')
            protocol=json.loads((out/'protocol.json').read_text())
            protocol['sources']['tabpfn_candidate.py']=hashlib.sha256(path.read_bytes()).hexdigest()
            protocol['checkpoint_sha256']=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
            protocol['attribution']='Built with PriorLabs-TabPFN; Prior Labs License v1.1 (May2025)'
            (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
