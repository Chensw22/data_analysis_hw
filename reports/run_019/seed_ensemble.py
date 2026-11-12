"""Fixed seed ensemble of the strongest existing feature/model structure."""
import argparse
import hashlib
import json
import shutil
import numpy as np
import iterate
import contrast_features

iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v34_contrasts_spatial':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v44_seed_ensemble':dict(context=False,model='seed_ensemble',seed_offsets=[0,100,200]),
}
iterate.BLEND=[]


class EnsemblePredictor:
    def __init__(self,models):
        self.models=models

    def predict_proba(self,x):
        return np.mean([m.predict_proba(x) for m in self.models],axis=0)


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name!='v44_seed_ensemble':
        return contrast_features.fit(name,matrices,train_idx,y,predict_idx,seed)
    results=[contrast_features.fit('v34_contrasts_spatial',matrices,train_idx,y,predict_idx,seed+offset)
             for offset in iterate.CONFIGS[name]['seed_offsets']]
    return np.mean([p for p,_ in results],axis=0),EnsemblePredictor([m for _,m in results])


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_019');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['seed_ensemble.py','contrast_features.py','spatial_expert.py']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['seed_ensemble_policy']='Same v34 features, model, data, spatial route, threshold. Three predeclared seed offsets 0/100/200; equal probability average, no selection of individual seeds.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
