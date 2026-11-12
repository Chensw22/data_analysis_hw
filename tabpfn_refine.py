"""Built with PriorLabs-TabPFN: fixed feature ablation and ensemble-size comparison."""
import argparse
import hashlib
import json
import shutil
import gc
import numpy as np
import tabpfn_candidate as candidate
import iterate

ORIGINAL_FIT=candidate.ORIGINAL_FIT
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'TabPFN_v2_compact8':dict(context=False,model='tabpfn',compact=True,n_estimators=8),
    'TabPFN_v2_engineered8':dict(context=False,model='tabpfn',compact=False,n_estimators=8),
}
iterate.BLEND=[]
COMPACT=['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt',
         'ShoppingMall','Spa','VRDeck','Deck','CabinNumber','Side']


class SelectedColumns:
    def __init__(self,model,columns):
        self.model=model; self.columns=columns

    def predict_proba(self,x):
        return self.model.predict_proba(x[self.columns])


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return ORIGINAL_FIT(name,matrices,train_idx,y,predict_idx,seed)
    config=iterate.CONFIGS[name]
    x,cats=matrices[False]
    if config['compact']:
        x=x[COMPACT];cats=[c for c in cats if c in COMPACT]
    model=candidate.TabPFNClassifier.create_default_for_version(candidate.ModelVersion.V2,
        model_path=candidate.WEIGHTS,n_estimators=config['n_estimators'],device='mps',
        random_state=seed,n_preprocessing_jobs=1,
        categorical_features_indices=[x.columns.get_loc(c) for c in cats])
    model.fit(x.iloc[train_idx],np.asarray(y))
    p=model.predict_proba(x.iloc[predict_idx])[:,1]
    model=SelectedColumns(model,list(x.columns))
    if len(train_idx)<8693:
        model=None;gc.collect();candidate.torch.mps.empty_cache()
    return p,model


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_008');args=parser.parse_args()
    candidate.torch.set_num_threads(4)
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['tabpfn_refine.py','tabpfn_candidate.py','requirements-tabpfn.lock.txt']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['checkpoint_sha256']=hashlib.sha256(candidate.WEIGHTS.read_bytes()).hexdigest()
        protocol['attribution']='Built with PriorLabs-TabPFN'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
