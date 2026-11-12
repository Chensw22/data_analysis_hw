"""Controlled CatBoost encoding experiments proposed by the two-agent review."""
import argparse
import hashlib
import json
import shutil

import iterate

iterate.CONFIGS = {
    iterate.BASELINE: dict(context=False,model='catboost',**iterate.CAT),
    'v16_onehot10': dict(context=False,model='catboost',**iterate.CAT,one_hot_max_size=10),
    'v17_ordered': dict(context=False,model='catboost',**iterate.CAT,boosting_type='Ordered'),
}
iterate.BLEND=[]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_004');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        path=iterate.ROOT/'refine.py'
        shutil.copy2(path,out/'refine_source.py')
        protocol=json.loads((out/'protocol.json').read_text())
        protocol['sources']['refine.py']=hashlib.sha256(path.read_bytes()).hexdigest()
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
