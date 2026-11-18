"""One separately reported group-disjoint holdout, identical fixed Deep Sets recipe."""
import json
from pathlib import Path
import time
import signal
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import deepsets_peer_probe as core
r=core.r;ROOT=core.ROOT;OUT=ROOT/'reports/deepsets_group_holdout'

def run():
    if OUT.exists():raise FileExistsError('No repeat')
    OUT.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(4)
    sources={p:r.sha(ROOT/p) for p in ['deepsets_group_holdout.py','deepsets_peer_probe.py','reports/deepsets_peer_probe/protocol.json','reports/deepsets_peer_probe/result.json','artifacts/error_rescue_20261117/catalog_rows.csv']}
    plan=dict(status='fixed_before_group_fit',outer='SGKF3shuffle seed20261112 firstfold only',inner='SGKF5shuffle seed20261112 firstfold of outertrain',
        seed=20261112,recipe=core.SETTINGS,control='Same masked-vs-mean peer network from random probe; no hyperparameter changes.',
        authorization='Parent explicitly requested one true whole-group holdout if initialCPU3fold fast; random completion23.998s.',
        no_v96_blend=True,no_base_reference_from_different_split=True,transductive='All12970 X may appear as peer attributes; whole queryGroup labels absent from training/selection.',
        limitations=['One group split, not threefold groupOOF or independent holdout.','Original546 set reporting only; already adaptively observed.'],sources_sha256=sources)
    r.write(OUT/'protocol.json',plan)
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU180s')));signal.setitimer(signal.ITIMER_REAL,180)
    try:
        data=core.overlap.data.load_data();dev,y=data['dev'],data['y'];frame=data['features']
        ids=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=['PassengerId']) for f in ['train.csv','test.csv']],ignore_index=True).PassengerId.astype(str)
        names=ids.str.split('_').str[0].to_numpy();_,codes=np.unique(names,return_inverse=True);g=torch.tensor(codes);size=torch.tensor(np.bincount(codes),dtype=torch.float32)
        a,b=next(StratifiedGroupKFold(3,shuffle=True,random_state=20261112).split(dev,y,names[dev]))
        ia,iv=next(StratifiedGroupKFold(5,shuffle=True,random_state=20261112).split(a,y[a],names[dev[a]]))
        if set(names[dev[a]])&set(names[dev[b]]) or set(names[dev[a[ia]]])&set(names[dev[a[iv]]]):raise RuntimeError('Groups overlap')
        sources.update(data['evidence']);r.write(OUT/'evidence.json',sources)
        pred={};logs={}
        with threadpool_limits(limits=4):
            for key,peers in [('masked',False),('peers',True)]:
                epoch,selection=core.train_stage(frame,dev[a[ia]],y[a[ia]],g,size,20261112,peers,dev[a[iv]],y[a[iv]])
                p,refit=core.train_stage(frame,dev[a],y[a],g,size,20261112,peers,epochs=epoch);pred[key]=p[dev[b]]
                logs[key]=dict(selection=selection,refit=refit);r.write(OUT/(key+'_training.json'),logs[key])
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for k,w in [('row_id',dev),('y',y),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[k],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool)[b]
        metrics={k:r.score(p,y[b]) for k,p in pred.items()}
        result=dict(status='complete_one_group_holdout',training_rows=len(a),query_rows=len(b),outer_group_intersection=0,inner_group_intersection=0,
            metrics=metrics,paired=r.pair(pred['peers'],pred['masked'],y[b]),hard546_subset_rows=int(hard.sum()),
            hard_correct={k:int((((p>=.5)==y[b])&hard).sum()) for k,p in pred.items()},hard_paired=r.pair(pred['peers'],pred['masked'],y[b],hard),
            selected_epochs={k:v['selection']['selected_epoch'] for k,v in logs.items()},hit_cap={k:v['selection']['hit_cap'] for k,v in logs.items()},
            seconds=time.monotonic()-start,protocol_sha256=r.sha(OUT/'protocol.json'),no_v96_blend=True)
        cache=ROOT/'artifacts/deepsets_group_holdout.npz';np.savez_compressed(cache,**pred,train_ids=dev[a],query_ids=dev[b],passenger_ids=np.asarray(data['passengers'][b],dtype=str))
        result['predictions_sha256']=r.sha(cache)
        for p,digest in sources.items():
            if r.sha(ROOT/p)!=digest:raise RuntimeError('Source changed '+p)
        r.write(OUT/'result.json',result);print(json.dumps(result,indent=2))
    except BaseException as exc:r.write(OUT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
