"""Fixed CPU Deep Sets peer-mean vs peer-masked matched training procedure."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import time
import signal
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/deepsets_peer_probe'
NUM=['Age','CabinNumber','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
CAT=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
SETTINGS=dict(hidden=64,dropout=.1,lr=.003,weight_decay=.001,epochs=250,patience=30,grad_clip=1.,CPU_threads=4,budget_seconds=600)

class Prep:
    def fit(self,x):
        n=x[NUM].to_numpy(float);n[:,2:]=np.log1p(n[:,2:]);self.median=np.nanmedian(n,axis=0);self.median=np.nan_to_num(self.median)
        n=np.where(np.isnan(n),self.median,n);self.mean=n.mean(0);self.std=n.std(0);self.std[self.std==0]=1
        self.vocab={c:sorted(set(x[c].astype(str))|{'__MISSING__'}) for c in CAT};return self
    def transform(self,x):
        n=x[NUM].to_numpy(float);missing=np.isnan(n).astype(float);n[:,2:]=np.log1p(n[:,2:]);n=(np.where(np.isnan(n),self.median,n)-self.mean)/self.std
        parts=[n,missing]
        for c in CAT:
            v=x[c].astype(str).to_numpy();voc=self.vocab[c];parts.extend([v[:,None]==np.asarray(voc),~np.isin(v,voc)[:,None]])
        result=np.column_stack(parts).astype(np.float32)
        if not np.isfinite(result).all():raise ValueError('Nonfinite prep')
        return result

def peer_mean(h,groups,size):
    total=torch.zeros((len(size),h.shape[1]),dtype=h.dtype,device=h.device).index_add(0,groups,h)
    denominator=(size[groups]-1).clamp(min=1).unsqueeze(1)
    return (total[groups]-h)/denominator

class Net(nn.Module):
    def __init__(self,n,use_peers):
        super().__init__();self.use_peers=use_peers
        self.encoder=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,64),nn.GELU())
        self.head=nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x,groups,size):
        h=self.encoder(x);p=peer_mean(h,groups,size)
        if not self.use_peers:p=torch.zeros_like(p)
        count=torch.log1p(size[groups]-1).unsqueeze(1)
        return self.head(torch.cat([h,p,count],dim=1)).squeeze(1)

def seeded(seed):torch.manual_seed(seed);np.random.seed(seed)

def train_stage(frame,train_ids,y,groups,size,seed,use_peers,val_ids=None,val_y=None,epochs=None):
    # No labels of peer nodes enter encoder/aggregation, only selected training y in BCE.
    seeded(seed);prep=Prep().fit(frame.iloc[train_ids]);x=torch.from_numpy(prep.transform(frame));model=Net(x.shape[1],use_peers)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001);target=torch.tensor(y,dtype=torch.float32);train=torch.tensor(train_ids)
    curves=[];best=float('inf');selected=0;bad=0;limit=SETTINGS['epochs'] if epochs is None else epochs
    for epoch in range(1,limit+1):
        model.train();opt.zero_grad(set_to_none=True);logits=model(x,groups,size)
        loss=nn.functional.binary_cross_entropy_with_logits(logits[train],target);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        row=dict(epoch=epoch,train_bce=float(loss.detach()))
        if val_ids is not None:
            model.eval()
            with torch.no_grad():vl=float(nn.functional.binary_cross_entropy_with_logits(model(x,groups,size)[val_ids],torch.tensor(val_y,dtype=torch.float32)))
            row['validation_logloss']=vl
            if vl<best:best=vl;selected=epoch;bad=0
            else:bad+=1
        curves.append(row)
        if val_ids is not None and bad>=SETTINGS['patience']:break
    if val_ids is not None:return selected,dict(curves=curves,selected_epoch=selected,hit_cap=len(curves)==limit,stop='patience' if bad>=30 else 'cap')
    model.eval()
    with torch.no_grad():p=torch.sigmoid(model(x,groups,size)).numpy()
    return p,dict(curves=curves,epochs=limit,parameters=sum(v.numel() for v in model.parameters()),input_dimension=x.shape[1])

def check():
    h=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    g=torch.tensor([0,0,1]);size=torch.tensor([2.,1.]);p=peer_mean(h,g,size)
    torch.testing.assert_close(p,torch.tensor([[3.,4.],[1.,2.],[0.,0.]]))
    order=torch.tensor([2,0,1]);torch.testing.assert_close(peer_mean(h[order],g[order],size),p[order])
    torch.manual_seed(1);m=Net(2,False).eval()
    with torch.no_grad():before=m(h,g,size);h2=h.clone();h2[1]+=100;after=m(h2,g,size)
    torch.testing.assert_close(before[0],after[0])

def run():
    check()
    if REPORT.exists():raise FileExistsError('Fixed experiment no overwrite')
    REPORT.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(4)
    paths=['deepsets_peer_probe.py','error_overlap_diagnostic.py','relational_error_corrector.py','exaone_validation_data.py',
        'reports/v96_oof_completion/result.json','artifacts/v96_oof_completion/result.json','artifacts/v96_oof_completion/oof.npz','artifacts/error_rescue_20261117/catalog_rows.csv']
    sources={p:r.sha(ROOT/p) for p in paths}
    plan=dict(status='fixed_before_training',settings=SETTINGS,
        architecture='two64widthselfencoder plus exclude-self mean of sameGroup allroster encodings; [self64,peer64,log(peerCount+1)]→64→binary. Matched control has peer vector0, retains samecountinput and architecture.',
        preprocessing='Each innertrain or fulloutertrain alone fits numeric logspending/median/mean/std and categorical vocabulary; numericmissing indicators and unknown onehot. No Name/PassengerId/GroupId as network feature.',
        labels='Only current training IDs BCE labels. All12970 public attribute rows may serve as unlabeled peers; no group label/stat/teacherprediction fed network. Old1739 labels never loaded.',
        evaluation='Original6954dev SKF3seed20261112 to compare/halfblend matchedv96; internal SGKF5 firstfold seed20261112+outer, train-onlyepochselection then discard/reseed/refit fresh sameepochs on fulloutertrain.',
        mixtures='For both independently reported masked and peer models, fixed0.5 NN+0.5 v96. No weight/model/seedselection.',
        limitations=['Outer random splits allow some supervised groupmates; this is transductive randomCV, not deployment-matched groupCV.','Both models receive groupcount; effect estimates learned peermean beyondcount.','Same reused dev labels; hard546 reporting only after predictions.','Short fixed CPU recipe cannot negate all DeepSets architectures; report cap/curves.'],sources_sha256=sources)
    r.write(REPORT/'protocol.json',plan)
    handler=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('FixedCPU600s budget')));signal.setitimer(signal.ITIMER_REAL,600)
    try:
        data,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y'];frame=data['features']
        fullids=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=['PassengerId']) for f in ['train.csv','test.csv']],ignore_index=True).PassengerId.astype(str)
        fullgroups=fullids.str.split('_').str[0].to_numpy();_,g=np.unique(fullgroups,return_inverse=True);groups=torch.tensor(g,dtype=torch.long);size=torch.tensor(np.bincount(g),dtype=torch.float32)
        report96=json.loads((ROOT/'reports/v96_oof_completion/result.json').read_text())
        if report96!=json.loads((ROOT/'artifacts/v96_oof_completion/result.json').read_text()) or report96['status']!='complete_v96_fixed15_original3fold_oof':raise RuntimeError('v96 incomplete')
        path=ROOT/'artifacts/v96_oof_completion/oof.npz'
        if r.sha(path)!=report96['oof_sha256']:raise RuntimeError('v96 OOF SHA mismatch')
        with np.load(path,allow_pickle=False) as z:
            for k,w in [('dev_ids',dev),('fold_id',folds),('passenger_ids',data['passengers'])]:np.testing.assert_array_equal(z[k],w)
            v96=z['prob'].copy()
        evidence.update(sources);r.write(REPORT/'evidence.json',evidence)
        pred={k:np.full(len(y),np.nan) for k in ['masked','peers','masked_half_v96','peers_half_v96']};pred['v96']=v96;pred['raw']=bases['rawMitra'];records=[];seen=np.zeros(len(y),bool)
        with threadpool_limits(limits=4):
            for fold,(a,b) in enumerate(splits):
                seed=20261112+fold;ia,iv=next(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(a,y[a],fullgroups[dev[a]]))
                if set(fullgroups[dev[a[ia]]])&set(fullgroups[dev[a[iv]]]):raise RuntimeError('Inner group leakage')
                seen[b]=np.isin(fullgroups[dev[b]],fullgroups[dev[a]])
                record=dict(fold=fold,seed=seed,models={})
                for key,peers in [('masked',False),('peers',True)]:
                    t=time.monotonic();epoch,selection=train_stage(frame,dev[a[ia]],y[a[ia]],groups,size,seed,peers,dev[a[iv]],y[a[iv]])
                    p,refit=train_stage(frame,dev[a],y[a],groups,size,seed,peers,epochs=epoch)
                    pred[key][b]=p[dev[b]];pred[key+'_half_v96'][b]=.5*pred[key][b]+.5*v96[b]
                    record['models'][key]=dict(selection=selection,refit=refit,seconds=time.monotonic()-t,metrics=r.score(pred[key][b],y[b]))
                    print(f'fold{fold} {key} epoch{epoch} {record["models"][key]["metrics"]}',flush=True)
                records.append(record);r.write(REPORT/f'fold{fold}.json',record)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for k,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[k],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        masks=dict(all=np.ones(len(y),bool),hard546=hard,group_seen=seen,group_unseen=~seen,hard546_group_unseen=hard&~seen)
        def stat(m):return {k:dict(correct=int((((p>=.5)==y)&m).sum()),vs_v96=r.pair(p,v96,y,m),vs_masked=r.pair(p,pred['masked'],y,m),vs_masked_half=r.pair(p,pred['masked_half_v96'],y,m)) for k,p in pred.items()}
        for i,record in enumerate(records):record['paired']=stat(folds==i);record['hard546']=stat((folds==i)&hard);r.write(REPORT/f'fold{i}.json',record)
        cache=ROOT/'artifacts/deepsets_peer_oof.npz';np.savez_compressed(cache,**pred,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str))
        for name,digest in evidence.items():
            if r.sha(ROOT/name)!=digest:raise RuntimeError('Evidence changed '+name)
        result=dict(status='complete',metrics={k:r.score(p,y) for k,p in pred.items()},subsets={k:stat(m) for k,m in masks.items()},
            folds=records,seconds=time.monotonic()-start,protocol_sha256=r.sha(REPORT/'protocol.json'),oof_sha256=r.sha(cache),old_holdout_labels_read=False,no_upload=True)
        r.write(REPORT/'result.json',result);print(json.dumps(result['metrics']),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
if __name__=='__main__':run()
