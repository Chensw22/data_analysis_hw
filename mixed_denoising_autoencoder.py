"""Mixed-type masked reconstruction versus matched scratch; CPU-only fixed3fold trial."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/mixed_denoising_autoencoder'
REPORT=ROOT/'reports/mixed_denoising_autoencoder'
NUM=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','CabinNumber']
CAT=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
SETTINGS=dict(seed=20261118,threads=4,batch=256,embedding=8,hidden=[128,64],dropout=.1,
    masking_probability=.3,pretrain_epochs=100,supervised_cap=200,patience=20,min_epochs=20,
    min_delta=.0001,inner_fraction=.2,learning_rate=.001,weight_decay=.0001,
    huber_delta=1.,threshold=.5,total_deadline_seconds=1800)
OWN=['mixed_denoising_autoencoder.py','error_overlap_diagnostic.py','exaone_validation_data.py','exception_error_detector_nov18.py']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


class Prep:
    def fit(self,frame):
        x=frame[NUM].to_numpy(float);x[:,1:6]=np.log1p(x[:,1:6])
        if np.isinf(x).any():raise ValueError('Invalid numeric values.')
        median=np.asarray([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in x.T])
        filled=np.where(np.isfinite(x),x,median)
        scale=filled.std(0);scale[scale==0]=1
        self.state=dict(median=median.tolist(),mean=filled.mean(0).tolist(),scale=scale.tolist(),
            vocab={c:sorted(set(frame[c].fillna('__MISSING__').astype(str))-{'__MISSING__'}) for c in CAT})
        return self

    def transform(self,frame):
        x=frame[NUM].to_numpy(float);known=np.isfinite(x);x[:,1:6]=np.log1p(x[:,1:6])
        x=(np.where(known,x,np.asarray(self.state['median']))-np.asarray(self.state['mean']))/np.asarray(self.state['scale'])
        cats=[];c_known=[]
        for c in CAT:
            s=frame[c].fillna('__MISSING__').astype(str);mp={v:i+3 for i,v in enumerate(self.state['vocab'][c])}
            mp['__MISSING__']=1;cats.append(s.map(mp).fillna(0).to_numpy(np.int64));c_known.append((s!='__MISSING__').to_numpy())
        return (torch.tensor(x,dtype=torch.float32),torch.tensor(np.column_stack(cats),dtype=torch.long),
                torch.tensor(np.column_stack([known,*c_known]),dtype=torch.bool))

    def cards(self):return [len(self.state['vocab'][c])+3 for c in CAT]


class Net(nn.Module):
    def __init__(self,cards):
        super().__init__();self.cards=cards
        self.emb=nn.ModuleList([nn.Embedding(c,8) for c in cards])
        self.encoder=nn.Sequential(nn.Linear(7+13+13+6*8,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(.1),
                                   nn.Linear(128,64),nn.LayerNorm(64),nn.ReLU())
        self.classifier=nn.Linear(64,1)
        self.decoder=nn.Sequential(nn.Linear(64,64),nn.ReLU())
        self.num_head=nn.Linear(64,7)
        self.cat_heads=nn.ModuleList([nn.Linear(64,max(1,c-3)) for c in cards])

    def encode(self,data,mask=None):
        x,cats,known=data
        if mask is None:mask=torch.zeros_like(known)
        else:mask=mask&known
        masked_x=x.masked_fill(mask[:,:7],0)
        masked_c=cats.masked_fill(mask[:,7:],2)
        e=torch.cat([self.emb[j](masked_c[:,j]) for j in range(6)],dim=1)
        return self.encoder(torch.cat([masked_x,known.float(),mask.float(),e],dim=1))

    def forward(self,data):return self.classifier(self.encode(data)).squeeze(1)

    def reconstruction_loss(self,data,mask):
        x,cats,known=data;mask=mask&known;h=self.decoder(self.encode(data,mask));pn=self.num_head(h)
        losses=[]
        for j in range(7):
            selected=mask[:,j]
            if selected.any():losses.append(F.huber_loss(pn[selected,j],x[selected,j],delta=1.))
        for j in range(6):
            selected=mask[:,7+j]&(cats[:,j]>=3)
            if selected.any():losses.append(F.cross_entropy(self.cat_heads[j](h[selected]),cats[selected,j]-3))
        if not losses:return h.sum()*0
        return torch.stack(losses).sum()/13


def take(data,idx):return tuple(v[idx] for v in data)
def initialize(cards,seed):torch.manual_seed(seed);return Net(cards)


def pretrain(model,data,seed):
    torch.manual_seed(seed+1000);generator=torch.Generator().manual_seed(seed+2000)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];start=time.monotonic()
    for epoch in range(1,101):
        model.train();total=0.;count=0
        for idx in torch.randperm(len(data[0]),generator=generator).split(256):
            batch=take(data,idx);mask=(torch.rand(batch[2].shape,generator=generator)<.3)&batch[2]
            opt.zero_grad(set_to_none=True);loss=model.reconstruction_loss(batch,mask)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite reconstruction loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(idx);count+=len(idx)
        history.append(total/count)
        if epoch%25==0:print('pretrain',epoch,history[-1],flush=True)
    opt.zero_grad(set_to_none=True);del opt
    return dict(epochs=100,loss=history,seconds=time.monotonic()-start,rows=len(data[0]))


def predict(model,data):
    model.eval()
    with torch.no_grad():return torch.cat([model(take(data,idx)).sigmoid() for idx in torch.arange(len(data[0])).split(512)]).numpy()


def supervised(model,data,y,seed,validation=None,epochs=None):
    torch.manual_seed(seed+3000);generator=torch.Generator().manual_seed(seed+4000)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    target=torch.tensor(y,dtype=torch.float32);history=[];best=float('inf');selected=0;stale=0;start=time.monotonic()
    for epoch in range(1,(epochs or 200)+1):
        model.train();summed=0.
        for idx in torch.randperm(len(y),generator=generator).split(256):
            opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(take(data,idx)),target[idx])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();summed+=float(loss.detach())*len(idx)
        row=dict(epoch=epoch,train_loss=summed/len(y))
        if validation is not None:
            val_data,val_y=validation;vl=float(log_loss(val_y,predict(model,val_data),labels=[0,1]));row['validation_loss']=vl
            if vl<best-.0001:best=vl;selected=epoch;stale=0
            else:stale+=1
        history.append(row)
        if epoch%25==0:print('supervised',row,flush=True)
        if validation is not None and epoch>=20 and stale>=20:break
    opt.zero_grad(set_to_none=True);del opt
    return dict(selected_epoch=selected if validation is not None else epochs,history=history,
        cap_hit=validation is not None and len(history)==200,seconds=time.monotonic()-start,rows=len(y))


def check():
    p=json.loads((REPORT/'protocol.json').read_text())
    if p['settings']!=SETTINGS:raise RuntimeError('Recipe differs.')
    for name,h in p['source_hashes'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Source changed: '+name)


def worker(fold):
    import pandas as pd
    torch.set_num_threads(4);torch.set_num_interop_threads(4);started=time.monotonic();record=dict(status='running',fold=fold,models={})
    try:
        check();packet=json.loads((OUT/f'fold{fold}_packet.json').read_text())
        if set(packet)!={'columns','train','query','train_y','train_ids','query_ids'}:raise RuntimeError('Unexpected packet fields.')
        if set(packet['train_ids'])&set(packet['query_ids']):raise RuntimeError('Train/query overlap.')
        train=pd.DataFrame(packet['train'],columns=packet['columns']);query=pd.DataFrame(packet['query'],columns=packet['columns'])
        for c in NUM:train[c]=pd.to_numeric(train[c]);query[c]=pd.to_numeric(query[c])
        y=np.asarray(packet['train_y'],dtype=np.int64);seed=20261118+fold
        a,b=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=seed)
        record['inner_train_ids']=np.asarray(packet['train_ids'])[a].tolist();record['inner_validation_ids']=np.asarray(packet['train_ids'])[b].tolist()
        outputs={}
        for mode in ['scratch','denoising']:
            prep=Prep().fit(train.iloc[a]);idata=prep.transform(train.iloc[a]);ival=prep.transform(train.iloc[b])
            model=initialize(prep.cards(),seed);entry={}
            if mode=='denoising':entry['inner_pretraining']=pretrain(model,idata,seed)
            entry['selection']=supervised(model,idata,y[a],seed,validation=(ival,y[b]));n=entry['selection']['selected_epoch']
            del model,prep,idata,ival;gc.collect()
            prep=Prep().fit(train);data=prep.transform(train);qdata=prep.transform(query);model=initialize(prep.cards(),seed)
            if mode=='denoising':entry['full_pretraining']=pretrain(model,data,seed)
            entry['refit']=supervised(model,data,y,seed,epochs=n);prob=predict(model,qdata)
            target=OUT/f'fold{fold}_{mode}.pt';schema=OUT/f'fold{fold}_{mode}_preprocessor.json'
            torch.save(model.state_dict(),target);write(schema,prep.state)
            del model,prep,data,qdata;gc.collect()
            prep=Prep();prep.state=json.loads(schema.read_text());loaded=initialize(prep.cards(),seed)
            loaded.load_state_dict(torch.load(target,map_location='cpu',weights_only=True),strict=True)
            restored=predict(loaded,prep.transform(query));np.testing.assert_array_equal(prob,restored)
            entry.update(reload_max_abs_diff=float(np.max(np.abs(prob-restored))),model_sha256=sha(target),schema_sha256=sha(schema),parameter_count=sum(p.numel() for p in loaded.parameters()))
            outputs[mode]=prob;record['models'][mode]=entry
            del loaded,prep;gc.collect();write(REPORT/f'fold{fold}_progress.json',record)
        if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('Runtime flag changed.')
        np.savez(OUT/f'fold{fold}.npz',**outputs,query_ids=np.asarray(packet['query_ids'],dtype=np.int64))
        check();record.update(status='complete',prediction_sha256=sha(OUT/f'fold{fold}.npz'))
    except BaseException as e:record.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        record.update(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(REPORT/f'fold{fold}.json',record)


def score_all(probabilities,y,masks,frame):
    result={}
    for name,p in probabilities.items():
        good=(p>=.5)==y
        r=dict(accuracy=float(good.mean()),log_loss=float(log_loss(y,p,labels=[0,1])),rows=len(y),paired={})
        for ref in ['raw','v96','scratch','half_scratch_v96']:
            before=(probabilities[ref]>=.5)==y;r['paired'][ref]=dict(repairs=int((good&~before).sum()),breaks=int((~good&before).sum()),net=int(good.sum()-before.sum()))
        r['historical_subsets']={n:dict(rows=int(m.sum()),correct=int((m&good).sum())) for n,m in masks.items()}
        r['Earth']={}
        for cryo in ['True','False']:
            m=(frame.HomePlanet.astype(str).to_numpy()=='Earth')&(frame.CryoSleep.astype(str).to_numpy()==cryo);pred=p[m]>=.5;t=y[m].astype(bool)
            r['Earth'][cryo]=dict(rows=int(m.sum()),tn=int((~pred&~t).sum()),fp=int((pred&~t).sum()),fn=int((~pred&t).sum()),tp=int((pred&t).sum()))
        result[name]=r
    return result


def run():
    import pandas as pd
    import error_overlap_diagnostic as overlap
    if OUT.exists() or (REPORT/'protocol.json').exists():raise FileExistsError('No overwrite/resume.')
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True);start=time.monotonic();records=[]
    try:
        inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];frame=inputs['features'].iloc[dev]
        vp=ROOT/'artifacts/v96_oof_completion/oof.npz';vr=ROOT/'reports/v96_oof_completion/result.json';vreport=json.loads(vr.read_text())
        if vreport['status']!='complete_v96_fixed15_original3fold_oof' or sha(vp)!=vreport['oof_sha256']:raise RuntimeError('Invalid v96.')
        with np.load(vp,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['dev_ids'],dev);np.testing.assert_array_equal(z['fold_id'],folds);v96=z['prob'].copy()
        evidence.update({str(p.relative_to(ROOT)):sha(p) for p in [vp,vr]})
        plan=dict(status='fixed_before_fit',settings=SETTINGS,numeric=NUM,categorical=CAT,
            architecture='6 category embeddings8; concatenate numeric7,known13,mask13; Linear128/LayerNorm/ReLU/Dropout.1/Linear64/LayerNorm/ReLU. Linear1 classifier. Decoder64/ReLU + numeric7 and six categorical heads.',
            preprocessing='Only applicable train rows: five spend log1p, median impute, population standardization. Categories unknown0/missing1/artificialmask2/known3+. Numeric missing and categorical missing are explicit known=false.',
            self_supervised_objective='Independent mask p=.3 per originally known attribute. Sum of per-attribute masked known Huber(delta1) or categorical CE means divided by13. No loss on natural missing/unknown targets. No contrastive positives or negatives.',
            supervision='BCE, same scratch architecture and initialization seed, clean inputs. Inner20% label-stratified validation selects epochs; fresh outer preprocessing/model and optimizer refit fixed selected epochs.',
            pretraining='Fixed100epochs. Inner phase only innertraining attributes; fullrefit phase only outertraining attributes. No test/outervalidation/innervalidation attributes used in pretraining.',
            control='Identical architecture, supervisedoptimizer,batches,seed and stopping; omit reconstruction pretraining only. Epoch selections may differ by observed innerlogloss.',
            labels_boundary='Workers receive outertraining labels only; no historical error masks. All outervalidation labels remain in parent.',
            reporting='All6954 rows, per-fold, fixed .5 v96 means for both variants; historical546/616/914/5263 only after predictions.',
            source_hashes={n:sha(ROOT/n) for n in OWN},evidence_sha256=evidence,
            environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            limitations=['Adaptive reuseddev evaluation, not independentholdout.','Pretraining reconstruction is distinct from prior SCARF InfoNCE but has no new external information.','Fixed100pretrain epochs; curves and supervisedcaphits reported, no family-wide failure claim.'])
        write(REPORT/'protocol.json',plan)
        probabilities={n:np.full(len(y),np.nan) for n in ['scratch','denoising']}
        def vals(f):return [[None if pd.isna(v) else v for v in row] for row in f.to_numpy(dtype=object).tolist()]
        for fold,(a,b) in enumerate(splits):
            write(OUT/f'fold{fold}_packet.json',dict(columns=list(frame.columns),train=vals(frame.iloc[a]),query=vals(frame.iloc[b]),
                train_y=y[a].tolist(),train_ids=dev[a].tolist(),query_ids=dev[b].tolist()))
            print('Starting matched scratch+DAE fold',fold,flush=True)
            remaining=1800-(time.monotonic()-start)
            if remaining<=0:raise RuntimeError('Total budget exhausted.')
            with (OUT/f'fold{fold}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,__file__,'--worker',str(fold)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('Fixed1800s total budget exhausted.')
            if code:raise RuntimeError(f'Fold{fold} failed; preserved log.')
            r=json.loads((REPORT/f'fold{fold}.json').read_text());path=OUT/f'fold{fold}.npz'
            if r['status']!='complete' or sha(path)!=r['prediction_sha256']:raise RuntimeError('Incomplete fold.')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['query_ids'],dev[b])
                for n in probabilities:probabilities[n][b]=z[n]
            records.append(r)
            print(json.dumps({'fold':fold,'seconds':r['seconds'],'selected_epochs':{n:v['selection']['selected_epoch'] for n,v in r['models'].items()},'accuracy':{n:float(((p[b]>=.5)==y[b]).mean()) for n,p in probabilities.items()}}),flush=True)
        check()
        probabilities.update(raw=bases['rawMitra'],v96=v96)
        probabilities['half_scratch_v96']=.5*(probabilities['scratch']+v96)
        probabilities['half_denoising_v96']=.5*(probabilities['denoising']+v96)
        if any(not np.isfinite(p).all() for p in probabilities.values()):raise RuntimeError('Incomplete probabilities.')
        from exception_error_detector_nov18 import reporting_masks
        masks=reporting_masks(dev,folds,y,inputs,bases,evidence)
        np.savez(OUT/'oof.npz',**probabilities,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        result=dict(status='complete_matched3fold',metrics=score_all(probabilities,y,masks,frame),
            folds=[dict(fold=f,metrics=score_all({n:p[b] for n,p in probabilities.items()},y[b],{n:m[b] for n,m in masks.items()},frame.iloc[b]),training=records[f]) for f,(a,b) in enumerate(splits)],
            seconds=time.monotonic()-start,oof_path=str((OUT/'oof.npz').relative_to(ROOT)),oof_sha256=sha(OUT/'oof.npz'),
            source_hashes=plan['source_hashes'],evidence_sha256=evidence,protocol_sha256=sha(REPORT/'protocol.json'),limitations=plan['limitations'],no_upload=True)
        write(REPORT/'result.json',result)
        print(json.dumps({'status':result['status'],'metrics':{n:dict(accuracy=m['accuracy'],log_loss=m['log_loss'],vs_scratch=m['paired']['scratch'],hard546=m['historical_subsets']['all_recipes_common_wrong546']['correct']) for n,m in result['metrics'].items()}}),flush=True)
    except BaseException as e:
        write(REPORT/'failure.json',dict(status='incomplete',error_type=type(e).__name__,error=str(e),seconds=time.monotonic()-start,completed_folds=len(records)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    worker(args.worker) if args.worker is not None else run()
