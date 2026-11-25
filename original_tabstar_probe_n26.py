"""Run the original TabSTAR checkpoint offline on one or more whole-booking splits."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from full_roster_booking_validation_n23 import ROOT, data

ART=ROOT/'artifacts/nov26_original_tabstar'
REPORT=ROOT/'reports/nov23_matched_learning_curve/original_tabstar_probe.json'
SHA='455272659647d442117ea31ca274e9b683d47295738957229a4ffb632f6fa465'
SOURCE='4d14a660ae32b978b9bcde8c9f39e119513cf030'


def audit():
    catalog=ART/f'TabSTAR-{SOURCE}/src/tabstar/datasets/all_datasets.py'
    mapping={}
    for kind in ast.parse(catalog.read_text()).body:
        if isinstance(kind,ast.ClassDef):
            for n in kind.body:
                if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant):
                    mapping[n.value.value]=n.targets[0].id
    ids=json.loads((ART/'manifest.json').read_text())['datasets']
    resolved=[mapping[i] for i in ids]
    assert len(set(ids))==len(resolved)==400
    matches={str(i):n for i,n in zip(ids,resolved) if any(k in (str(i)+' '+n).lower() for k in ('titanic','spaceship','cryo','transported'))}
    assert matches=={'40945':'BIN_TRANSPORTATION_TITANIC_SURVIVAL'}
    return dict(count=len(ids),resolved=len(resolved),matches=matches,manifest_sha256=hashlib.sha256((ART/'manifest.json').read_bytes()).hexdigest(),
                catalog_commit=SOURCE,weight_sha256=SHA,checkpoint_revision='73560fb23d34307f4b6fa849f38f8299a8980a4a',
                model_card='https://huggingface.co/alana89/TabSTAR',classic_titanic='https://www.openml.org/d/40945',
                manifest_url='https://huggingface.co/alana89/TabSTAR/blob/73560fb23d34307f4b6fa849f38f8299a8980a4a/pretrain_args.json',
                catalog_url=f'https://github.com/alanarazi7/TabSTAR/blob/{SOURCE}/src/tabstar/datasets/all_datasets.py',
                audit_scope='Matched public training manifest identifiers to author catalog names. Dataset contents were not fingerprinted.',
                implementation='tabstar==1.2.2, original TabStarModel',local_device='cpu')


def run(k,finetune=False,embedding_only=False,extra_holdout=(),output_dir=ART):
    import torch
    from safetensors.torch import load_file
    from tabstar.arch.arch import TabStarModel
    from tabstar.arch.config import TabStarConfig
    from tabstar.tabstar_verbalizer import TabSTARVerbalizer
    torch.set_num_threads(4)
    pd.options.future.infer_string=False  # Original TabSTAR's verbalizer expects object text on pandas 2.
    info=audit()
    path=ART/'tabstar.safetensors'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA
    raw,y,g,f=data()
    if extra_holdout and (not finetune or output_dir==ART):
        raise ValueError('Nested fine-tuning requires a separate output directory.')
    held=np.isin(f,(k,*extra_holdout))
    q=np.flatnonzero(held)
    rng=np.random.default_rng(8251+k)
    groups=rng.permutation(np.unique(g[~held]))
    count=pd.Series(g).value_counts()
    take=np.searchsorted(np.cumsum(count.loc[groups]),1536)+1
    t=np.flatnonzero(np.isin(g,groups[:take]))
    if finetune or embedding_only:
        t=np.flatnonzero(~held)
    assert not np.isin(g[t],g[q]).any()
    frame=raw.drop(columns=['PassengerId','Cabin']).copy()
    cabin=raw.Cabin.str.split('/',expand=True)
    frame['Deck'],frame['Berth'],frame['Side']=cabin[0],pd.to_numeric(cabin[1]),cabin[2]
    frame['Booking size']=raw.groupby(g).PassengerId.transform('size')
    ids=np.r_[t,q]
    v=TabSTARVerbalizer(is_cls=True)
    v.fit(frame.iloc[t],pd.Series(y[t].astype(int),index=frame.index[t],name='Transported'))
    table=v.transform(frame.iloc[ids],y=None)
    assert table.d_output==2
    model=TabStarModel(TabStarConfig.from_pretrained(ART,local_files_only=True))
    model.load_state_dict(load_file(str(path)),strict=True)
    model.eval()
    started=time.monotonic()
    with torch.inference_mode():
        reference=model(table.x_txt[:4],table.x_num[:4],2)
        unique,ix=np.unique(table.x_txt,return_inverse=True)
        lookup=model.get_textual_embedding(unique.reshape(-1,1)).squeeze(1)
        ix=ix.reshape(table.x_txt.shape)
        # Use the frozen contextual text representation once per distinct string. The numerical
        # fusion, interaction network and prediction head still run for every passenger.
        encs=[];scores=[]
        for offset in range(0,len(ids),32):
            sl=slice(offset,offset+32)
            z=model.numerical_fusion(lookup[ix[sl]],torch.as_tensor(table.x_num[sl],dtype=lookup.dtype))
            hidden=model.tabular_encoder(z)[:,:2]
            score=model.cls_head(hidden).squeeze(-1)
            if not offset: torch.testing.assert_close(reference,score[:4],atol=1.e-4,rtol=1.e-4)
            scores.extend(torch.softmax(score,dim=1)[:,1].tolist())
            encs.append(hidden.flatten(1).numpy())
    embedding=np.concatenate(encs)
    cache=ART/f'base_embedding{k}.npz'
    if (finetune or embedding_only) and not extra_holdout and not cache.exists():
        temporary=ART/f'.base_embedding{k}.{os.getpid()}.npz'
        np.savez_compressed(temporary,ids=ids,embedding=embedding)
        temporary.replace(cache)
    if embedding_only:
        return
    if finetune:
        # Fit all numerical-fusion and table-interaction weights, keeping the pretrained
        # text encoder frozen. Only external-fold labels enter supervised optimization.
        torch.manual_seed(5591+k)
        lookup=lookup.clone()
        x_num=torch.as_tensor(table.x_num,dtype=lookup.dtype)
        targets=torch.from_numpy(y[t].astype(np.int64))
        modules=[model.numerical_fusion,model.tabular_encoder,model.cls_head]
        for module in modules:
            module.train()
        opt=torch.optim.AdamW([p for mod in modules for p in mod.parameters()],lr=2.e-5,weight_decay=.01)
        for epoch in range(3):
            for rows in torch.randperm(len(t)).split(64):
                z=model.numerical_fusion(lookup[ix[rows.numpy()]],x_num[rows])
                logits=model.cls_head(model.tabular_encoder(z)[:,:2]).squeeze(-1)
                loss=torch.nn.functional.cross_entropy(logits,targets[rows])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for mod in modules for p in mod.parameters()],1.)
                opt.step()
            print(f'Fold {k}, completed training epoch {epoch+1}',flush=True)
        model.eval()
        preds=[]
        with torch.inference_mode():
            for offset in range(len(t),len(ids),32):
                sl=slice(offset,offset+32)
                z=model.numerical_fusion(lookup[ix[sl]],x_num[sl])
                logits=model.cls_head(model.tabular_encoder(z)[:,:2]).squeeze(-1)
                preds.extend(torch.softmax(logits,dim=1)[:,1].tolist())
        prob=np.array(preds)
        receipt=output_dir/f'.tuned{k}.{os.getpid()}.npz'
        np.savez_compressed(receipt,t=t,q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str),
                            train_y=y[t],checkpoint=SHA,finetuned=prob)
        receipt.replace(output_dir/f'tuned{k}.npz')
        print(dict(fold=k,seconds=round(time.monotonic()-started,2),held=len(q),
                   accuracy=accuracy_score(y[q],prob>=.5),log_loss=log_loss(y[q],prob)),flush=True)
        return
    scale=StandardScaler().fit(embedding[:len(t)])
    head=LogisticRegression(C=.01,max_iter=1000).fit(scale.transform(embedding[:len(t)]),y[t])
    p=head.predict_proba(scale.transform(embedding[len(t):]))[:,1]
    native=np.asarray(scores[len(t):])
    torch.manual_seed(4801)
    model.cls_head.train()
    opt=torch.optim.AdamW(model.cls_head.parameters(),lr=1.e-4,weight_decay=.01)
    tokens=torch.from_numpy(embedding.reshape(len(ids),2,-1))
    targets=torch.from_numpy(y[t].astype(np.int64))
    for _ in range(10):
        for idx in torch.randperm(len(t)).split(64):
            logits=model.cls_head(tokens[idx]).squeeze(-1)
            loss=torch.nn.functional.cross_entropy(logits,targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        adapted=torch.softmax(model.cls_head(tokens[len(t):]).squeeze(-1),dim=1)[:,1].numpy()
    np.savez_compressed(ART/f'fold{k}.npz',t=t,q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str),
                        train_y=y[t],checkpoint=SHA,readout=p,native=native,adapted_head=adapted)
    report=json.loads(REPORT.read_text()) if REPORT.exists() else dict(audit=info,folds={})
    report['folds'][str(k)]=dict(support=len(t),held=len(q),train_groups=take.item(),held_groups=len(np.unique(g[q])),
              unique_verbalizations=len(unique),seconds=round(time.monotonic()-started,2),
              native_accuracy=accuracy_score(y[q],native>=.5),readout_accuracy=accuracy_score(y[q],p>=.5),
              native_loss=log_loss(y[q],native),readout_loss=log_loss(y[q],p),
              adapted_head_accuracy=accuracy_score(y[q],adapted>=.5),adapted_head_loss=log_loss(y[q],adapted))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['folds'][str(k)],flush=True)


def error_coverage():
    from outside_panel_coverage_n26 import archives,projection
    from arbitration_against_scored_nine_n26 import champions
    from expanded_joint_error_witnesses_n26 import FEES,load_scores,sources
    raw,y,g,f,views,panel=projection(include_full=True)
    nine,known,_,_=champions()
    archived,_=archives(raw,y,g,f,with_limix=True)
    other=np.column_stack([score>=.5 for score in archived.values()])
    extra=load_scores(raw,y,g,sources()[-1])['full']>=.5
    dry=~raw[FEES].gt(0).any(axis=1).to_numpy()
    report={}
    witnesses={}
    for name in ('native','readout','adapted_head','finetuned'):
        prob=np.full(len(y),np.nan)
        for k in range(8):
            path=ART/(f'tuned{k}.npz' if name=='finetuned' else f'fold{k}.npz')
            if not path.exists():
                continue
            with np.load(path) as z:
                t,q=z['t'],z['q']
                np.testing.assert_array_equal(q,np.flatnonzero(f==k))
                if name=='finetuned':
                    np.testing.assert_array_equal(t,np.flatnonzero(f!=k))
                np.testing.assert_array_equal(z['train_y'],y[t])
                np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
                assert not np.isin(g[t],g[q]).any() and str(z['checkpoint'])==SHA
                prob[q]=z[name]
        valid=np.isfinite(prob)
        if not valid.any():
            continue
        result=prob>=.5
        accuracy=lambda m: float(np.mean(result[m]==y[m]))
        per_fold={str(k):dict(held=int((f==k).sum()),accuracy=accuracy(f==k)) for k in np.unique(f[valid])}
        info=dict(held=int(valid.sum()),accuracy=accuracy(valid),log_loss=log_loss(y[valid],prob[valid]),
                  folds=np.unique(f[valid]).tolist(),per_fold=per_fold,views={})
        for key,(mask,old) in views.items():
            both=valid & known & mask & (nine==old)
            wrong=both & (old!=y)
            correct=both & ~wrong
            info['views'][key]=dict(shared_wrong=int(wrong.sum()),recovered=int((wrong & (result==y)).sum()),
                                    shared_correct=int(correct.sum()),broken=int((correct & (result!=y)).sum()))
            previous_consensus=both & (other==old[:,None]).all(axis=1)
            new_wrong=wrong & previous_consensus
            new_correct=correct & previous_consensus
            info['views'][key]['also_all_12_outsiders']=dict(
                shared_wrong=int(new_wrong.sum()),recovered=int((new_wrong & (result==y)).sum()),
                shared_correct=int(new_correct.sum()),broken=int((new_correct & (result!=y)).sum()))
            disagrees=panel!=old[:,None]
            disagrees[dry,-2:]=False
            blind=previous_consensus & ~disagrees.any(axis=1) & (extra==old[:,None]).all(axis=1)
            blind_wrong=blind & wrong
            blind_correct=blind & correct
            if key=='full' and valid.all():
                assert blind_wrong.sum()==178
            info['views'][key]['uncovered_after_all_archives_and_11']=dict(
                shared_wrong=int(blind_wrong.sum()),recovered=int((blind_wrong & (result==y)).sum()),
                shared_correct=int(blind_correct.sum()),broken=int((blind_correct & (result!=y)).sum()))
            if key=='full':
                witnesses[name+'_recover']=np.flatnonzero(blind_wrong & (result==y))
                witnesses[name+'_harm']=np.flatnonzero(blind_correct & (result!=y))
                if valid.all():
                    witnesses['all_wrong']=np.flatnonzero(blind_wrong)
                    witnesses['all_correct']=np.flatnonzero(blind_correct)
            if name=='finetuned':
                confidence=np.maximum(prob,1-prob)
                thresholds={}
                for level in (.70,.85,.95):
                    active=both & (old!=result) & (confidence>=level)
                    thresholds[str(level)]=dict(changed=int(active.sum()),recovered=int((wrong & active).sum()),
                                                broken=int((correct & active).sum()))
                info['views'][key]['prespecified_confidence']=thresholds
        unanimous=valid & (panel==panel[:,:1]).all(axis=1)
        wrong=unanimous & (panel[:,0]!=y)
        correct=unanimous & ~wrong
        info['panel_99']=dict(unanimous_wrong=int(wrong.sum()),recovered=int((wrong & (result==y)).sum()),
                              unanimous_correct=int(correct.sum()),broken=int((correct & (result!=y)).sum()))
        strict=unanimous & (other==panel[:,:1]).all(axis=1)
        wrong=strict & (panel[:,0]!=y)
        correct=strict & ~wrong
        info['panel_99_and_12_outsiders']=dict(unanimous_wrong=int(wrong.sum()),recovered=int((wrong & (result==y)).sum()),
                                                unanimous_correct=int(correct.sum()),broken=int((correct & (result!=y)).sum()))
        report[name]=info
    output=REPORT.with_name('original_tabstar_error_coverage.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ART/'strict_witnesses.npz',**witnesses)
    metadata=json.loads(REPORT.read_text())
    metadata['audit']=audit()
    metadata['protocol']={
        'excluded_fields':['Name','PassengerId'],
        'grouping':'Whole PassengerId prefix orders; eight shuffled GroupKFold partitions from full_roster_booking_validation_n23.',
        'input':'Age, expenses, destination, home planet, VIP, CryoSleep, cabin components and observed booking size.',
        'small_support':'Shuffle non-held booking groups with seed 8251+fold; select whole groups until at least 1536 passengers. The verbalizer, scaler and fitted heads use only training orders.',
        'readout':'Logistic regression C=0.01 on two flattened contextualized target tokens after the original table encoder.',
        'adapted_head':'Freeze all prior stages and fit original classification head for ten epochs at learning rate 0.0001.',
        'finetuned':'All non-held orders; train numerical fusion, table interaction and prediction head, freeze E5. Three epochs, batches of 64, AdamW lr=0.00002 and weight decay=0.01; seed=5591+fold.',
        'reference_forward':'Validated official complete-model forward and cached full-model path on four rows per fold to 1e-4 tolerance.',
        'probability_keys':'fold{k}.npz: native,readout,adapted_head. tuned{k}.npz: finetuned. Training and query indices t/q alongside train_y, held PassengerId, checkpoint SHA.',
        'software':'Python 3.12; torch 2.13.0, transformers 4.57.6, tabstar 1.2.2, peft 0.17.1, accelerate 1.10.1, pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, skrub 0.10.1, safetensors 0.8.0.'
    }
    REPORT.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,choices=range(8),default=0)
    parser.add_argument('--finetune',action='store_true')
    parser.add_argument('--base-embedding',action='store_true')
    parser.add_argument('--coverage',action='store_true')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['E5_SMALL_LOCAL_PATH']=str(ART/'e5')
    sys.path.insert(0,str(ART/'package'))
    args=parser.parse_args()
    if args.coverage:
        error_coverage()
    elif args.finetune:
        lock=ART/f'fold{args.fold}.lock'
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            output=ART/f'tuned{args.fold}.npz'
            if output.exists():
                with np.load(output) as z:
                    assert str(z['checkpoint'])==SHA and np.isfinite(z['finetuned']).all()
                print(f'Validated cached fine-tuned predictions for fold {args.fold}.')
            else:
                run(args.fold,finetune=True)
    else:
        run(args.fold,args.finetune,args.base_embedding)
