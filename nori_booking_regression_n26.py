"""Nori-6M local regression. Run from the project root on Apple Silicon.

For a fresh checkout, place the official Kaggle train.csv in data/raw/train.csv.
The original Python 3.12.13 runtime is .venv-tabpfn/bin/python. An equivalent
isolated runtime can be created with uv venv --python 3.12.13 .venv-nori, then:
    uv pip install --python .venv-nori/bin/python --no-deps -r requirements-tabpfn.lock.txt
    uv pip install --python .venv-nori/bin/python --no-deps huggingface-hub==0.36.2 transformers==4.57.6 tokenizers==0.22.2 regex==2026.11.19 psutil==7.2.2
This reuses the project's PyTorch runtime; no TabPFN checkpoint is used.

Create artifacts/nov26_nori/lib. Download nori.pt from the exact revision
https://huggingface.co/Synthefy/Nori/resolve/1aca005143362552bb56c62e110ab77767cf4f77/nori.pt
to artifacts/nov26_nori/nori.pt. hf-mirror.com served the verified bytes in the
experiment. Download synthefy_nori-0.24.0-py3-none-any.whl from PyPI to the same
directory (https://pypi.org/project/synthefy-nori/0.24.0/#files). Extract it:
    unzip -q artifacts/nov26_nori/*.whl -d artifacts/nov26_nori/lib
All 92 packaged synthefy_nori/ files are compared byte-for-byte with this wheel.
Install overlay dependencies; commands leave the existing project environment intact:
    uv pip install --python .venv-nori/bin/python --target artifacts/nov26_nori/lib --no-deps pyvinecopulib==1.0.1 kditransform==1.2.0 synthefy==9.1.0 numba==0.68.1 pyarrow==26.0.1 llvmlite==0.50.0 array-api-compat==1.16.0 pydot==4.0.1
    uv pip install --python .venv-nori/bin/python --target artifacts/nov26_nori/lib --no-deps httpx==0.27.2 httpcore==1.0.9 anyio==4.16.0 h11==0.16.0 certifi==2026.11.20 idna==3.20 sniffio==1.3.1 typing-extensions==4.16.0

Run predict --context 384, then 1536 and 0 (entire training fold). Inference is
offline. After inference, report with the corresponding context under .venv.
The comparisons require the existing held-order Mitra and frozen release
artifacts; the stand-alone Nori predictions need only the competition CSV.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from time import monotonic

import numpy as np
import pandas as pd
from full_roster_booking_validation_n23 import ROOT,OUT,data

CACHE = ROOT/'artifacts/nov26_nori'
PUBLIC = ROOT/'reports/nov23_matched_learning_curve/nori_booking_regression.json'
SHA = 'a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323'
WHL = '4cb1f0da3a397db79dc28932f860b6df90e4aff3c0c29027987c11b6d55d40c0'
FEES = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def design(raw,groups,t):
    cabin = raw.Cabin.str.split('/',expand=True)
    parts = []
    for c in [raw.HomePlanet,raw.Destination,raw.VIP,raw.CryoSleep,cabin[0],cabin[2]]:
        s=c.astype('string').fillna('?')
        levels=sorted(s.iloc[t].unique())
        parts.append(pd.Categorical(s.where(s.isin(levels)),categories=levels).codes)
    fees = raw[FEES]
    size = raw.groupby(groups).PassengerId.transform('size')
    parts.extend([raw.Age,pd.to_numeric(cabin[1]),size,np.log1p(fees.sum(axis=1,min_count=1)),np.log1p(fees).to_numpy()])
    return np.column_stack(parts).astype(np.float32)


def predict(folds,context):
    os.environ['MPLCONFIGDIR']=str(CACHE/'mpl')
    os.environ['HF_HUB_OFFLINE']='1'
    sys.path.insert(0,str(CACHE/'lib'))
    import torch
    from synthefy_nori import NoriRegressor
    torch.set_num_threads(3)
    weight = CACHE/'nori.pt'
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==SHA
    whl=CACHE/'synthefy_nori-0.24.0-py3-none-any.whl'
    assert hashlib.sha256(whl.read_bytes()).hexdigest()==WHL
    with zipfile.ZipFile(whl) as archive:
        modules=[n for n in archive.namelist() if n.startswith('synthefy_nori/') and not n.endswith('/')]
        assert len(modules)==92 and all((CACHE/'lib'/n).read_bytes()==archive.read(n) for n in modules)
    members=json.loads((CACHE/'lib/synthefy_nori/configs/default_inference.json').read_text())
    recipe=CACHE/'one_member.json'
    recipe.write_text(json.dumps(members[:1]))
    raw,y,g,f=data()
    for k in folds:
        path=packet_dir(context)/f'fold{k}.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['checkpoint'])==SHA and str(z['recipe'])==recipe.read_text()
            continue
        start=monotonic()
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        ids=t if context==0 else np.sort(np.random.default_rng(6117+k).choice(t,context,replace=False))
        assert not np.isin(g[ids],g[q]).any()
        x=design(raw,g,ids)
        np.random.seed(6117+k)
        torch.manual_seed(6117+k)
        model=NoriRegressor(model_path=str(weight),model='nori-6m',device='cpu',inference_config=str(recipe),
                            categorical_columns=[],augmentations=())
        model.fit(x[ids],y[ids].astype(float))
        p=np.asarray(model.predict(x[q]),dtype=float).ravel()
        assert len(p)==len(q) and np.isfinite(p).all()
        np.savez_compressed(path,t=t,q=q,context=ids,ids=raw.PassengerId.iloc[q].to_numpy(str),labels=y[ids],p=p,
                            elapsed=monotonic()-start,checkpoint=SHA,recipe=recipe.read_text())
        print('Nori fold',k,'rows',len(q),'accuracy',float(np.mean((p>=.5)==y[q])),'seconds',round(monotonic()-start,1),flush=True)


def packet_dir(context):
    folder = CACHE if context==384 else CACHE/f'context_{context}'
    folder.mkdir(exist_ok=True)
    return folder


def report(context):
    from exhaustive_family_quorums_n26 import inputs
    from scipy.special import logit
    raw,y,g,f=data()
    _,cur,_=inputs()
    prior=np.load(OUT/'mitra96.npz')
    np.testing.assert_array_equal(prior['passenger'],raw.PassengerId.to_numpy(str))
    np.testing.assert_array_equal(prior['fold'],f)
    p=np.zeros(len(y)); complete=np.zeros(len(y),bool); by_fold={}
    for path in sorted(packet_dir(context).glob('fold[0-7].npz')):
        k=int(path.stem[-1])
        with np.load(path) as z:
            t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['t'],t)
            np.testing.assert_array_equal(z['ids'],raw.PassengerId.iloc[q].to_numpy(str))
            np.testing.assert_array_equal(z['labels'],y[z['context']])
            assert not np.isin(g[q],g[z['context']]).any() and str(z['checkpoint'])==SHA
            p[q]=z['p'];complete[q]=True
            by_fold[k]=dict(rows=len(q),provided_context_rows=len(z['context']),accuracy=float(np.mean((p[q]>=.5)==y[q])),seconds=float(z['elapsed']))
    base=cur['parent']; ref=prior['p']
    pred = p>=.5
    blend=3*logit(ref.clip(.001,.999))+logit(p.clip(.001,.999))>=0
    actions={'nori':pred!=base,'mitra':(ref>=.5)!=base,'nori_only':(pred!=base)&((ref>=.5)==base),
             'blend':blend!=base,'blend_extra':(blend!=base)&((ref>=.5)==base)}
    table={}
    for scope,eligible in [('known',cur['known']),('common',cur['common']),('new_orders',cur['known']&cur['new'])]:
        m=eligible&complete
        table[scope]={key:dict(rows=int(m.sum()),fixed=int((m&act&(base!=y)).sum()),harmed=int((m&act&(base==y)).sum()))
                      for key,act in actions.items()}
    result=dict(package='synthefy-nori==0.24.0',wheel_sha256=WHL,model='Synthefy/Nori nori-6m',model_sha256=SHA,
      weight_url='https://huggingface.co/Synthefy/Nori/blob/1aca005143362552bb56c62e110ab77767cf4f77/nori.pt',
      code='https://github.com/Synthefy/synthefy-nori',license='Apache-2.0',context=context,folds=by_fold,
      recipe='first official default preprocessing member; no target-specific training; threshold 0.5; 3:1 Mitra/Nori log-odds blend',
      accuracy=float(np.mean(pred[complete]==y[complete])),scopes=table)
    if context==0 and complete.all():
        aligned=dict(row=np.arange(len(y)),passenger=raw.PassengerId.to_numpy(str),fold=f,y=y)
        for size in (384,1536,0):
            all_scores=np.full(len(y),np.nan)
            for k in range(8):
                with np.load(packet_dir(size)/f'fold{k}.npz') as record:
                    q=np.flatnonzero(f==k)
                    np.testing.assert_array_equal(record['ids'],aligned['passenger'][q])
                    np.testing.assert_array_equal(record['q'],q)
                    all_scores[q]=record['p']
            assert np.isfinite(all_scores).all()
            aligned['full' if size==0 else f'context_{size}']=all_scores
        np.savez_compressed(CACHE/'aligned.npz',**aligned)
    results=json.loads(PUBLIC.read_text()) if PUBLIC.exists() else {}
    results[str(context)]=result
    PUBLIC.write_text(json.dumps(results,indent=2)+'\n')
    print(result['accuracy'],table,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['predict','report'])
    parser.add_argument('--folds',type=int,nargs='+',default=list(range(8)))
    parser.add_argument('--context',type=int,choices=[0,384,1536],default=384)
    args=parser.parse_args()
    predict(args.folds,args.context) if args.action=='predict' else report(args.context)
