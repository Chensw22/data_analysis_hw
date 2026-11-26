"""Offline RT-J smoke inference on anonymous passenger, booking and actual cabin relations."""
import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT, data
from rtj_checkpoint_setup_n26 import OUT, WEIGHTS

DB = 'spaceship-orders'
TASK = 'transport'
REPORT = ROOT/'reports/nov23_matched_learning_curve/rtj_order_smoke.json'


def prepare(outer=None,mode='edges'):
    raw,y,g,fold = data()
    rng=np.random.default_rng(128831)
    if outer is None:
        split=[rng.choice(np.unique(g[mask]),n,replace=False) for mask,n in [(fold!=0,384),(fold==0,8)]]
    else:
        split=[np.unique(g[mask]) for mask in (fold!=outer,fold==outer)]
    assert not np.intersect1d(*split).size
    loc=np.flatnonzero(np.isin(g,np.concatenate(split)))
    p=raw.iloc[loc].reset_index(drop=True).copy()
    passenger=p.pop('PassengerId')
    p['passenger']=np.arange(len(p))
    p['booking']=pd.factorize(g[loc],sort=True)[0]
    p['position']=passenger.str.split('_').str[1].astype(int)
    p['cabin']=pd.Series(pd.factorize(p.Cabin,sort=True)[0]).replace(-1,None).astype('Int64')
    cabins=p.loc[p.cabin.notna(),['cabin','Cabin']].drop_duplicates().sort_values('cabin')
    cabins[['deck','number','side']]=cabins.pop('Cabin').str.split('/',expand=True)
    cabins['number']=cabins.number.astype(float)
    decks=sorted(cabins.deck.unique())
    cabins['deck_key']=cabins.deck.map(dict(zip(decks,range(len(decks)))))
    p=p.drop(columns=['Cabin'])
    for field in ('VIP','CryoSleep'):
        p[field]=p[field].astype('string')
    name=DB if outer is None else f'{DB}-fold{outer}-{mode}'
    folder=OUT/name
    db=folder/'db';task=folder/'tasks'/TASK
    db.mkdir(exist_ok=True,parents=True);task.mkdir(exist_ok=True,parents=True)
    tables=dict(passengers=p, bookings=p.groupby('booking').size().rename('party_size').reset_index(),
                cabins=cabins.drop(columns='deck'), decks=pd.DataFrame(dict(deck_key=range(len(decks)),deck=decks)))
    spec=dict(passengers=dict(pkey='passenger',fkeys=dict(booking='bookings',cabin='cabins')),
              bookings=dict(pkey='booking'),cabins=dict(pkey='cabin',fkeys=dict(deck_key='decks')),decks=dict(pkey='deck_key'))
    if outer is not None:
        # Identical personal/aggregate fields in both branches. Planet edges let a
        # query reach training examples even after booking and cabin edges are cut.
        pp=tables['passengers'].copy()
        pp['party_size']=pp.groupby('booking').passenger.transform('size')
        pp['own_deck']=pp.cabin.map(cabins.set_index('cabin').deck).astype('string')
        pp['own_side']=pp.cabin.map(cabins.set_index('cabin').side).astype('string')
        pp['own_cabin_number']=pp.cabin.map(cabins.set_index('cabin').number)
        homes=sorted(pp.HomePlanet.dropna().unique())
        pp['home_key']=pp.HomePlanet.map({home: i for i,home in enumerate(homes)}).astype('Int64')
        tables['homes']=pd.DataFrame(dict(home_key=range(len(homes)),planet=homes))
        spec['homes']=dict(pkey='home_key')
        spec['passengers']['fkeys']['home_key']='homes'
        if mode=='no_edges':
            pp=pp.drop(columns=['booking','cabin'])
            spec['passengers']['fkeys']=dict(home_key='homes')
        tables['passengers']=pp
    for table_name,frame in tables.items():
        frame.to_parquet(db/(table_name+'.parquet'),index=False)
    (folder/'manifest.yaml').write_text(json.dumps(dict(name=name,tables=spec)))
    (task/'manifest.yaml').write_text(json.dumps(dict(name=TASK,kind='forecast',task_type='binary_classification',
               entity_table='passengers',entity_col='passenger',target_col='transported')))
    for tag,orders in zip(('train','test'),split):
        idx=np.flatnonzero(np.isin(g[loc],orders))
        # Query outcomes are kept out of every database and task table; zero is a placeholder.
        df=pd.DataFrame(dict(passenger=idx,transported=y[loc[idx]].astype(float) if tag=='train' else 0.))
        df.to_parquet(task/(tag+'.parquet'),index=False)
    return passenger.iloc[idx].to_numpy(str),y[loc[idx]],dict(tables={n:len(x) for n,x in tables.items()},
                    training_bookings=len(split[0]),query_bookings=len(split[1]),query_fold=outer if outer is not None else 0,database=name)


def use_sdpa(device):
    """The released module imports a compiled FlexAttention kernel. Apply the same
    query/key mask and scale with PyTorch's CPU scaled-dot-product attention.
    The Stanford module's projections, normalization, gates and forward stay intact.
    """
    import torch
    from rt.model import net

    def dense(q,k,v,block_mask,scale):
        b=torch.arange(q.size(0),device=q.device)[:,None,None,None]
        h=torch.zeros(1,1,1,1,dtype=torch.long,device=q.device)
        i=torch.arange(q.size(2),device=q.device)[None,None,:,None]
        j=torch.arange(k.size(2),device=q.device)[None,None,None,:]
        mask=block_mask[:,None] if isinstance(block_mask,torch.Tensor) else block_mask.mask_mod(b,h,i,j)
        has_key=mask.any(-1,keepdim=True)
        allowed=mask|(~has_key&(j==0))
        output=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allowed,scale=scale)
        return output.masked_fill(~has_key,0)
    from types import SimpleNamespace
    gen=torch.Generator().manual_seed(914)
    q,k,v=[torch.randn((2,3,9,8),generator=gen) for _ in range(3)]
    rule=lambda b,h,i,j: ((i+j+b)%3!=0)&(i>0)
    mask=rule(torch.arange(2)[:,None,None,None],None,torch.arange(9)[None,None,:,None],
              torch.arange(9)[None,None,None,:])
    expected=torch.nan_to_num(((q@k.transpose(-1,-2))/8).masked_fill(~mask,-torch.inf).softmax(-1))@v
    observed=dense(q,k,v,SimpleNamespace(mask_mod=rule),1/8)
    torch.testing.assert_close(observed,expected,rtol=1e-5,atol=1e-6)
    if device=='mps':
        # RT already materialized the exact Boolean masks before building the
        # sparse block-mask object; SDPA consumes that Boolean tensor directly.
        net._make_block_mask=lambda mask, **kwargs: mask
    runtime_q,runtime_k,runtime_v=[v0.to(device,torch.bfloat16) for v0 in (q,k,v)]
    runtime_mask=mask.squeeze(1).to(device)
    runtime_observed=dense(runtime_q,runtime_k,runtime_v,runtime_mask,1/8).float().cpu()
    r_q,r_k,r_v=[value.bfloat16().float() for value in (q,k,v)]
    runtime_ref=torch.nan_to_num(((r_q@r_k.transpose(-1,-2))/8).masked_fill(~mask,-torch.inf).softmax(-1))@r_v
    torch.testing.assert_close(runtime_observed,runtime_ref,rtol=.012,atol=.012)
    net.flex_attention=dense
    return dict(float32_cpu=float((observed-expected).abs().max()),runtime_bfloat16=float((runtime_observed-runtime_ref).abs().max()))


def run(ctx,device,outer=None,mode='edges'):
    import torch
    from rt import RelationalTransformer
    from rt.data import get_tasks
    from rt.eval import build_evaluator
    from rt.eval.relbench import _seed_offset
    from rt.preprocess import one
    kernel_error=use_sdpa(device)
    torch.set_num_threads(2)
    assert hashlib.sha256((OUT/'weights_pinned/model.safetensors').read_bytes()).hexdigest()==WEIGHTS
    src=json.loads((OUT/'sources.json').read_text())
    ids,y,meta=prepare(outer,mode)
    dbname=meta['database']
    bs=2 if outer is None else 8
    os.chdir(OUT)
    start=perf_counter()
    one(dataset=str(OUT/dbname),out_dir=str(OUT/'pre'),embedder='all-MiniLM-L12-v2',batch_size=64,
        skip_tasks=False,embed=True,upload_repo=None,public=False,revision=None)
    pre_time=perf_counter()-start
    model=RelationalTransformer.from_pretrained(str(OUT/'weights_pinned'),device=device).to(torch.bfloat16).eval()
    tasks=get_tasks(str(OUT/'pre'),[(dbname,TASK)],('test',))
    evaluator=build_evaluator(tasks,str(OUT/'pre'),embedder=model.config['embedder'],d_text=model.config['d_text'],
        device=device,ctx_size_list=[ctx],local_ctx_size=min(96,ctx),bfs_width=16,num_walks=512,walk_length=12,
        tokens_per_gpu=ctx*bs,items_per_task=len(ids)+bs,num_workers=0,prefetch_factor=None,prefer_latest=True,
        shuffle_seed=0,context_seed=0,mmap_populate=False,vector_db_path=None,db_cutoff=None)
    now=perf_counter()
    with torch.inference_mode():
        _,_,_,out,nlabels,nodes=next(iter(evaluator.evaluate_raw([(model,'')],[ctx],with_node_idxs=True)))
    offset=_seed_offset(str(OUT/'pre'),dbname,TASK,'test',model.config['embedder'])
    index=nodes-offset
    np.testing.assert_array_equal(np.sort(index),np.arange(len(ids)))
    logits=out[''][np.argsort(index)]
    print('held logits',logits.tolist()[:16], 'context labels',np.asarray(nlabels).tolist()[:16],flush=True)
    assert np.isfinite(logits).all() and np.std(logits)>0
    report=dict(**meta,rt_revision=src['rt']['revision'],model_sha256=WEIGHTS,
                embedder_revision=src['embedder']['revision'],rt_version=version('relational-transformer'),
                torch_version=torch.__version__,device=device,ctx=ctx,attention_backend='torch SDPA using official mask_mod and scale',
                kernel_check_max_error=kernel_error,preprocessing_seconds=pre_time,
                forward_and_sampling_seconds=perf_counter()-now,query_passenger_ids=ids.tolist(),
                in_context_training_label_counts=np.asarray(nlabels)[np.argsort(index)].tolist(),logits=logits.tolist(),
                correct=int(((logits>0)==y).sum()),query_rows=len(y),eval_labels_in_db=False)
    if outer is None:
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
        print(report,flush=True)
    else:
        fold_dir=OUT/'oof'
        fold_dir.mkdir(exist_ok=True)
        dest=fold_dir/f'{outer}_{mode}.npz'
        temp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(temp,ids=ids,y=y,logits=logits,index_order=np.sort(index),ctx=ctx,
                            train_bookings=meta['training_bookings'],context_labels=np.asarray(nlabels)[np.argsort(index)],
                            elapsed=report['forward_and_sampling_seconds'],weights=WEIGHTS,
                            torch_version=torch.__version__,rt_version=version('relational-transformer'))
        temp.replace(dest)
        print('fold result',outer,mode,len(y),report['correct'],report['forward_and_sampling_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ctx',type=int,default=512)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--fold',type=int,choices=range(8))
    parser.add_argument('--mode',choices=['edges','no_edges'],default='edges')
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    if args.prepare: print(prepare()[2])
    elif args.all:
        for k in range(8):
            for mode in ['edges','no_edges']:
                dest=OUT/'oof'/f'{k}_{mode}.npz'
                if dest.exists():
                    with np.load(dest,allow_pickle=False) as z:
                        assert int(z['ctx'])==args.ctx and str(z['weights'])==WEIGHTS
                    continue
                run(args.ctx,args.device,k,mode)
    else: run(args.ctx,args.device,args.fold,args.mode)
