"""Train a residual network to distinguish actual teacher errors from similar-confidence correct bookings."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logit

from full_roster_booking_validation_n23 import ROOT, data

SOURCE=ROOT/'artifacts/nov25_grouped_peer_teacher'
OUT=ROOT/'artifacts/nov25_matched_error_ranker'
REPORT=ROOT/'reports/nov23_matched_learning_curve/matched_error_ranker.json'
MODES=('likelihood','matched','random_order')


def covariates(raw, group, t, q, k):
    from passenger_joint_likelihood_n25 import encode
    from within_bin_bill_precision_n25 import BILLS, coordinates
    enc, cards=encode(raw,t)
    pos,_=coordinates(raw,enc,t,q,k)
    categorical=[np.eye(width,dtype=np.float32)[enc[:,j]] for j,width in enumerate(cards[:-1])]
    n=raw.PassengerId.groupby(group).transform('size')
    party=[np.log(n),pd.to_numeric(raw.PassengerId.str.split('_').str[1])/n]
    for col in BILLS:
        values=np.log1p(raw[col])
        others=values.notna().groupby(group).transform('sum')-values.notna().astype(int)
        amount=values.groupby(group).transform('sum')-values.fillna(0)
        party.extend([amount.div(others.replace(0,np.nan)).fillna(0),others.gt(0)])
    return np.column_stack([*categorical,pos.numpy()[:,8:13],*party]).astype('float32')


def train_predict(matrix, raw_p, query_p, labels, group, mode, seed):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    def features(block,p):
        return torch.as_tensor(np.column_stack([block,p>=.5,np.abs(logit(p.clip(.002,.998)))]),dtype=torch.float32)
    n=len(labels)
    x=features(matrix[:n],raw_p)
    test=features(matrix[n:],query_p)
    ref=raw_p>=.5
    target=ref!=labels
    anchor=-np.abs(logit(raw_p.clip(.002,.998)))
    base=torch.as_tensor(anchor,dtype=torch.float32)
    z=torch.as_tensor(target,dtype=torch.float32)
    cells=[]
    for side in (False,True):
        loc=np.flatnonzero(ref==side)
        decile=np.searchsorted(np.quantile(anchor[loc],np.arange(1,10)/10),anchor[loc])
        for band in np.unique(decile):
            members=loc[decile==band]
            wrong,right=members[target[members]],members[~target[members]]
            if len(wrong) and len(right):
                cells.append((wrong,right))
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,24),
                        nn.SiLU(),nn.Linear(24,1))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.02)
    for epoch in range(12):
        error_ids,correct_ids=[],[]
        for wrong,right in cells:
            # Equal numbers per confidence band prevent common borderline cases from dominating this loss.
            a=rng.choice(wrong,160)
            b=rng.choice(right,160)
            different=group[a]!=group[b]
            error_ids.extend(a[different]);correct_ids.extend(b[different])
        a,b=np.asarray(error_ids),np.asarray(correct_ids)
        order=rng.permutation(len(a))
        swaps=rng.random(len(a))<.5
        if mode=='random_order':
            a,b=np.where(swaps,b,a),np.where(swaps,a,b)
        population=rng.permutation(n)
        for batch,pairs in zip(np.array_split(population,24),np.array_split(order,24),strict=True):
            dev=model(x[batch]).ravel()
            loss=F.binary_cross_entropy_with_logits(base[batch]+dev,z[batch])
            if mode!='likelihood':
                diff=model(x[a[pairs]]).ravel()-model(x[b[pairs]]).ravel()
                loss=loss+.25*F.softplus(-diff).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.inference_mode():
        return model(test).ravel().numpy()


def generate():
    import torch
    torch.set_num_threads(2)
    raw,y,g,f=data()
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        with np.load(SOURCE/f'outer{k}.npz') as z:
            for key,expected in [('t',t),('q',q),('labels',y[t]),('passenger',raw.PassengerId.iloc[t])]:
                np.testing.assert_array_equal(z[key],expected)
            tr=z['p']
        with np.load(SOURCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            p=z['native']
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'{k}.npz'
        if not path.exists():
            matrix=covariates(raw,g,t,q,k)[np.r_[t,q]]
            pred={name:train_predict(matrix,tr,p,y[t],g[t],name,8073+k) for name in MODES}
            assert all(np.isfinite(v).all() and len(v)==len(q) for v in pred.values())
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream,t=t,q=q,labels=y[t],native=p,**pred)
            tmp.replace(path)
        print('paired training', k, flush=True)


def evaluate():
    from adjudicate_amount_dissent_n25 import counts
    from disjoint_tree_fields_n25 import ART
    from full_roster_six_failure_archive_n24 import prediction_sets
    from grouped_peer_release_replay_n25 import edit_effect_bounds
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    votes,old=[(np.column_stack(x)>=.5).sum(1) for x in panels]
    common=(votes==0)|(votes==68)
    repeated=np.zeros(len(y),bool)
    repeated[rows]=common[rows]&((old==0)|(old==68))&((old>34)==(votes[rows]>34))
    new=~np.isin(g,g[rows])
    results={}
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        with np.load(SOURCE/f'release{k}.npz') as z:
            p,prior,lo,hi=[z[n] for n in ('native','v131','nine_on_low_branch','nine_on_high_branch')]
            np.testing.assert_array_equal(z['q'],q)
        ref=p>=.5
        with np.load(OUT/f'{k}.npz') as z:
            for key,val in [('q',q),('t',t),('labels',y[t]),('native',p)]:
                np.testing.assert_array_equal(z[key],val)
            for mode in MODES:
                # A zero residual preserves the original label, including exact ties.
                for scale in (.5,1.0):
                    accept=scale*z[mode]>np.abs(logit(p.clip(.002,.998)))
                    a,b=edit_effect_bounds(ref^accept,ref,prior,lo,hi,y[q])
                    known=prior>=0
                    np.testing.assert_array_equal(a[known],b[known])
                    results.setdefault(mode+'/'+str(scale),[]).append((q,known,a,b))
    report={}
    for name,parts in results.items():
        ix,known,a,b=(np.concatenate([v[j] for v in parts]) for j in range(4))
        np.testing.assert_array_equal(np.sort(ix),np.arange(len(y)))
        r=dict(full_bounds=[int(a.sum()),int(b.sum())])
        for key,mask in [('known',known),('new',known&new[ix]),('unanimous',known&common[ix]),
                         ('common_error',known&common[ix]&((votes[ix]>34)!=y[ix])),('repeated',known&repeated[ix])]:
            r[key]=counts(a,mask,f[ix])
        report[name]=r
        print(name,r['full_bounds'],[(s,r[s]['fixed'],r[s]['harmed']) for s in ('known','new','unanimous')])
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['generate','evaluate'])
    generate() if parser.parse_args().action=='generate' else evaluate()
