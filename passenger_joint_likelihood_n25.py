"""Estimate transport odds from a missing-field network and from its class-conditional manifest likelihoods."""
import argparse
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov25_passenger_joint_likelihood'
BLOCKS={'registration':(0,1,2,3),'sleep':(4,),'cabin':(5,6,7),'bills':(8,9,10,11,12)}
ORDERS=[(*BLOCKS['registration'],*BLOCKS['sleep'],*BLOCKS['cabin'],*BLOCKS['bills']),
        (*reversed(BLOCKS['bills']),*reversed(BLOCKS['cabin']),4,*reversed(BLOCKS['registration']))]
CATEGORY={0,1,2,4,5,6}
TARGET=13


def encode(raw,t):
    cabin=raw.Cabin.str.split('/',expand=True)
    fields=[raw.HomePlanet,raw.Destination,raw.VIP,raw.Age,raw.CryoSleep,cabin[0],cabin[2],
            pd.to_numeric(cabin[1]),*[raw[k] for k in ('RoomService','FoodCourt','ShoppingMall','Spa','VRDeck')]]
    encoded=[]
    cards=[]
    for j,s in enumerate(fields):
        if j in CATEGORY:
            text=s.astype('string').fillna('?')
            levels=sorted(text.iloc[t].unique())
            index=pd.Categorical(text,categories=levels).codes.astype(int)+1
            card=len(levels)+1
        else:
            v=s.to_numpy(float)
            positive=v[t][np.isfinite(v[t])&(v[t]>0)]
            cuts=np.unique(np.quantile(positive,np.arange(1,16)/16))
            index=np.where(np.isnan(v),0,np.where(v==0,1,2+np.searchsorted(cuts,v)))
            card=3+len(cuts)
        encoded.append(index)
        cards.append(card)
    return np.column_stack(encoded),np.array([*cards,2])


class JointTable(nn.Module):
    def __init__(self,cards):
        super().__init__()
        self.n=len(cards)
        self.width=int(max(cards))
        self.register_buffer('offset',torch.as_tensor(np.r_[0,np.cumsum(cards+1)[:-1]]))
        self.register_buffer('mask_id',torch.as_tensor(cards))
        self.register_buffer('invalid',torch.arange(self.width)[None,:]>=torch.as_tensor(cards)[:,None])
        self.embedding=nn.Embedding(int(sum(cards+1)),10)
        self.net=nn.Sequential(nn.Linear(self.n*10,144),nn.LayerNorm(144),nn.SiLU(),nn.Dropout(.12),
                               nn.Linear(144,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),
                               nn.Linear(96,self.n*self.width))

    def forward(self,values,observed,position=None):
        ids=torch.where(observed,values,self.mask_id)+self.offset
        embedding=self.embedding(ids)
        if position is not None:
            # Only positive bill bins receive a nonzero within-bin coordinate.
            # Interpolate along the direction between adjacent positive-bin embeddings.
            upper=torch.minimum(values+1,self.mask_id-1)+self.offset
            lower=torch.maximum(values-1,torch.minimum(self.mask_id-1,torch.full_like(values,2)))+self.offset
            slope=(self.embedding(upper)-self.embedding(lower))*.5
            embedding=embedding+torch.where(observed,position,0.)[:,:,None]*slope
        logits=self.net(embedding.flatten(1)).reshape(-1,self.n,self.width)
        return logits.masked_fill(self.invalid,-1e6)


def train_model(x,cards,t,seed,joint,position=None):
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed+1)
    model=JointTable(cards)
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.012)
    for _ in range(36):
        model.train()
        for idx in torch.randperm(len(t),generator=rng).split(256):
            rows=t[idx.numpy()]
            v=x[rows]
            within=None if position is None else position[rows]
            full=torch.ones_like(v,dtype=torch.bool)
            full[:,TARGET]=False
            logits=model(v,full,within)[:,TARGET,:2]
            loss=F.cross_entropy(logits,v[:,TARGET])
            if joint:
                # For a random fraction of observed fields, predict all remaining fields.
                # Make class-conditioned and class-hidden contexts equally frequent.
                prob=torch.rand((len(v),1),generator=rng)
                observed=torch.rand(v.shape,generator=rng)<prob
                observed[:,TARGET]=torch.rand(len(v),generator=rng)<.5
                scores=model(v,observed,within)
                hidden=~observed
                partial=F.cross_entropy(scores.transpose(1,2),v,reduction='none')
                w=hidden.to(partial.dtype)
                loss=loss+(partial*w).sum()/w.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.inference_mode()
def predict(model,x,base_rate,joint):
    # Labels on input queries are dummy values; only the two explicitly assigned hypotheses enter scoring.
    n=len(x)
    observed=torch.ones_like(x,dtype=torch.bool)
    observed[:,TARGET]=False
    direct=model(x,observed)[:,TARGET,:2].softmax(1)[:,1]
    result={'direct':direct.numpy()}
    if not joint:
        return result
    for name,columns in BLOCKS.items():
        present=observed.clone()
        present[:,list(columns)]=False
        p=model(x,present)[:,TARGET,:2].softmax(1)[:,1]
        result['omit_'+name]=p.numpy()
        work=x.repeat(2,1)
        work[:,TARGET]=torch.arange(2).repeat_interleave(n)
        seen=present.repeat(2,1)
        seen[:,TARGET]=True
        lr=torch.zeros(n)
        for j in columns:
            lp=model(work,seen)[:,j].log_softmax(1).gather(1,work[:,j,None]).ravel()
            lr+=lp[n:]-lp[:n]
            seen[:,j]=True
        result['restore_'+name]=(p.clamp(.0001,.9999).logit()+lr).sigmoid().numpy()
    for i,order in enumerate(ORDERS):
        work=x.repeat(2,1)
        work[:,TARGET]=torch.arange(2).repeat_interleave(n)
        seen=torch.zeros_like(work,dtype=torch.bool)
        seen[:,TARGET]=True
        ll=torch.zeros(2*n)
        for j in order:
            ll+=model(work,seen)[:,j].log_softmax(1).gather(1,work[:,j,None]).ravel()
            seen[:,j]=True
        result['chain_'+str(i)]=(torch.tensor(float(base_rate)).logit()+ll[n:]-ll[:n]).sigmoid().numpy()
    return result


def run(historical=False):
    torch.set_num_threads(2)
    raw,y,g,fold=data()
    OUT.mkdir(exist_ok=True)
    rows=np.arange(len(y))
    if historical:
        with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
            rows,fold=z['rows'],z['fold']
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows])
    for k in map(int,np.unique(fold)):
        q=rows[fold==k]
        t=rows[(fold!=k)&~np.isin(g[rows],g[q])]
        assert not np.isin(g[t],g[q]).any()
        encoded,cards=encode(raw,t)
        inputs=torch.as_tensor(np.column_stack([encoded,y.astype(int)]))
        query=inputs[q].clone()
        query[:,TARGET]=0
        digest=hashlib.sha256(encoded.tobytes()).hexdigest()
        prefix='historical' if historical else 'fold'
        path=OUT/f'{prefix}{k}.npz'
        if not path.exists():
            estimates={}
            for joint in (False,True):
                model=train_model(inputs,cards,t,1241+k,joint)
                predictions=predict(model,query,y[t].mean(),joint)
                for name,p in predictions.items():
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    estimates[name if joint else 'outcome_only']=p
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,q=q,t=t,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                                    feature_digest=digest,**estimates)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for key,val in [('t',t),('q',q),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q]),('feature_digest',digest)]:
                np.testing.assert_array_equal(z[key],val)
        print('completed fold',k,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--historical',action='store_true')
    run(parser.parse_args().historical)
