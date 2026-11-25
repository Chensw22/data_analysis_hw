"""Run train with .venv-neural, report with .venv. Routing derives from arXiv:1710.09829."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT, OUT, data

CACHE = ROOT/'artifacts/nov26_booking_capsules'
REPORT = ROOT/'reports/nov23_matched_learning_curve/booking_capsule_routing.json'
SEEDS = (4537,8912)
BILLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
EPOCHS = 28


def blocks(raw,g,t):
    def categories(cols):
        parts = []
        for c in cols:
            strings = c.astype('string').fillna('?')
            levels = sorted(strings.iloc[t].unique())
            parts.extend(strings.eq(v).to_numpy(float) for v in levels)
        return np.column_stack(parts)
    def numeric(series):
        arr = series.to_numpy(float)
        available = np.isfinite(arr)
        knots = np.unique(np.quantile(arr[t][available[t]],np.linspace(0,1,11)))
        return np.column_stack([~available, *(np.maximum(0,1-np.abs(np.interp(
            np.nan_to_num(arr,nan=knots[0]),knots,np.arange(len(knots)))-j))*available for j in range(len(knots)))])
    cabin = raw.Cabin.str.split('/',expand=True)
    reg = np.column_stack([categories([raw.HomePlanet,raw.Destination,raw.VIP,cabin[0],cabin[2]]),
                           numeric(pd.to_numeric(cabin[1]))])
    bio = np.column_stack([categories([raw.CryoSleep]),numeric(raw.Age)])
    amounts = raw[BILLS]
    costs = np.column_stack([*(numeric(np.log1p(amounts[c])) for c in BILLS),
                             numeric(np.log1p(amounts.sum(1,min_count=1))),amounts.gt(0)])
    own = np.column_stack([categories([raw.HomePlanet,raw.Destination,raw.CryoSleep,cabin[0],cabin[2]]),
                           raw.Age.fillna(0)/50,raw.Age.isna(),np.log1p(amounts.fillna(0))/8,amounts.isna()])
    roster = pd.DataFrame(own).groupby(g)
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    peer = (roster.transform('sum').to_numpy()-own)/np.maximum(1,size[:,None]-1)
    party = np.column_stack([np.log(size),size==1,peer,peer-own])
    return [v.astype('float32') for v in (reg,bio,costs,party)]


def fit(block,y,t,q,seed,steps):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.votes = nn.ModuleList(nn.Sequential(nn.Linear(x.shape[1],32),nn.SiLU(),
                         nn.Linear(32,2*2*8)) for x in block)
            self.bias = nn.Parameter(torch.zeros(2,8))

        def forward(self,x):
            votes = torch.cat([enc(part).reshape(-1,2,2,8) for enc,part in zip(self.votes,x)],dim=1)
            logits = votes.new_zeros(votes.shape[:3])
            for k in range(steps):
                weights = logits.softmax(-1)
                sums = (weights.unsqueeze(-1)*votes).sum(1)+self.bias
                square = sums.square().sum(-1,keepdim=True)
                outputs = sums*(square+1e-8).sqrt()/(1+square)
                if k+1<steps:
                    logits = logits+(votes*outputs.unsqueeze(1)).sum(-1).detach()
            return 9*(outputs.norm(dim=-1)[:,1]-outputs.norm(dim=-1)[:,0])
    torch.manual_seed(seed)
    model = Net()
    opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.005)
    target = torch.tensor(y,dtype=torch.float32)
    xs = [torch.from_numpy(z) for z in block]
    order = np.random.default_rng(seed)
    scores = []
    for epoch in range(EPOCHS):
        for ids in np.array_split(order.permutation(t),int(np.ceil(len(t)/512))):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model([x[ids] for x in xs]),target[ids])
            loss.backward()
            opt.step()
        if epoch in (19,23,27):
            with torch.no_grad():
                scores.append(model([x[q] for x in xs]).sigmoid().numpy())
    return np.mean(scores,axis=0)


def train():
    import torch
    import hashlib
    torch.set_num_threads(2)
    raw,y,g,f = data()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    CACHE.mkdir(exist_ok=True)
    for k in range(8):
        t,q = np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        x = blocks(raw,g,t)
        for seed in SEEDS:
            file = CACHE/f'fold{k}_{seed}.npz'
            if file.exists():
                with np.load(file) as z:
                    if str(z['source'])==fingerprint:
                        continue
            fixed,dynamic = [fit(x,y,t,q,seed,s) for s in (1,3)]
            np.savez_compressed(file,source=fingerprint,q=q,t=t,labels=y[t],ids=raw.PassengerId.iloc[q].to_numpy(str),
                                fixed=fixed,dynamic=dynamic)
            print(int(k),seed,round(((dynamic>=.5)==y[q]).mean(),3),flush=True)


def report():
    from exhaustive_family_quorums_n26 import inputs
    from scipy.special import logit
    raw,y,g,f = data()
    _,cur,_ = inputs()
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        prior = z['p']
    preds = np.empty((len(y),2,len(SEEDS)))
    for k in range(8):
        t,q = np.flatnonzero(f!=k),np.flatnonzero(f==k)
        for j,seed in enumerate(SEEDS):
            with np.load(CACHE/f'fold{k}_{seed}.npz') as z:
                for key,expected in [('t',t),('q',q),('labels',y[t]),('ids',raw.PassengerId.iloc[q].to_numpy(str))]:
                    np.testing.assert_array_equal(z[key],expected)
                preds[q,:,j] = np.column_stack([z['fixed'],z['dynamic']])
    result = dict(epochs=EPOCHS,seeds=list(SEEDS),source='https://arxiv.org/abs/1710.09829',
                  reference='Mitra2 adapted; 3:1 reference/capsule log odds',runs={})
    anchor = cur['parent']
    def score(choose,mask):
        fixes = mask&choose&(anchor!=y)
        hurts = mask&choose&(anchor==y)
        return dict(n=int(mask.sum()),fixed=int(fixes.sum()),harmed=int(hurts.sum()),net=int(fixes.sum()-hurts.sum()))
    for name,p in [('mean',preds.mean(axis=2)),*[(str(s),preds[:,:,j]) for j,s in enumerate(SEEDS)]]:
        fixed,dyn = p.T
        vote,plain = dyn>=.5,fixed>=.5
        baseline = prior>=.5
        blend = 3*logit(prior.clip(.001,.999))+logit(dyn.clip(.001,.999))>=0
        ctrl = 3*logit(prior.clip(.001,.999))+logit(fixed.clip(.001,.999))>=0
        actions = dict(dynamic=vote!=anchor,fixed=plain!=anchor,dynamic_only=(vote!=anchor)&(plain==anchor),
                       blend=blend!=anchor,blend_extra=(blend!=anchor)&(baseline==anchor),
                       blend_routing_only=(blend!=anchor)&(baseline==anchor)&(ctrl==anchor))
        result['runs'][name] = dict(accuracy=dict(dynamic=float((vote==y).mean()),fixed=float((plain==y).mean())),
            scopes={s:{k:score(a,mask) for k,a in actions.items()} for s,mask in [
            ('eight_fold',cur['known']),('eleven_agree',cur['common']),('new_orders',cur['known']&cur['new'])]})
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['runs']['mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('train','report'))
    train() if parser.parse_args().action=='train' else report()
