"""Compare shared and partner-specific field embeddings on the full passenger population."""
from itertools import combinations
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from broad_bill_completion_states_n23 import REPORT
from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from exaone_validation_data import CATEGORICAL
from manifest_tensor_outcomes import hat


def encode(frame, train):
    blocks=[]
    for field in CATEGORICAL:
        known=np.unique(frame[field].iloc[train])
        blocks.append(np.column_stack([(frame[field]==v).to_numpy() for v in known]+
                                      [(~frame[field].isin(known)).to_numpy()]))
    for field,knots in [('Age',[0,5,12,17,25,40,60,80]),('CabinNumber',[0,250,500,750,1000,1250,1500,2000]),
                        ('BookingSize',[1,2,3,4,6,8])]:
        blocks.append(hat(frame[field],knots))
    for field in BILLS:
        blocks.append(hat(np.log1p(frame[field]),np.log1p([0,1,100,500,850,1400,3000,10000,40000])))
    width=max(v.shape[1] for v in blocks)
    x=np.stack([np.pad(v,((0,0),(0,width-v.shape[1]))) for v in blocks],axis=1)
    np.testing.assert_allclose(x.sum(2),1.)
    return torch.from_numpy(x.astype(np.float32))


class FieldPairs(nn.Module):
    def __init__(self,fields,width,partner_specific):
        super().__init__()
        self.linear=nn.Parameter(torch.zeros(fields,width))
        self.bias=nn.Parameter(torch.zeros(()))
        self.shared=nn.Parameter(.06*torch.randn(fields,width,4))
        self.specific=nn.Parameter(torch.zeros(fields,width,fields,4)) if partner_specific else None
        left,right=zip(*combinations(range(fields),2))
        self.register_buffer('left',torch.tensor(left))
        self.register_buffer('right',torch.tensor(right))

    def forward(self,x):
        linear=torch.einsum('bfw,fw->b',x,self.linear)
        z=torch.einsum('bfw,fwr->bfr',x,self.shared)
        a,b=z[:,self.left],z[:,self.right]
        if self.specific is not None:
            effect=torch.einsum('bfw,fwgr->bfgr',x,self.specific)
            a=a+effect[:,self.left,self.right]
            b=b+effect[:,self.right,self.left]
        return self.bias+linear+.5*(a*b).sum((1,2))


def train_pair(x,y,t,q,mode,seed=604):
    torch.manual_seed(seed)
    learner=FieldPairs(*x.shape[1:],mode)
    optimizer=torch.optim.AdamW(learner.parameters(),lr=.009,weight_decay=.05)
    rng=np.random.default_rng(seed)
    result=[]
    for epoch in range(24):
        for idx in np.array_split(rng.permutation(t),int(np.ceil(len(t)/384))):
            optimizer.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(learner(x[idx]),y[idx])
            if mode:
                loss=loss+.0006*learner.specific.square().sum()/len(learner.left)
            loss.backward()
            nn.utils.clip_grad_norm_(learner.parameters(),2.)
            optimizer.step()
        if epoch in (15,19,23):
            with torch.no_grad():
                result.append(torch.cat([learner(x[part]).sigmoid() for part in np.array_split(q,6)]).numpy())
    return np.mean(result,axis=0)


def run_fold(frame,groups,y,t,q):
    assert not np.intersect1d(groups[t],groups[q]).size
    x=encode(frame,t)
    target=torch.from_numpy(y.astype(np.float32))
    p={name:train_pair(x,target,t,q,mode) for name,mode in [('shared',False),('partner_specific',True)]}
    return p,dict(train=len(t),query=len(q),fields=x.shape[1])


if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(2):
        evaluate('partner_specific_fields',run_fold,REPORT)
