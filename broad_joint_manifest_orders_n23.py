"""Predict the outcome at different positions in a probabilistic model of the whole passenger record."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT/'reports/nov23_release_vote_information'


def tokens(frame, y, train):
    fields, widths = [], []
    for col in frame:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(dtype=float)
            keep = np.isfinite(v[train])
            if col in BILLS:
                keep &= v[train]>0
            cuts = np.unique(np.quantile(v[train][keep], np.arange(1,12)/12))
            code = np.where(np.isfinite(v), 1+np.searchsorted(cuts,v), 0)
            if col in BILLS:
                code = np.where(v==0, len(cuts)+2, code)
            width = len(cuts)+2+(col in BILLS)
        else:
            levels = {a:i+1 for i,a in enumerate(sorted(set(s.iloc[train].dropna())))}
            code = s.map(levels).fillna(0).to_numpy(int)
            width = len(levels)+1
        fields.append(code)
        widths.append(width)
    return torch.tensor(np.column_stack([*fields,y]).astype('int64')), [*widths,2]


class Manifest(nn.Module):
    def __init__(self, widths):
        super().__init__()
        n = len(widths)
        self.mask_id = max(widths)
        self.emb = nn.Embedding(n*(self.mask_id+1), 10)
        self.register_buffer('offset', torch.arange(n)*(self.mask_id+1))
        self.register_buffer('invalid', torch.arange(self.mask_id)[None,:]>=torch.tensor(widths)[:,None])
        self.net = nn.Sequential(nn.Linear(n*10,128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Linear(128,128), nn.GELU(), nn.Linear(128,n*self.mask_id))

    def forward(self, observed):
        em = self.emb(observed+self.offset)
        scores = self.net(em.flatten(1)).reshape(len(em),len(self.offset),self.mask_id)
        return scores.masked_fill(self.invalid,-1e6)


def fit(rows, widths, joint):
    torch.manual_seed(711)
    m = Manifest(widths)
    opt = torch.optim.AdamW(m.parameters(),lr=.0014,weight_decay=.01)
    for _ in range(42):
        for ids in torch.randperm(len(rows)).split(256):
            z=rows[ids]
            blank=torch.zeros_like(z,dtype=torch.bool)
            blank[:,-1]=True
            if joint:
                use = torch.rand(len(ids)) < .75
                rate = torch.rand((len(ids),1))*.96+.02
                blank[use] = (torch.rand(z.shape) < rate)[use]
                blank[use,torch.randint(z.shape[1],(int(use.sum()),))] = True
            scores=m(z.masked_fill(blank,m.mask_id))
            loss=nn.functional.cross_entropy(scores[blank],z[blank])
            opt.zero_grad();loss.backward();opt.step()
    return m.eval()


@torch.no_grad()
def readouts(m, records):
    # Never provide a query outcome to either conditional likelihood. Evaluate
    # two explicit hypothetical outcomes for the same observed manifest instead.
    query=records.clone()
    query[:,-1]=m.mask_id
    n,d=query.shape
    direct=m(query)[:,-1,:2].softmax(-1)[:,1].numpy()
    result={'direct':direct}
    rng=np.random.default_rng(913)
    orders=[rng.permutation(d-1) for _ in range(6)]
    for cut in (0,(d-1)//2):
        panel=[]
        for order in orders:
            start=torch.full_like(query,m.mask_id)
            start[:,order[:cut]]=query[:,order[:cut]]
            prior=m(start)[:,-1,:2]
            odds=prior[:,1]-prior[:,0]
            # Earlier feature probabilities cancel because the label was hidden.
            a=torch.cat([start,start])
            a[:n,-1]=0
            a[n:,-1]=1
            for j in order[cut:]:
                ll=m(a)[:,j].log_softmax(-1).gather(1,query[:,j:j+1].repeat(2,1)).flatten()
                odds=odds+ll[n:]-ll[:n]
                a[:,j]=query[:,j].repeat(2)
            panel.append(torch.sigmoid(odds).numpy())
        result['first' if cut==0 else 'middle']=np.mean(panel,axis=0)
    result['ordered_mean']=np.mean(list(result.values()),axis=0)
    return result


def run():
    row,y,frame,g,_=load()
    out=ART/'nov23_joint_manifest_orders'
    out.mkdir(exist_ok=True)
    plans,hist,commons={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,opinions,_=read_opinions(seed,row,y)
        pred={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            z,widths=tokens(frame,y,t)
            conditional=fit(z[t],widths,False)
            test=z[q].clone()
            test[:,-1]=conditional.mask_id
            with torch.no_grad():
                direct=conditional(test)[:,-1,:2].softmax(-1)[:,1].numpy()
            joint=fit(z[t],widths,True)
            bank=readouts(joint,test)
            bank['discriminative']=direct
            for k,p in bank.items():
                pred.setdefault(k,np.full(len(y),np.nan))[q]=p
        votes={'reference':ref}
        for k,p in pred.items():
            assert np.isfinite(p).all()
            votes[k]=p>=.5
            votes['quarter_'+k]=np.where(ref==((opinions[:,0])>=.5),(.75*opinions[:,0]+.25*p)>=.5,ref)
        h={k:v==y for k,v in votes.items()}
        if seed!='v131':
            commons.append(((opinions>=.5)!=y[:,None]).all(1))
            for k,v in h.items():
                hist.setdefault(k,[]).append(v)
        plans[seed]={k:paired(v,ref==y) for k,v in h.items()}
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=opinions[:,0],ref=ref,**pred)
        print(seed,plans[seed]['ordered_mean'],plans[seed]['quarter_ordered_mean'],flush=True)
    hits={k:np.stack(v) for k,v in hist.items()}
    common=np.stack(commons).all(0)
    assert common.sum()==332
    misses=(~hits['reference']).sum(0)
    stats={k:dict(common=int((v.all(0)&common).sum()),by_baseline_errors=[
            int((v.all(0)&(misses==i)).sum()) for i in range(4)], net_ci95_pp=interval(
            (v.astype(float)-hits['reference']).mean(0),g)) for k,v in hits.items()}
    write(REPORT/'manifest_orders.json',dict(plans=plans,grouped=stats))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):run()
