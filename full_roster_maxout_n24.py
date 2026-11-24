"""Train a difference of max-affine scores on raw fields and test it on both validation designs."""
import json
from math import ceil

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities

OUT=ROOT/'artifacts/nov24_maxout'


def fields(raw):
    x=passenger_vectors(raw)
    cost=raw[BILLS].fillna(0)
    all_cost=cost.sum(axis=1)
    group=raw.PassengerId.str.split('_').str[0]
    size=group.map(group.value_counts()).to_numpy()
    # Keep individual service charges and explicitly represent bill composition.
    return np.column_stack([x,np.log1p(size),np.log1p(all_cost)/6,
                            cost.to_numpy()/np.maximum(all_cost.to_numpy()[:,None],1)])


class Classifier(nn.Module):
    def __init__(self,width,mode):
        super().__init__()
        self.mode=mode
        self.skip=nn.Linear(width,1)
        if mode!='linear':
            self.maps=nn.Linear(width,32)
        if mode=='relu':
            self.readout=nn.Linear(32,1,bias=False)

    def forward(self,x):
        value=self.skip(x).squeeze(1)
        if self.mode=='linear':
            return value
        scores=self.maps(x)
        if self.mode=='relu':
            return value+self.readout(F.relu(scores)).squeeze(1)
        # Each of the eight groups selects one of four learned field combinations.
        pieces=scores.reshape(len(x),8,4).max(dim=2).values
        return value+(pieces[:,:4].sum(1)-pieces[:,4:].sum(1))/2


def fit(x,y,t,q,mode,seed):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    center=x[t].mean(0)
    scale=x[t].std(0).clip(.25,None)
    a=torch.from_numpy(((x[t]-center)/scale).astype('float32'))
    b=torch.from_numpy(((x[q]-center)/scale).astype('float32'))
    truth=torch.tensor(y[t],dtype=torch.float32)
    model=Classifier(x.shape[1],mode)
    opt=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.025)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70,eta_min=.0004)
    snapshots=[]
    for epoch in range(70):
        for ids in np.array_split(rng.permutation(len(t)),ceil(len(t)/384)):
            score=model(a[ids])
            loss=F.binary_cross_entropy_with_logits(score,truth[ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
        schedule.step()
        if epoch in (49,59,69):
            with torch.inference_mode():
                snapshots.append(model(b).numpy())
    return expit(np.mean(snapshots,axis=0))


def scores(x,y,groups,rows,fold,tag,mode):
    OUT.mkdir(exist_ok=True)
    predicted=np.empty(len(rows))
    for f in np.unique(fold):
        q=rows[fold==f]
        t=rows[(fold!=f)&~np.isin(groups[rows],groups[q])]
        assert not np.isin(groups[t],groups[q]).any()
        path=OUT/f'{mode}_{tag}_{f}.npz'
        if not path.exists():
            pair=np.column_stack([fit(x,y,t,q,mode,seed) for seed in [403,1742]])
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=t,q=q,y=y[t],pair=pair)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,v in [('t',t),('q',q),('y',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            predicted[fold==f]=z['pair'].mean(1)
    assert np.isfinite(predicted).all()
    return predicted


def run():
    _,y,g,rows,oldfold,_,champion,_,_,_,_,old,_=aligned()
    raw,all_y,groups,fold=data()
    names,half,full=banks(raw,all_y,fold)
    base=latest_reference(all_y,groups,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,all_y,groups,fold)
    np.testing.assert_array_equal(gate>=.5,base)
    x=fields(raw)
    extra=~np.isin(np.arange(len(all_y)),rows)
    cold=extra&~np.isin(groups,g)
    shared=((full>=.5)!=all_y[:,None]).all(1)
    hard=shared[rows]&(champion!=y)
    report={}
    for mode in ['linear','maxout','relu']:
        historical=scores(x,all_y,groups,rows,oldfold,'historical',mode)
        current=scores(x,all_y,groups,np.arange(len(raw)),fold,'eightfold',mode)
        out={}
        for policy,weight in [('standalone',1.),('quarter',.25),('half',.5)]:
            if policy=='standalone':
                a,b=historical>=.5,current>=.5
            else:
                a=np.where(champion==(old[:,0]>=.5),((1-weight)*old[:,0]+weight*historical)>=.5,champion)
                b=((1-weight)*gate+weight*current)>=.5
            def describe(p,ref,truth,group,split,mask):
                d=((p==truth).astype(int)-(ref==truth))[mask]
                return dict(rows=int(mask.sum()),accuracy=float((p[mask]==truth[mask]).mean()),
                           fixed=int((d==1).sum()),harmed=int((d==-1).sum()),ci_pp=interval(d,group[mask]),
                           fold_net=[int(d[split[mask]==f].sum()) for f in np.unique(split[mask])])
            out[policy]=dict(historical=describe(a,champion,y,g,oldfold,np.ones(len(y),bool)),
                eightfold=describe(b,base,all_y,groups,fold,np.ones(len(raw),bool)),
                extra_new_orders=describe(b,base,all_y,groups,fold,cold),
                extra=describe(b,base,all_y,groups,fold,extra),
                shared_fixed=int((shared&(b==all_y)).sum()),
                shared_twice_fixed=int((hard&(a==y)&(b[rows]==y)).sum()),
                harmed_twice=int(((champion==y)&(base[rows]==y)&(a!=y)&(b[rows]!=y)).sum()))
        report[mode]=out
        print(mode,[(k,[(v[c]['fixed'],v[c]['harmed']) for c in ['historical','eightfold','extra_new_orders']])
                    for k,v in out.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/maxout.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
