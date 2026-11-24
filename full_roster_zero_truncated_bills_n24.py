"""Compare independent service use with a zero-truncated five-service spending model."""
import json
from math import ceil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS,ROOT,banks,data,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities


def inputs(raw):
    cabin=raw.Cabin.str.split('/',expand=True)
    ctx=raw[['HomePlanet','CryoSleep','VIP','Destination']].astype('string').assign(deck=cabin[0],side=cabin[2])
    numeric=pd.to_numeric(cabin[1],errors='coerce')
    group=raw.PassengerId.str.split('_').str[0]
    x=np.column_stack([pd.get_dummies(ctx.fillna('missing'),dtype=float),raw.Age.fillna(30)/35,
                       raw.Age.isna(),numeric.fillna(800)/1000,numeric.isna(),np.log1p(group.map(group.value_counts()))])
    return x,raw[BILLS].to_numpy(float),raw.CryoSleep.eq(True).to_numpy()


def pattern_log_likelihood(logits, active_logit, observed, paid, mode, asleep):
    # Final axis indexes the five services. Integrate all possible unobserved service flags.
    log_pattern=(torch.where(paid,F.logsigmoid(logits),F.logsigmoid(-logits))*observed).sum(-1)
    if mode=='independent':
        return log_pattern
    log_zero=F.logsigmoid(-logits).sum(-1)
    # Given at least one service, conditioning on a positive observed bill
    # excludes the all-zero pattern. An all-zero observed prefix still allows
    # a positive bill in any of its missing entries.
    log_nonzero=torch.log((-torch.expm1(log_zero)).clamp_min(1e-12))
    any_positive=paid.any(-1)
    log_missing_zero=(F.logsigmoid(-logits)*(~observed)).sum(-1)
    possible_missing=(~observed).any(-1)
    log_positive_missing=torch.log((-torch.expm1(log_missing_zero)).clamp_min(1e-12))
    conditional=log_pattern-log_nonzero+torch.where(any_positive,0.,log_positive_missing)
    conditional=conditional.masked_fill(~any_positive & ~possible_missing,-torch.inf)
    log_active,log_inactive=F.logsigmoid(active_logit),F.logsigmoid(-active_logit)
    result=torch.logaddexp(log_active+conditional,log_inactive.masked_fill(any_positive,-torch.inf))
    if mode=='reported_sleep':
        result=torch.where(asleep[:,None],torch.zeros_like(result),result)
    return result


class SpendingModel(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(width,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU())
        self.prior=nn.Linear(32,2)
        self.params=nn.Linear(32,32)

    def forward(self,x,observed,paid,amount,asleep,mode):
        h=self.body(x)
        params=self.params(h).reshape(len(x),2,16)
        mask=pattern_log_likelihood(params[:,:,:5],params[:,:,5],observed[:,None,:],paid[:,None,:],mode,asleep)
        mean=params[:,:,6:11]
        scale=.25+F.softplus(params[:,:,11:16])
        normal=-torch.log(scale)-.5*((amount[:,None,:]-mean)/scale)**2-.5*np.log(2*np.pi)
        return F.log_softmax(self.prior(h),dim=1)+mask+(normal*paid[:,None,:]).sum(2)


def fit(x,bills,asleep,y,t,q,mode,seed,measure=None):
    torch.manual_seed(seed)
    gen=np.random.default_rng(seed)
    z=np.where(bills>0,np.log1p(np.maximum(bills,0)),np.nan)
    amounts=np.nan_to_num((z-np.nanmean(z[t],axis=0))/np.nanstd(z[t],axis=0).clip(.1,None))
    context=torch.tensor((x-x[t].mean(0))/x[t].std(0).clip(.25,None),dtype=torch.float32)
    args=(context,torch.tensor(np.isfinite(bills)),torch.tensor(bills>0),
          torch.tensor(amounts,dtype=torch.float32),torch.tensor(asleep))
    truth=torch.tensor(y,dtype=torch.long)
    model=SpendingModel(x.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.02)
    snapshots=[]
    for epoch in range(48):
        for ids in np.array_split(gen.permutation(t),ceil(len(t)/384)):
            joint=model(*(a[ids] for a in args),mode)
            observed=joint.gather(1,truth[ids,None]).mean()
            loss=-observed
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (31,39,47):
            with torch.inference_mode():
                query=[a[q] for a in args]
                if measure is None:
                    value=torch.softmax(model(*query,mode),dim=1)[:,1]
                else:
                    value=measure(model,query,mode)
                snapshots.append(value.numpy())
    p=np.mean(snapshots,axis=0)
    assert np.isfinite(p).all()
    return p


def run():
    _,y,g,rows,oldfold,_,champ,_,_,_,_,old,_=aligned()
    raw,labels,groups,fold=data()
    names,half,full=banks(raw,labels,fold)
    ref=latest_reference(labels,groups,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,labels,groups,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    x,bills,sleep=inputs(raw)
    paid=(bills>0).any(1)
    assert not (sleep & paid).any()
    shared=((full>=.5)!=labels[:,None]).all(1)
    hard=shared[rows] & (champ!=y)
    unseen=~np.isin(groups,g)
    awake_zero=(raw.CryoSleep.eq(False).to_numpy() & np.isfinite(bills).all(1) & ~paid)
    unknown_sleep=raw.CryoSleep.isna().to_numpy()
    path=ROOT/'artifacts/nov24_zero_truncated'
    path.mkdir(exist_ok=True)
    report={}
    for mode in ['independent','truncated','reported_sleep']:
        predictions=[]
        for tag,ix,splits in [('historical',rows,oldfold),('full',np.arange(len(raw)),fold)]:
            p=np.full(len(ix),np.nan)
            for f in np.unique(splits):
                q=ix[splits==f]
                t=ix[(splits!=f)&~np.isin(groups[ix],groups[q])]
                assert not np.isin(groups[t],groups[q]).any()
                p[splits==f]=fit(x,bills,sleep,labels,t,q,mode,1981+int(f))
            assert np.isfinite(p).all()
            predictions.append(p)
        a,b=predictions
        np.savez_compressed(path/f'{mode}.npz',rows=rows,oldfold=oldfold,fold=fold,passenger=raw.PassengerId.to_numpy(str),
                            historical=a,full=b)
        def stats(pred,base,truth,group,split,mask):
            d=((pred==truth).astype(int)-(base==truth))[mask]
            return dict(rows=int(mask.sum()),accuracy=float((pred[mask]==truth[mask]).mean()),fixed=int((d==1).sum()),
                   harmed=int((d==-1).sum()),fold_net=[int(d[split[mask]==f].sum()) for f in np.unique(split[mask])],
                   ci_pp=interval(d,group[mask]))
        result={}
        for key,weight in [('direct',1.),('quarter',.25)]:
            ao,bo=a>=.5,b>=.5
            if weight<1:
                ao=np.where(champ==(old[:,0]>=.5),((1-weight)*old[:,0]+weight*a)>=.5,champ)
                bo=((1-weight)*gate+weight*b)>=.5
            result[key]=dict(historical=stats(ao,champ,y,g,oldfold,np.ones(len(y),bool)),
                full=stats(bo,ref,labels,groups,fold,np.ones(len(labels),bool)),
                unseen=stats(bo,ref,labels,groups,fold,unseen),awake_zero=stats(bo,ref,labels,groups,fold,awake_zero),
                unknown_sleep=stats(bo,ref,labels,groups,fold,unknown_sleep),
                shared_fixed=int((shared&(bo==labels)).sum()),stable_hard_fixed=int((hard&(ao==y)&(bo[rows]==y)).sum()))
        report[mode]=result
        print(mode,[(k,v['full']['fixed'],v['full']['harmed'],v['shared_fixed']) for k,v in result.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/zero_truncated_bills.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
