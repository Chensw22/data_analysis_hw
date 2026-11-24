"""Fit class-conditional normalizing flows only on the observed positive expenditure dimensions."""
import json
import math

import numpy as np
import torch
from torch import nn
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_two_view_reliability_n24 import design,tree


class BillDensity(nn.Module):
    def __init__(self,width,dependent):
        super().__init__()
        self.dependent = dependent
        count = 4 if dependent else 1
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(width+6,36),nn.SiLU(),nn.Linear(36,10)) for _ in range(count))
        for net in self.layers:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def log_density(self,amount,active,context,label):
        z = amount
        jacobian = torch.zeros_like(label)
        for k,net in enumerate(self.layers):
            fixed = ((torch.arange(5)%2)==(k%2)) if self.dependent else torch.zeros(5,dtype=torch.bool)
            transformed = (~fixed) & active
            inputs = torch.cat([context,label[:,None],z*fixed],dim=1)
            shift,log_scale = net(inputs).chunk(2,dim=1)
            scale = 1.5*torch.tanh(log_scale/1.5)
            z = torch.where(transformed,(z-shift)*torch.exp(-scale),z)
            jacobian = jacobian-(scale*transformed).sum(1)
        return -.5*((z.square()+math.log(2*math.pi))*active).sum(1)+jacobian


def inputs(raw,g):
    registration,_ = design(raw,g)
    bills = raw[BILLS].to_numpy(float)
    active = bills>0
    x = np.column_stack([np.nan_to_num(registration,nan=-1.),active,np.isnan(bills),raw.CryoSleep.eq(True),raw.CryoSleep.isna()])
    return x,np.log1p(np.nan_to_num(bills)),active


def estimate(arrays,y,g,t,q):
    context,amount,active = arrays
    assert not np.isin(g[t],g[q]).any()
    p = tree(19).fit(context[t],y[t]).predict_proba(context[q])[:,1]
    mean,std = zip(*[(amount[t,j][active[t,j]].mean(),amount[t,j][active[t,j]].std()) for j in range(5)])
    standardized = np.where(active,(amount-mean)/np.maximum(std,.1),0.)
    x = torch.tensor(((context-context[t].mean(0))/context[t].std(0).clip(.2)).clip(-7,7),dtype=torch.float32)
    z = torch.tensor(standardized,dtype=torch.float32)
    mask = torch.tensor(active)
    labels = torch.tensor(y,dtype=torch.float32)
    # With no positive services the amount likelihood is one under both classes.
    scores = [p]
    for dependent in (False,True):
        torch.manual_seed(224)
        rng = np.random.default_rng(219)
        density = BillDensity(x.shape[1],dependent)
        opt = torch.optim.AdamW(density.parameters(),lr=.002,weight_decay=.01)
        selected = t[active[t].any(1)]
        for _ in range(24):
            for batch in np.array_split(rng.permutation(selected),int(np.ceil(len(selected)/256))):
                opt.zero_grad()
                loss = -density.log_density(z[batch],mask[batch],x[batch],labels[batch]).mean()
                loss.backward()
                opt.step()
        with torch.inference_mode():
            ll = [density.log_density(z[q],mask[q],x[q],torch.full((len(q),),c)).numpy() for c in (0.,1.)]
        odds = ll[1]-ll[0]
        assert np.isfinite(odds).all() and np.all(odds[~active[q].any(1)]==0)
        scores.append(expit(logit(p.clip(.0001,.9999))+odds))
    return np.column_stack(scores)


def check_jacobian():
    torch.manual_seed(181)
    m = BillDensity(3,True).double()
    for net in m.layers:
        nn.init.normal_(net[-1].weight,std=.1)
    x = torch.randn(1,3,dtype=torch.float64)
    flag = torch.tensor([[True,False,True,True,False]])
    state = torch.randn(3,dtype=torch.float64)
    y = torch.ones(1,dtype=torch.float64)
    def transform(v):
        z = torch.zeros(1,5,dtype=torch.float64).masked_scatter(flag,v)
        for k,layer in enumerate(m.layers):
            fixed = (torch.arange(5)%2)==k%2
            b,l = layer(torch.cat([x,y[:,None],z*fixed],dim=1)).chunk(2,dim=1)
            z = torch.where(flag & ~fixed,(z-b)*torch.exp(-1.5*torch.tanh(l/1.5)),z)
        return z[flag]
    jac = torch.autograd.functional.jacobian(transform,state)
    p = -.5*(transform(state).square()+math.log(2*math.pi)).sum()+torch.linalg.slogdet(jac).logabsdet
    z = torch.zeros(1,5,dtype=torch.float64).masked_scatter(flag,state)
    torch.testing.assert_close(m.log_density(z,flag,x,y)[0],p)


def run():
    check_jacobian()
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    modes = ('registration_state','affine','flow')
    observations = inputs(raw,g)
    report = {}
    for (tag,ix,f,ref,p),(six,arch) in zip([('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                                              ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        v = protocol(observations,y,g,ix,f,tag,estimate,modes,'conditional_bill_flow')
        missed = (ref!=y[ix]) & ((six>=.5)==ref[:,None]).all(1)
        all_missed = missed & ((arch>=.5)==ref[:,None]).all(1)
        report[tag] = {}
        for j,name in enumerate(modes):
            for w in (1.,.25):
                pred = v[:,j]>=.5 if w==1 else np.where((p>=.5)==ref,(1-w)*p+w*v[:,j]>=.5,ref)
                r = stats(pred,ref,y[ix],g[ix],f)
                r.update(six_fixed=int((missed & (pred==y[ix])).sum()),archive_fixed=int((all_missed & (pred==y[ix])).sum()),
                         hard_fixed=int((hard[ix] & (ref!=y[ix]) & (pred==y[ix])).sum()))
                if tag=='full':
                    new = ~np.isin(g[ix],og)
                    r['new_net']=int((pred[new]==y[ix][new]).sum()-(ref[new]==y[ix][new]).sum())
                report[tag][f'{name}_{w}']=r
                print(tag,name,w,r['corrected'],r['harmed'],r['archive_fixed'],r['hard_fixed'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/conditional_bill_flow.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run()
